import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
from typing import Optional, Tuple, Union

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from munch import munchify
from colorama import Fore, Style
from torch.multiprocessing import Lock
from scipy.ndimage import binary_erosion
from multiprocessing.connection import Connection
import torch.multiprocessing as mp

from thirdparty.gaussian_splatting.utils.image_utils import psnr
from thirdparty.gaussian_splatting.utils.system_utils import mkdir_p
from thirdparty.gaussian_splatting.gaussian_renderer import render
from thirdparty.gaussian_splatting.utils.general_utils import (
    rotation_matrix_to_quaternion,
    quaternion_multiply,
    multiply_quaternions,
    quaternion_to_matrix_optimized,
)
from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel
from thirdparty.gaussian_splatting.utils.graphics_utils import (
    getProjectionMatrix2,
    getWorld2View2,
)
from src.depth_video import DepthVideo
from src.utils.datasets import get_dataset, load_metric_depth, load_img_feature
from src.utils.common import as_intrinsics_matrix, setup_seed
from src.utils.Printer import Printer, FontColor
from src.utils.pose_utils import update_pose
from src.utils.slam_utils import (
    get_loss_mapping,
    get_loss_mapping_uncertainty,
    get_loss_tracking,
)
from src.utils.camera_utils import Camera
from src.utils.dyn_uncertainty import mapping_utils as map_utils
from src.utils.dyn_uncertainty.median_filter import MedianPool2d
from src.utils.plot_utils import create_gif_from_directory
from src.gui import gui_utils

from src.utils.Spline import SplineN_linear_qt

from eval_utils.dycheck_metrics import compute_lpips
from lib_mosca.gs_utils.ssim_helper import ssim

@torch.jit.script
def quaternion_to_matrix_optimized(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices (truly optimized version).

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    if quaternions.dtype != torch.float32 and quaternions.dtype != torch.float64:
        quaternions = quaternions.float()
    
    r, i, j, k = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    
    two_s = 2.0 / (r*r + i*i + j*j + k*k)
    
    ii, jj, kk = i*i, j*j, k*k
    ij, ik, jk, ir, jr, kr = i*j, i*k, j*k, i*r, j*r, k*r
    
    return torch.stack((
        1 - two_s * (jj + kk),  # m00
        two_s * (ij - kr),      # m01
        two_s * (ik + jr),      # m02
        two_s * (ij + kr),      # m10
        1 - two_s * (ii + kk),  # m11
        two_s * (jk - ir),      # m12
        two_s * (ik - jr),      # m20
        two_s * (jk + ir),      # m21
        1 - two_s * (ii + jj),  # m22
    ), -1).reshape(quaternions.shape[:-1] + (3, 3))

def rotation_matrix_to_axis_angle(R, device="cpu"):
    angle = torch.acos((torch.trace(R) - 1) / 2)
    if angle < 1e-6:
        return torch.zeros(3, device=device)
    rx = R[2,1] - R[1,2]
    ry = R[0,2] - R[2,0]
    rz = R[1,0] - R[0,1]
    axis = torch.tensor([rx, ry, rz], device=device)
    axis = axis / (2 * torch.sin(angle))
    return axis * angle

def identity_regularization(rot, trans, lambda_rot=0.1, lambda_trans=0.1):
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=rot.device)
    rot_loss = torch.mean((rot - identity_quat) ** 2)
    
    trans_loss = torch.mean(trans ** 2)
    
    return lambda_rot * rot_loss + lambda_trans * trans_loss


class Mapper(object):
    """
    Mapper thread.

    """

    def __init__(
        self, slam, pipe: Connection, uncer_network: Optional[nn.Module] = None, 
        q_main2vis: Optional[mp.Queue] = None, q_vis2main: Optional[mp.Queue] = None
    ):
        # setup seed
        setup_seed(slam.cfg["setup_seed"])
        # torch.autograd.set_detect_anomaly(True)

        self.config = slam.cfg
        self.printer: Printer = slam.printer
        self.pipe = pipe
        self.verbose = slam.verbose
        self.device = torch.device(self.config["device"])
        self.video: DepthVideo = slam.video

        # Set gaussian model
        self.model_params = munchify(self.config["mapping"]["model_params"])
        self.opt_params = munchify(self.config["mapping"]["opt_params"])
        self.pipeline_params = munchify(self.config["mapping"]["pipeline_params"])
        use_spherical_harmonics = self.config["mapping"]["Training"][
            "spherical_harmonics"
        ]
        self.model_params.sh_degree = 3 if use_spherical_harmonics else 0
        self.gaussians = GaussianModel(self.model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        self.gaussians.training_setup(self.opt_params)

        # Set background color
        bg_color = [0, 0, 0]
        self.background = torch.tensor(
            bg_color, dtype=torch.float32, device=self.device
        )

        # Set hyperparams
        self._set_hyperparams()

        # Set frame reader (where we get the input dataset)
        self.frame_reader = get_dataset(self.config, device=self.device)
        self.intrinsics = as_intrinsics_matrix(self.frame_reader.get_intrinsic()).to(
            self.device
        )

        # Prepare projection matrix
        if self.config["mapping"]["full_resolution"]:
            intrinsic_full = self.frame_reader.get_intrinsic_full_resol()
            fx, fy, cx, cy = [intrinsic_full[i].item() for i in range(4)]
            W_out, H_out = self.frame_reader.W_out_full, self.frame_reader.H_out_full
        else:
            fx, fy, cx, cy = (
                self.frame_reader.fx,
                self.frame_reader.fy,
                self.frame_reader.cx,
                self.frame_reader.cy,
            )
            W_out, H_out = self.frame_reader.W_out, self.frame_reader.H_out

        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            W=W_out,
            H=H_out,
        ).transpose(0, 1)
        self.projection_matrix = projection_matrix.to(device=self.device)

        # Setup for uncertainty-aware mapping
        self.vis_uncertainty_online = False
        self.uncer_params = munchify(self.config["mapping"]["uncertainty_params"])
        self.uncertainty_aware = self.uncer_params["activate"]
        if self.uncertainty_aware:
            self.uncer_network = uncer_network
            self.uncer_optimizer = torch.optim.Adam(
                self.uncer_network.parameters(),
                lr=self.uncer_params["lr"],
                weight_decay=self.uncer_params["weight_decay"],
            )

            self.vis_uncertainty_online = self.uncer_params["vis_uncertainty_online"]

        # Setup queue object for gui communication
        self.q_main2vis = q_main2vis
        self.q_vis2main = q_vis2main
        self.pause = False

        # self.masker = Masker()

        self.sample_flag = False
        self.t_s = self.config["is_blur"]
        self.r_s = self.config["is_blur"]

        self.lr_opt = 0.0003
        self.lr_opt_get_all_nodes = 0.0003
        self.fix_num_poses = self.config["fix_num_poses"]
        self.is_blur = self.config["is_blur"]
        self.is_exposure = self.config["is_exposure"]
        self.test_training_time_mode = self.config["test_training_time_mode"]

    def run(self):
        """
        Trigger mapping process, get estimated pose and depth from tracking process,
        send continue signal to tracking process when the mapping of the current frame finishes.
        """
        # Initialize list to keep track of Keyframes
        # In short, for any idx "i",
        # self.video.timestamp[video_idx[i]] = self.frame_idxs[i]
        self.frame_idxs = []  # the indices of keyframes in the original frame sequence
        self.video_idxs = []  # keyframe numbering (I sometimes call it kf_idx)

        while True:
            if self.config['gui']:
                if self.q_vis2main.empty():
                    if self.pause:
                        continue
                else:
                    data_vis2main = self.q_vis2main.get()
                    self.pause = data_vis2main.flag_pause
                    if self.pause:
                        self.printer.print("You have paused the process", FontColor.MAPPER)
                        continue
                    else:
                        self.printer.print("You have resume the process", FontColor.MAPPER)

            frame_info = self.pipe.recv()
            frame_idx, video_idx = frame_info["timestamp"], frame_info["video_idx"]
            is_init, is_finished = frame_info["just_initialized"], frame_info["end"]

            if is_finished:
                self.printer.print("Done with Mapping and Tracking", FontColor.MAPPER)
                break

            if self.verbose:
                self.printer.print(f"\nMapping Frame {frame_idx} ...", FontColor.MAPPER)

            if is_init:
                self.printer.print("Initializing the mapping", FontColor.MAPPER)
                self.initialize_mapper(video_idx)
                self.pipe.send("continue")
                continue

            viewpoint, invalid = self._get_viewpoint(video_idx, frame_idx)
            if self.is_blur:
                with torch.no_grad():
                    device = viewpoint.device
                    low, high = 0.001, 0.003

                    viewpoint_last = self.cameras[video_idx-1]

                    R_rel = viewpoint.R @ (viewpoint_last.R.float()).T
                    T_rel = viewpoint.T - (viewpoint_last.T.float())

                    rot_axis_angle = rotation_matrix_to_axis_angle(R_rel, device=device)
                    rot_axis_angle = rot_axis_angle / (torch.norm(rot_axis_angle) + 1e-8)

                    trans_direction = T_rel / (torch.norm(T_rel) + 1e-8)

                    trans_magnitude = (high - low) * torch.rand(1) + low
                    rand1_trans = trans_direction * trans_magnitude.to(device)
                    rand1_trans = rand1_trans.to(device)
                    rand2_trans = -rand1_trans

                    rot_magnitude = (high - low) * torch.rand(1) + low
                    rand1_rot = rot_axis_angle * rot_magnitude.to(device)
                    rand1_rot = rand1_rot.to(device)
                    rand2_rot = -rand1_rot

                    perturbation1_rot = torch.cat([
                        torch.ones_like(rand1_rot[..., :1]),
                        rand1_rot
                    ], dim=-1).to(device)

                    perturbation2_rot = torch.cat([
                        torch.ones_like(rand2_rot[..., :1]),
                        rand2_rot
                    ], dim=-1).to(device)

                    viewpoint.opt_rot_start.data = multiply_quaternions(
                        viewpoint.opt_rot_start.data.unsqueeze(0),
                        perturbation1_rot.unsqueeze(0)
                    ).squeeze(0)
                    viewpoint.opt_trans_start.data = viewpoint.opt_trans_start.data + rand1_trans

                    viewpoint.opt_rot_end.data = multiply_quaternions(
                        viewpoint.opt_rot_end.data.unsqueeze(0),
                        perturbation2_rot.unsqueeze(0)
                    ).squeeze(0)
                    viewpoint.opt_trans_end.data = viewpoint.opt_trans_end.data + rand2_trans
                    
                    ### exp use this 

                    # device = viewpoint.device 
                    # low, high = 0.001, 0.003
                    # rand1_trans = (high - low) * torch.rand(3) + low
                    # rand1_trans = rand1_trans.to(device)
                    
                    # rand2_trans = -rand1_trans
    
                    # rand1_rot = (high - low) * torch.rand(3) + low
                    # rand1_rot = rand1_rot.to(device)
                    # perturbation1_rot = torch.cat([
                    #     torch.ones_like(rand1_rot[..., :1]),
                    #     rand1_rot
                    # ], dim=-1).to(device)
                    
                    # rand2_rot = -rand1_rot
                    # perturbation2_rot = torch.cat([
                    #     torch.ones_like(rand2_rot[..., :1]),
                    #     rand2_rot
                    # ], dim=-1).to(device)
                    
                    # viewpoint.opt_rot_start.data = multiply_quaternions(viewpoint.opt_rot_start.data.unsqueeze(0), perturbation1_rot.unsqueeze(0)).squeeze(0)
                    # viewpoint.opt_trans_start.data=viewpoint.opt_trans_start.data + rand1_trans
    
                    # viewpoint.opt_rot_end.data = multiply_quaternions(viewpoint.opt_rot_end.data.unsqueeze(0), perturbation2_rot.unsqueeze(0)).squeeze(0)
                    # viewpoint.opt_trans_end.data = viewpoint.opt_trans_end.data + rand2_trans

                    ###

            if invalid:
                # Only happens when not using metric depth for tracking regularization
                self.printer.print("WARNING: Too few valid pixels from droid depth", FontColor.MAPPER)
                self.is_kf[video_idx] = False
                self.pipe.send("continue")
                continue  # too few valid pixels from droid depth
            
            # Update the map if depth/pose of any keyframe has been updated
            self._update_keyframes_from_frontend()
            self.frame_idxs.append(frame_idx)
            self.video_idxs.append(video_idx)

            # We need to render from the current pose to obtain the "n_touched" variable
            # which is used later on
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            curr_visibility = (render_pkg["n_touched"] > 0).long()

            # Always create kf
            self.cameras[video_idx] = viewpoint
            self.current_window, _ = self._add_to_window(
                video_idx,
                curr_visibility,
                self.occ_aware_visibility,
                self.current_window,
            )
            self.is_kf[video_idx] = True
            self.depth_dict[video_idx] = torch.tensor(viewpoint.depth).to(self.device)
            self.frame_count_log[video_idx] = 0

            self.gaussians.extend_from_pcd_seq(
                viewpoint, kf_id=video_idx, init=False, depthmap=viewpoint.depth
            )

            opt_params = []
            
            for cam_idx in range(len(self.current_window)):
                viewpoint = self.cameras[self.current_window[cam_idx]]

                opt_params.append(
                    {
                        "params": [viewpoint.opt_rot_start],
                        "lr": self.lr_opt,
                        "name": "opt_rot_start_{}".format(viewpoint.uid),
                    }
                )
                opt_params.append(
                    {
                        "params": [viewpoint.opt_trans_start],
                        "lr": self.lr_opt,
                        "name": "opt_trans_start_{}".format(viewpoint.uid),
                    }
                )
                opt_params.append(
                    {
                        "params": [viewpoint.opt_rot_end],
                        "lr": self.lr_opt,
                        "name": "opt_rot_end_{}".format(viewpoint.uid),
                    }
                )
                opt_params.append(
                    {
                        "params": [viewpoint.opt_trans_end],
                        "lr": self.lr_opt,
                        "name": "opt_trans_end_{}".format(viewpoint.uid),
                    }
                )
                    

                if self.current_window[cam_idx] == 0 or self.is_exposure == False:
                    # Do not add first frame for exposure optimization
                    continue
                opt_params.append(
                    {
                        "params": [viewpoint.exposure_a],
                        "lr": 0.01,
                        "name": "exposure_a_{}".format(viewpoint.uid),
                    }
                )
                opt_params.append(
                    {
                        "params": [viewpoint.exposure_b],
                        "lr": 0.01,
                        "name": "exposure_b_{}".format(viewpoint.uid),
                    }
                )
            self.keyframe_optimizers = torch.optim.Adam(opt_params)

            viewpoint = self.cameras[video_idx]
            with torch.no_grad():
                viewpoint.exposure_a.data = self.cameras[video_idx - 1].exposure_a.data.detach().clone().requires_grad_(True)
                viewpoint.exposure_b.data = self.cameras[video_idx - 1].exposure_b.data.detach().clone().requires_grad_(True)

            with Lock():
                if self.config['fast_mode']:
                    # We are in fast mode,
                    # update map and uncertainty MLP every 4 key frames
                    if video_idx % 4 == 0:
                        gaussian_split = self.map_opt_online(
                            self.current_window, iters=self.mapping_itr_num
                        )
                    else:
                        self._update_occ_aware_visibility(self.current_window)
                else:
                    gaussian_split = self.map_opt_online(
                        self.current_window, iters=self.mapping_itr_num
                    )

                if gaussian_split:
                    # do one more iteration after densify and prune
                    self.map_opt_online(self.current_window, iters=1)

            torch.cuda.empty_cache()

            if self.config['gui']:
                self._send_to_gui(video_idx)

            self.pipe.send("continue")

    """
    Utility functions
    """

    def _set_hyperparams(self):
        self.cameras_extent = 6.0
        mapping_config = self.config["mapping"]

        self.init_itr_num = mapping_config["Training"]["init_itr_num"]
        self.init_gaussian_update = mapping_config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = mapping_config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = mapping_config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * mapping_config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = mapping_config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = mapping_config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = mapping_config["Training"][
            "gaussian_update_offset"
        ]
        self.gaussian_th = mapping_config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * mapping_config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = mapping_config["Training"]["gaussian_reset"]
        self.size_threshold = mapping_config["Training"]["size_threshold"]
        self.window_size = mapping_config["Training"]["window_size"]

        self.save_dir = self.config["data"]["output"] + "/" + self.config["scene"]

        self.deform_gaussians = self.config["mapping"]["deform_gaussians"]
        self.online_plotting = self.config["mapping"]["online_plotting"]

    def _get_viewpoint(self, video_idx: int, frame_idx: int) -> Tuple[Camera, bool]:
        """
        Create and initialize a Camera object for a given frame.

        Args:
            video_idx (int): Index in the video class (keyframe idx).
            frame_idx (int): Index in the original frame sequences.

        Returns:
            Tuple[Camera, bool]: Initialized Camera object and an invalid flag.
        """
        # Load color image based on resolution configuration
        if self.config["mapping"]["full_resolution"]:
            color = (
                self.frame_reader.get_color_full_resol(frame_idx)
                .to(self.device)
                .squeeze()
            )
            load_feature_suffix = "full"
        else:
            color = self.frame_reader.get_color(frame_idx).to(self.device).squeeze()
            load_feature_suffix = ""

        # Load metric depth
        metric_depth = load_metric_depth(frame_idx, self.save_dir).to(self.device)

        # Load features if uncertainty-aware
        if self.uncertainty_aware:
            features = load_img_feature(
                frame_idx, self.save_dir, suffix=load_feature_suffix
            ).to(self.device)
        else:
            features = None

        # Get estimated depth and camera pose
        est_depth, est_w2c, invalid = self.get_w2c_and_depth(
            video_idx, frame_idx, metric_depth
        )

        # Prepare data dictionary for Camera initialization
        camera_data = {
            "idx": video_idx,
            "gt_color": color,
            "est_depth": est_depth.cpu().numpy(),
            "est_pose": est_w2c,
            "features": features,
        }

        # Initialize Camera object
        viewpoint = Camera.init_from_dataset(
            self.frame_reader,
            camera_data,
            self.projection_matrix,
            full_resol=self.config["mapping"]["full_resolution"],
        )

        # Update camera pose and compute gradient mask
        # The Camera class is based on MonoGS and
        # init_from_dataset function only updates the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
        viewpoint.compute_grad_mask(self.config)

        return viewpoint, invalid

    def _update_keyframes_from_frontend(self):
        """
        Update keyframe information based on the latest frontend data.
        This includes updating camera poses, depths, and mapping points (deform gaussians)
        for all keyframes.
        """
        for keyframe_idx, frame_idx in zip(self.video_idxs, self.frame_idxs):
            # Get updated pose and depth
            if self.video.metric_depth_reg:
                c2w_updated = self.video.get_pose(keyframe_idx, self.device)
                w2c_updated = torch.linalg.inv(c2w_updated)
                depth_updated = None
                invalid = False
            else:
                metric_depth = load_metric_depth(frame_idx, self.save_dir).to(
                    self.device
                )
                depth_updated, w2c_updated, invalid = self.get_w2c_and_depth(
                    keyframe_idx, frame_idx, metric_depth
                )

            # Get old pose
            w2c_old = torch.eye(4, device=self.device)
            w2c_old[:3, :3] = self.cameras[keyframe_idx].R
            w2c_old[:3, 3] = self.cameras[keyframe_idx].T

            pose_unchanged = torch.allclose(w2c_old, w2c_updated, atol=1e-6)
            if pose_unchanged and depth_updated is None:
                continue

            # Update camera
            self.cameras[keyframe_idx].update_RT(
                w2c_updated[:3, :3], w2c_updated[:3, 3]
            )
            if depth_updated is not None:
                self.cameras[keyframe_idx].depth = depth_updated.cpu().numpy()
                self.depth_dict[keyframe_idx] = depth_updated

            # Update viewpoint if it exists
            if self.is_kf[keyframe_idx]:
                self.cameras[keyframe_idx].update_RT(
                    w2c_updated[:3, :3], w2c_updated[:3, 3]
                )
                if depth_updated is not None:
                    self.cameras[keyframe_idx].depth = depth_updated.cpu().numpy()

            # Update mapping parameters
            if self.deform_gaussians and self.is_kf[keyframe_idx]:
                if invalid or depth_updated is None:
                    self._update_mapping_points(
                        keyframe_idx,
                        w2c_updated,
                        w2c_old,
                        depth_updated,
                        self.depth_dict[keyframe_idx],
                        method="rigid",
                    )
                else:
                    self._update_mapping_points(
                        keyframe_idx,
                        w2c_updated,
                        w2c_old,
                        depth_updated,
                        self.depth_dict[keyframe_idx],
                    )

    def _update_mapping_points(
        self, frame_idx, w2c, w2c_old, depth, depth_old, method=None
    ):
        """Refer to splat-slam"""
        if method == "rigid":
            # just move the points according to their SE(3) transformation without updating depth
            frame_idxs = (
                self.gaussians.unique_kfIDs
            )  # idx which anchored the set of points
            frame_mask = frame_idxs == frame_idx  # global variable
            if frame_mask.sum() == 0:
                return
            # Retrieve current set of points to be deformed
            # But first we need to retrieve all mean locations and clone them
            means = self.gaussians.get_xyz.detach()
            # Then move the points to their new location according to the new pose
            # The global transformation can be computed by composing the old pose
            # with the new pose
            transformation = torch.linalg.inv(torch.linalg.inv(w2c_old) @ w2c)
            pix_ones = torch.ones(frame_mask.sum(), 1).cuda().float()
            pts4 = torch.cat((means[frame_mask], pix_ones), dim=1)
            means[frame_mask] = (transformation @ pts4.T).T[:, :3]
            # put the new means back to the optimizer
            self.gaussians._xyz = self.gaussians.replace_tensor_to_optimizer(
                means, "xyz"
            )["xyz"]
            # transform the corresponding rotation matrices
            rots = self.gaussians.get_rotation.detach()
            # Convert transformation to quaternion
            transformation = rotation_matrix_to_quaternion(transformation.unsqueeze(0))
            rots[frame_mask] = quaternion_multiply(
                transformation.expand_as(rots[frame_mask]), rots[frame_mask]
            )

            with torch.no_grad():
                self.gaussians._rotation = self.gaussians.replace_tensor_to_optimizer(
                    rots, "rotation"
                )["rotation"]
        else:
            # Update pose and depth by projecting points into the pixel space to find updated correspondences.
            # This strategy also adjusts the scale of the gaussians to account for the distance change from the camera
            depth = depth.to(self.device)
            frame_idxs = (
                self.gaussians.unique_kfIDs
            )  # idx which anchored the set of points
            frame_mask = frame_idxs == frame_idx  # global variable
            if frame_mask.sum() == 0:
                return

            # Retrieve current set of points to be deformed
            means = self.gaussians.get_xyz.detach()[frame_mask]

            # Project the current means into the old camera to get the pixel locations
            pix_ones = torch.ones(means.shape[0], 1).cuda().float()
            pts4 = torch.cat((means, pix_ones), dim=1)
            pixel_locations = (self.intrinsics @ (w2c_old @ pts4.T)[:3, :]).T
            pixel_locations[:, 0] /= pixel_locations[:, 2]
            pixel_locations[:, 1] /= pixel_locations[:, 2]
            pixel_locations = pixel_locations[:, :2].long()
            height, width = depth.shape
            # Some pixels may project outside the viewing frustum.
            # Assign these pixels the depth of the closest border pixel
            pixel_locations[:, 0] = torch.clamp(
                pixel_locations[:, 0], min=0, max=width - 1
            )
            pixel_locations[:, 1] = torch.clamp(
                pixel_locations[:, 1], min=0, max=height - 1
            )

            # Extract the depth at those pixel locations from the new depth
            depth = depth[pixel_locations[:, 1], pixel_locations[:, 0]]
            depth_old = depth_old[pixel_locations[:, 1], pixel_locations[:, 0]]
            # Next, we can either move the points to the new pose and then adjust the
            # depth or the other way around.
            # Lets adjust the depth per point first
            # First we need to transform the global means into the old camera frame
            pix_ones = torch.ones(frame_mask.sum(), 1).cuda().float()
            pts4 = torch.cat((means, pix_ones), dim=1)
            means_cam = (w2c_old @ pts4.T).T[:, :3]

            rescale_scale = (1 + 1 / (means_cam[:, 2]) * (depth - depth_old)).unsqueeze(
                -1
            )  # shift
            # account for 0 depth values - then just do rigid deformation
            rigid_mask = torch.logical_or(depth == 0, depth_old == 0)
            rescale_scale[rigid_mask] = 1
            if (rescale_scale <= 0.0).sum() > 0:
                rescale_scale[rescale_scale <= 0.0] = 1

            rescale_mean = rescale_scale.repeat(1, 3)
            means_cam = rescale_mean * means_cam

            # Transform back means_cam to the world space
            pts4 = torch.cat((means_cam, pix_ones), dim=1)
            means = (torch.linalg.inv(w2c_old) @ pts4.T).T[:, :3]

            # Then move the points to their new location according to the new pose
            # The global transformation can be computed by composing the old pose
            # with the new pose
            transformation = torch.linalg.inv(torch.linalg.inv(w2c_old) @ w2c)
            pts4 = torch.cat((means, pix_ones), dim=1)
            means = (transformation @ pts4.T).T[:, :3]

            # reassign the new means of the frame mask to the self.gaussian object
            global_means = self.gaussians.get_xyz.detach()
            global_means[frame_mask] = means
            # print("mean nans: ", global_means.isnan().sum()/global_means.numel())
            self.gaussians._xyz = self.gaussians.replace_tensor_to_optimizer(
                global_means, "xyz"
            )["xyz"]

            # update the rotation of the gaussians
            rots = self.gaussians.get_rotation.detach()
            # Convert transformation to quaternion
            transformation = rotation_matrix_to_quaternion(transformation.unsqueeze(0))
            rots[frame_mask] = quaternion_multiply(
                transformation.expand_as(rots[frame_mask]), rots[frame_mask]
            )
            self.gaussians._rotation = self.gaussians.replace_tensor_to_optimizer(
                rots, "rotation"
            )["rotation"]

            # Update the scale of the Gaussians
            scales = self.gaussians._scaling.detach()
            scales[frame_mask] = scales[frame_mask] + torch.log(rescale_scale)
            self.gaussians._scaling = self.gaussians.replace_tensor_to_optimizer(
                scales, "scaling"
            )["scaling"]

    def _update_occ_aware_visibility(self, current_window):
        self.occ_aware_visibility = {}
        for kf_idx in current_window:
            viewpoint = self.cameras[kf_idx]
            render_pkg = render(
                viewpoint,
                self.gaussians,
                self.pipeline_params,
                self.background,
            )
            self.occ_aware_visibility[kf_idx] = (
                render_pkg["n_touched"] > 0
            ).long()

    def get_w2c_and_depth(self, video_idx, idx, mono_depth, print_info=False):
        est_frontend_depth, valid_depth_mask, c2w = self.video.get_depth_and_pose(
            video_idx, self.device
        )
        c2w = c2w.to(self.device)
        w2c = torch.linalg.inv(c2w)

        if self.video.metric_depth_reg: # True
            return est_frontend_depth, w2c, False

        # The following is only useful when no metric depth used for tracking regularization
        # Code is from Splat-SLAM
        if print_info:
            self.printer.print(
                f"valid depth number: {valid_depth_mask.sum().item()}, "
                f"valid depth ratio: {(valid_depth_mask.sum()/(valid_depth_mask.shape[0]*valid_depth_mask.shape[1])).item()}",
                FontColor.MAPPER
            )

        if valid_depth_mask.sum() < 100:
            invalid = True
            self.printer.print(
                f"Skip mapping frame {idx} at video idx {video_idx} because of not enough valid depth ({valid_depth_mask.sum()}).", FontColor.MAPPER
            )
        else:
            invalid = False

        est_frontend_depth[~valid_depth_mask] = 0
        if not invalid:
            mono_depth[mono_depth > 4 * mono_depth.mean()] = 0
            mono_depth = mono_depth.cpu().numpy()
            binary_image = (mono_depth > 0).astype(int)
            # Add padding around the binary_image to protect the borders
            iterations = 5
            padded_binary_image = np.pad(
                binary_image, pad_width=iterations, mode="constant", constant_values=1
            )
            structure = np.ones((3, 3), dtype=int)
            # Apply binary erosion with padding
            eroded_padded_image = binary_erosion(
                padded_binary_image, structure=structure, iterations=iterations
            )
            # Remove padding after erosion
            eroded_image = eroded_padded_image[
                iterations:-iterations, iterations:-iterations
            ]
            # set mono depth to zero at mask
            mono_depth[eroded_image == 0] = 0

            if (mono_depth == 0).sum() > 0:
                mono_depth = torch.from_numpy(
                    cv2.inpaint(
                        mono_depth,
                        (mono_depth == 0).astype(np.uint8),
                        inpaintRadius=3,
                        flags=cv2.INPAINT_NS,
                    )
                ).to(self.device)
            else:
                mono_depth = torch.from_numpy(mono_depth).to(self.device)

            valid_mask = (
                torch.from_numpy(eroded_image).to(self.device) * valid_depth_mask
            )  # new

            cur_wq = self.video.get_depth_scale_and_shift(
                video_idx, mono_depth, est_frontend_depth, valid_mask
            )
            mono_depth_wq = mono_depth * cur_wq[0] + cur_wq[1]

            est_frontend_depth[~valid_depth_mask] = mono_depth_wq[~valid_depth_mask]

        return est_frontend_depth, w2c, invalid

    def _add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        """Refer to MonoGS"""
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["mapping"]["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["mapping"]["Training"]
                else 0.4
            )
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.window_size:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame
    
    def _send_to_gui(self, video_idx):
        """Send data to the GUI for visualization.
        """
        viewpoint = self.cameras[video_idx]
        keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]
        uncertainty_map = self.get_viewpoint_uncertainty_no_grad(viewpoint)
        uncertainty_map = uncertainty_map.cpu().squeeze(0).numpy()
        
        current_window_dict = {}
        current_window_dict[self.current_window[0]] = self.current_window[1:]
        keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]
        self.q_main2vis.put(
            gui_utils.GaussianPacket(
                current_frame=viewpoint,
                gaussians=self.gaussians,
                gtcolor=viewpoint.original_image.squeeze(),
                gtdepth=viewpoint.depth,
                keyframes=keyframes,
                kf_window=current_window_dict,
                uncertainty=uncertainty_map,
            )
        )


    def initialize_mapper(self, cur_video_idx):
        self.printer.print("Resetting the mapper", FontColor.MAPPER)

        self.iteration_count = 0
        self.iterations_after_densify_or_reset = 0
        self.occ_aware_visibility = {}
        self.frame_count_log = {}
        self.current_window = []
        self.keyframe_optimizers = None

        # Keys are video_idx and value is boolean.
        # This is only useful in ablation study of no depth-regularization
        self.is_kf = {}

        # Create dictionary which stores the depth maps from the previous iteration
        # This depth is used during map deformation if we have missing pixels
        self.depth_dict = {}

        # Dictionary of Camera objects at the frame index
        # self.cameras contains all cameras.
        self.cameras = {}

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)

        opt_params = []

        for video_idx in range(cur_video_idx + 1):
            frame_idx = int(self.video.timestamp[video_idx])
            self.frame_idxs.append(frame_idx)
            self.video_idxs.append(video_idx)

            viewpoint, invalid = self._get_viewpoint(video_idx, frame_idx)
            if self.is_blur:
                with torch.no_grad():
                    device = viewpoint.device
                    low, high = 0.001, 0.003
                    rand1_trans = (high - low) * torch.rand(3) + low
                    rand1_trans = rand1_trans.to(device)
                    
                    rand2_trans = -rand1_trans
    
                    rand1_rot = (high - low) * torch.rand(3) + low
                    rand1_rot = rand1_rot.to(device)
                    perturbation1_rot = torch.cat([
                        torch.ones_like(rand1_rot[..., :1]),
                        rand1_rot
                    ], dim=-1).to(device)
                    
                    rand2_rot = -rand1_rot
                    perturbation2_rot = torch.cat([
                        torch.ones_like(rand2_rot[..., :1]),
                        rand2_rot
                    ], dim=-1).to(device)
                    
                    viewpoint.opt_rot_start.data = multiply_quaternions(viewpoint.opt_rot_start.data.unsqueeze(0), perturbation1_rot.unsqueeze(0)).squeeze(0)
                    viewpoint.opt_trans_start.data=viewpoint.opt_trans_start.data + rand1_trans
    
                    viewpoint.opt_rot_end.data = multiply_quaternions(viewpoint.opt_rot_end.data.unsqueeze(0), perturbation2_rot.unsqueeze(0)).squeeze(0)
                    viewpoint.opt_trans_end.data = viewpoint.opt_trans_end.data + rand2_trans
                
            self.cameras[video_idx] = viewpoint

            if invalid:
                # Only happens when not using metric depth for tracking regularization
                self.printer.print("WARNING: Too few valid pixels from droid depth", FontColor.MAPPER)
                self.is_kf[video_idx] = False
                continue  # too few valid pixels from droid depth

            # update the dictionaries
            self.depth_dict[video_idx] = torch.tensor(viewpoint.depth).to(self.device)
            self.is_kf[video_idx] = True
            self.frame_count_log[video_idx] = 0

            self.gaussians.extend_from_pcd_seq(
                viewpoint, kf_id=video_idx, init=True, depthmap=viewpoint.depth
            )

            self.current_window.append(video_idx)

            
            opt_params.append(
                {
                    "params": [viewpoint.opt_rot_start],
                    "lr": self.lr_opt,
                    "name": "opt_rot_start_{}".format(viewpoint.uid),
                }
            )
            opt_params.append(
                {
                    "params": [viewpoint.opt_trans_start],
                    "lr": self.lr_opt,
                    "name": "opt_trans_start_{}".format(viewpoint.uid),
                }
            )
            opt_params.append(
                {
                    "params": [viewpoint.opt_rot_end],
                    "lr": self.lr_opt,
                    "name": "opt_rot_end_{}".format(viewpoint.uid),
                }
            )
            opt_params.append(
                {
                    "params": [viewpoint.opt_trans_end],
                    "lr": self.lr_opt,
                    "name": "opt_trans_end_{}".format(viewpoint.uid),
                }
            )
            
            # Do not add first frame for exposure optimization
            if video_idx != 0 and self.is_exposure:
                opt_params.append(
                    {
                        "params": [viewpoint.exposure_a],
                        "lr": 0.01,
                        "name": "exposure_a_{}".format(viewpoint.uid),
                    }
                )
                opt_params.append(
                    {
                        "params": [viewpoint.exposure_b],
                        "lr": 0.01,
                        "name": "exposure_b_{}".format(viewpoint.uid),
                    }
                )
        self.keyframe_optimizers = torch.optim.Adam(opt_params)
        # self.keyframe_optimizers = torch.optim.SGD(opt_params)

        self.initialize_map_opt()
        # Only keep the recent <self.window_size> number of keyframes in the window
        self.current_window = self.current_window[-self.window_size :]

        if self.config['gui']:
            self._send_to_gui(cur_video_idx)

    def refine_pose_non_key_frame(
        self, frame_idx: int, w2c_init: torch.Tensor, features: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Optimize the pose for a non-key frame.

        Args:
            frame_idx (int): Camera index.
            w2c_init (torch.Tensor): Initial world-to-camera transformation.
            features (torch.Tensor, optional): Image features for uncertainty estimation.

        Returns:
            torch.Tensor: Refined world-to-camera transformation.
        """
        # We always use the downsampled image for tracking.
        # Full resolution only for the mapping even if it's activated.
        color = self.frame_reader.get_color(frame_idx).to(self.device).squeeze()

        data = {
            "idx": frame_idx,
            "gt_color": color.squeeze(),
            "est_depth": None,
            "est_pose": w2c_init,
            "features": features,
        }

        # Using uncertainty only when uncertainty-aware tracking is activated
        if self.video.uncertainty_aware:
            with torch.no_grad():
                uncer = self.uncer_network(features.to(color.device), frame_idx)
                uncer = torch.clip(uncer, min=0.1) + 1e-3
                uncer_resized = F.interpolate(
                    uncer.unsqueeze(0).unsqueeze(0),
                    size=color.shape[-2:],
                    mode="bilinear",
                ).squeeze()
                data_rate = 1 + 1 * map_utils.compute_bias_factor(
                    self.config["mapping"]["uncertainty_params"]["train_frac_fix"], 0.8
                )
                uncer_resized = (uncer_resized - 0.1) * data_rate + 0.1
        else:
            uncer_resized = None

        viewpoint = Camera.init_from_dataset(
            self.frame_reader,
            data,
            self.projection_matrix,
        )
        viewpoint.compute_grad_mask(self.config)
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        opt_params = [
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["mapping"]["Training"]["lr"]["cam_rot_delta"],
                "name": f"rot_{viewpoint.uid}",
            },
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["mapping"]["Training"]["lr"]["cam_trans_delta"],
                "name": f"trans_{viewpoint.uid}",
            },
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": f"exposure_a_{viewpoint.uid}",
            },
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": f"exposure_b_{viewpoint.uid}",
            },
        ]

        pose_optimizer = torch.optim.Adam(opt_params)
        tracking_itr_num = 100  # This number is taken from monoGS
        for _ in range(tracking_itr_num):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config["mapping"],
                image,
                depth,
                opacity,
                viewpoint,
                uncertainty=uncer_resized,
            )
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)

            if converged:
                break

        w2c_refined = torch.eye(4, device=viewpoint.R.device)
        w2c_refined[:3, :3] = viewpoint.R
        w2c_refined[:3, 3] = viewpoint.T

        return w2c_refined

    """
    Map Optimization functions (init, online, final_refine)
    """
    def initialize_map_opt(self):
        viewpoint_stack = []
        viewpoint_id_stack = []
        for kf_idx in self.current_window:
            viewpoint = self.cameras[kf_idx]
            viewpoint_stack.append(viewpoint)
            viewpoint_id_stack.append(kf_idx)
        
        for mapping_iteration in tqdm(range(self.init_itr_num)):
            self.iteration_count += 1
            self.iterations_after_densify_or_reset += 1
            # randomly select a viewpoint from the first K keyframes
            cam_idx = np.random.choice(range(len(viewpoint_stack)))
            viewpoint = viewpoint_stack[cam_idx]
            kf_idx = viewpoint_id_stack[cam_idx]

            rgb_stack = []

            if mapping_iteration > self.init_itr_num/2 and self.is_blur:
                rgb_stack, total_num_poses = self.render_blur(viewpoint)
                
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )

            rgb_stack.append(image)
            image = torch.mean(torch.stack(rgb_stack, dim=0), dim=0)

            if not self.uncertainty_aware:
                loss_init = get_loss_mapping(
                    self.config["mapping"],
                    image,
                    depth,
                    viewpoint,
                    opacity,
                    initialization=True,
                )
            else:
                train_frac = self.uncer_params["train_frac_fix"]
                ssim_frac = self.uncer_params["train_frac_fix"]
                if self.config["mapping"]["full_resolution"]:
                    depth = F.interpolate(
                        depth.unsqueeze(0), viewpoint.depth.shape, mode="bicubic"
                    ).squeeze(0)
                current_uncertainty, loss_init = get_loss_mapping_uncertainty(
                    self.config["mapping"],
                    image,
                    depth,
                    viewpoint,
                    opacity,
                    self.uncer_network,
                    train_frac,
                    ssim_frac,
                    initialization=True,
                )

                stride = self.config["mapping"]["uncertainty_params"]["reg_stride"]
                feature_buffer = [
                    viewpoint.features[::stride, ::stride].to(device=image.device),
                ]
                uncer_buffer = [
                    current_uncertainty[::stride, ::stride].unsqueeze(-1),
                ]
                loss_init += self.config["mapping"]["uncertainty_params"][
                    "reg_mult"
                ] * map_utils.compute_dino_regularization_loss(
                    uncer_buffer, feature_buffer
                )

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_init += 10 * isotropic_loss.mean()

            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )
                    self.iterations_after_densify_or_reset = 0

                if self.iteration_count == self.init_gaussian_reset:
                    self.gaussians.reset_opacity()
                    self.iterations_after_densify_or_reset = 0

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)

                if self.uncertainty_aware:
                    self.uncer_optimizer.step()
                    self.uncer_optimizer.zero_grad()



                self.frame_count_log[kf_idx] += 1

            self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()
        self.printer.print("Initialized map", FontColor.MAPPER)

        # online plotting
        if self.online_plotting and not self.test_training_time_mode:
            plot_dir = self.save_dir + "/online_plots"
            suffix = "_init"
            for cur_idx in self.current_window:
                self.save_fig_everything(cur_idx, plot_dir, suffix)

        return render_pkg

    def map_opt_online(self, current_window, iters=1):
        if len(current_window) == 0:
            raise ValueError("No keyframes in the current window")

        # Online plot before optimization
        cur_idx = current_window[np.array(current_window).argmax()]

        if self.online_plotting and not self.test_training_time_mode:
            plot_dir = self.save_dir + "/online_plots"
            suffix = "_before_opt"
            self.save_fig_everything(cur_idx, plot_dir, suffix)

        viewpoint_stack, viewpoint_kf_idx_stack = [], []
        for kf_idx, viewpoint in self.cameras.items():
            if self.is_kf[kf_idx]:
                viewpoint_stack.append(viewpoint)
                viewpoint_kf_idx_stack.append(kf_idx)

        # We set the current frame to be chosen by prob at least 50%
        # and the rest frame evenly distribute the remaining prob
        cur_window_prob = 0.5
        prob = np.full(
            len(viewpoint_stack),
            (1 - cur_window_prob)
            * iters
            / (len(viewpoint_stack) - len(current_window)),
        )
        assert viewpoint_kf_idx_stack[-1] == cur_idx
        if len(current_window) <= len(viewpoint_stack) / 2.0:
            for view_idx in range(len(viewpoint_kf_idx_stack)):
                kf_idx = viewpoint_kf_idx_stack[view_idx]
                if kf_idx in current_window:
                    prob[view_idx] = cur_window_prob * iters / (len(current_window))
        prob /= prob.sum()

        for cur_iter in tqdm(range(iters)):

            self.iteration_count += 1
            self.iterations_after_densify_or_reset += 1

            loss_mapping = 0

            cam_idx = np.random.choice(np.arange(len(viewpoint_stack)), p=prob)
            viewpoint = viewpoint_stack[cam_idx]

            rgb_stack = []
           
            if (cur_iter > iters/2 or 1) and self.is_blur:
                rgb_stack, total_num_poses = self.render_blur(viewpoint)

            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )

            rgb_stack.append(image)
            image = torch.mean(torch.stack(rgb_stack, dim=0), dim=0)

            if self.config["mapping"]["full_resolution"]:
                depth = F.interpolate(
                    depth.unsqueeze(0), viewpoint.depth.shape, mode="bicubic"
                ).squeeze(0)
            if not self.uncertainty_aware:
                loss_mapping += get_loss_mapping(
                    self.config["mapping"], image, depth, viewpoint, opacity
                )
            else:
                train_frac = self.uncer_params["train_frac_fix"]
                ssim_frac = self.uncer_params["train_frac_fix"]

                (
                    current_uncertainty,
                    current_loss_mapping,
                ) = get_loss_mapping_uncertainty(
                    self.config["mapping"],
                    # (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b, # bug? 
                    image, # maybe
                    depth,
                    viewpoint,
                    opacity,
                    self.uncer_network,
                    train_frac,
                    ssim_frac,
                    freeze_uncertainty_loss=self.iterations_after_densify_or_reset < 20,
                )

                loss_mapping += current_loss_mapping

                # Dino_regularization loss
                if self.iterations_after_densify_or_reset >= 20:
                    stride = self.config["mapping"]["uncertainty_params"]["reg_stride"]
                    reg_multi = self.config["mapping"]["uncertainty_params"]["reg_mult"]

                    viewpoint = viewpoint_stack[cam_idx]
                    feature_buffer = [
                        viewpoint_stack[reg_cam_idx].features.to(device=image.device)
                        for reg_cam_idx in range(
                            max(0, cam_idx - 2), min(len(viewpoint_stack), cam_idx + 3)
                        )
                    ]
                    feat_dim = feature_buffer[0].shape[-1]
                    feature_buffer = torch.stack(feature_buffer).view(-1, feat_dim)
                    num_samples = feature_buffer.shape[0] // (stride ** 4)
                    sampled_feature = feature_buffer[
                        torch.randperm(feature_buffer.shape[0])[:num_samples]
                    ].unsqueeze(0)
                    sampled_uncer = self.uncer_network(sampled_feature, viewpoint.uid)
                    loss_mapping += (
                        reg_multi
                        * map_utils.compute_dino_regularization_loss(
                            sampled_uncer, sampled_feature
                        )
                    )

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()

            loss_mapping.backward()
            gaussian_split = False
            # Deinsifying / Pruning Gaussians
            with torch.no_grad():
                if cur_iter == iters - 1:
                    self._update_occ_aware_visibility(current_window)

                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )
                    gaussian_split = True
                    self.iterations_after_densify_or_reset = 0
                    self.printer.print("Densify and prune the Gaussians", FontColor.MAPPER)

                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    self.printer.print(
                        "Resetting the opacity of non-visible Gaussians",
                        FontColor.MAPPER,
                    )
                    self.gaussians.reset_opacity_nonvisible([visibility_filter])
                    gaussian_split = True
                    self.iterations_after_densify_or_reset = 0


                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                if self.uncertainty_aware:
                    self.uncer_optimizer.step()
                    self.uncer_optimizer.zero_grad()

            self.frame_count_log[viewpoint_kf_idx_stack[cam_idx]] += 1

        # Online plotting
        if self.online_plotting and not self.test_training_time_mode:
            plot_dir = self.save_dir + "/online_plots"
            suffix = "_after_opt"
            self.save_fig_everything(cur_idx, plot_dir, suffix)

        # online plot the uncertainty mask
        if self.vis_uncertainty_online and not self.test_training_time_mode:
            self._vis_uncertainty_mask_all()
        return gaussian_split

    def final_refine(self, iters=26000, just_gs=False):
        self.printer.print("Starting final refinement", FontColor.MAPPER)

        c = 0
        fix_refine_iters = c # 0
        iters = iters + fix_refine_iters + (c - fix_refine_iters)

        opt_params = []
        plot_dir = self.save_dir + "/test/test"

        if not just_gs:
            # Do final update of depths and poses
            self._update_keyframes_from_frontend()

        random_viewpoint_stack, random_viewpoint_kf_idx_stack = [], []
        for kf_idx, viewpoint in self.cameras.items():
            if self.is_kf[kf_idx]:
                random_viewpoint_stack.append(viewpoint)
                random_viewpoint_kf_idx_stack.append(kf_idx)

        for refine_iter in tqdm(range(iters)):
            if refine_iter == fix_refine_iters:
                for cam_idx, viewpoint in tqdm(self.cameras.items()):
                    opt_params.append(
                        {
                            "params": [viewpoint.opt_rot_start],
                            "lr": self.lr_opt,
                            "name": "opt_rot_start_{}".format(viewpoint.uid),
                        }
                    )
                    opt_params.append(
                        {
                            "params": [viewpoint.opt_trans_start],
                            "lr": self.lr_opt,
                            "name": "opt_trans_start_{}".format(viewpoint.uid),
                        }
                    )
                    opt_params.append(
                        {
                            "params": [viewpoint.opt_rot_end],
                            "lr": self.lr_opt,
                            "name": "opt_rot_end_{}".format(viewpoint.uid),
                        }
                    )
                    opt_params.append(
                        {
                            "params": [viewpoint.opt_trans_end],
                            "lr": self.lr_opt,
                            "name": "opt_trans_end_{}".format(viewpoint.uid),
                        }
                    )
                 

                    if cam_idx == 0 or self.is_exposure == False:
                        # Do not add first frame for exposure optimization
                        continue
                    opt_params.append(
                        {
                            "params": [viewpoint.exposure_a],
                            "lr": 0.01,
                            "name": "exposure_a_{}".format(viewpoint.uid),
                        }
                    )
                    opt_params.append(
                        {
                            "params": [viewpoint.exposure_b],
                            "lr": 0.01,
                            "name": "exposure_b_{}".format(viewpoint.uid),
                        }
                    )
                self.keyframe_optimizers = torch.optim.Adam(opt_params)

            self.iteration_count += 1
            self.iterations_after_densify_or_reset += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            uncer_buffer, feature_buffer = [], []

            rand_idx = np.random.choice(range(len(random_viewpoint_stack)))
            random_viewpoint_kf_idxs = []

            viewpoint = random_viewpoint_stack[rand_idx]

            if not self.test_training_time_mode and refine_iter%10 == 0 and refine_iter < fix_refine_iters + 100:
                suffix = f"_refine_online_before_step_{refine_iter}"
                self.save_fig_everything(rand_idx, plot_dir, suffix)

            random_viewpoint_kf_idxs.append(random_viewpoint_kf_idx_stack[rand_idx])


            rgb_stack = []
            if (refine_iter > iters/2 or 1) and self.is_blur:
                rgb_stack, total_num_poses = self.render_blur(viewpoint)

            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )

            rgb_stack.append(image)
            image = torch.mean(torch.stack(rgb_stack, dim=0), dim=0)

            if self.config["mapping"]["full_resolution"]:
                depth = F.interpolate(
                    depth.unsqueeze(0), viewpoint.depth.shape, mode="bicubic"
                ).squeeze(0)
            if not self.uncertainty_aware:
                loss_mapping += get_loss_mapping(
                    self.config["mapping"], image, depth, viewpoint, opacity
                )
            else:
                train_frac = self.uncer_params["train_frac_fix"]
                ssim_frac = self.uncer_params["train_frac_fix"]

                (
                    current_uncertainty,
                    loss_mapping_this_frame,
                ) = get_loss_mapping_uncertainty(
                    self.config["mapping"],
                    image,
                    depth,
                    viewpoint,
                    opacity,
                    self.uncer_network,
                    train_frac,
                    ssim_frac,
                    freeze_uncertainty_loss=self.iterations_after_densify_or_reset
                    < 200,
                )
                loss_mapping += loss_mapping_this_frame

                stride = self.config["mapping"]["uncertainty_params"]["reg_stride"]
                uncer_buffer.append(
                    current_uncertainty[::stride, ::stride].unsqueeze(-1)
                )
                feature_buffer.append(
                    viewpoint.features[::stride, ::stride].to(device=image.device)
                )

            if self.uncertainty_aware and self.iterations_after_densify_or_reset >= 200:
                stride = self.config["mapping"]["uncertainty_params"]["reg_stride"]
                reg_multi = self.config["mapping"]["uncertainty_params"]["reg_mult"]
                viewpoint = random_viewpoint_stack[rand_idx]
                feature_buffer = [
                    random_viewpoint_stack[reg_cam_idx].features.to(device=image.device)
                    for reg_cam_idx in range(
                        max(0, rand_idx - 2),
                        min(len(random_viewpoint_stack), rand_idx + 3),
                    )
                ]
                feat_dim = feature_buffer[0].shape[-1]
                feature_buffer = torch.stack(feature_buffer).view(-1, feat_dim)
                num_samples = feature_buffer.shape[0] // (stride ** 4)
                sampled_feature = feature_buffer[
                    torch.randperm(feature_buffer.shape[0])[:num_samples]
                ].unsqueeze(0)
                sampled_uncer = self.uncer_network(sampled_feature, viewpoint.uid)
                loss_mapping += reg_multi * map_utils.compute_dino_regularization_loss(
                    sampled_uncer, sampled_feature
                )

            viewspace_point_tensor_acm.append(viewspace_point_tensor)
            visibility_filter_acm.append(visibility_filter)
            radii_acm.append(radii)
            n_touched_acm.append(n_touched)

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()

            with torch.no_grad():
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                # Optimize the exposure compensation
                if refine_iter >= fix_refine_iters:
                    self.keyframe_optimizers.step()
                    self.keyframe_optimizers.zero_grad(set_to_none=True)
                    if self.uncertainty_aware:
                        self.uncer_optimizer.step()
                self.uncer_optimizer.zero_grad()

            for kf_idx in random_viewpoint_kf_idxs:
                self.frame_count_log[kf_idx] += 1

        if self.vis_uncertainty_online:
            self._vis_uncertainty_mask_all(is_final=True)
        
        if self.config['gui']:
            self._send_to_gui(self.current_window[np.array(self.current_window).argmax()])

        self.printer.print("Final refinement done", FontColor.MAPPER)

    """
    Viusalization functions
    """
    def get_all_nodes(self, iter_per_frame=1, outlier_timestamps=[]):
        self.printer.print("Starting get all nodes", FontColor.MAPPER)
        self.psnr_list, self.ssim_list, self.lpips_list = [], [], []

        random_viewpoint_stack, random_viewpoint_kf_idx_stack = [], []
        random_viewpoint_dict = {}
        for kf_idx, viewpoint in self.cameras.items():
            if kf_idx in outlier_timestamps:
                continue
            random_viewpoint_dict[kf_idx] = viewpoint
        
        for video_idx, viewpoint in enumerate(tqdm(random_viewpoint_dict.values())):
            opt_params = []

            frame_idx = video_idx

            if video_idx != 0:
                viewpoint.exposure_a.data = random_viewpoint_dict[video_idx - 1].exposure_a.data.detach().clone().requires_grad_(True)
                viewpoint.exposure_b.data = random_viewpoint_dict[video_idx - 1].exposure_b.data.detach().clone().requires_grad_(True)
            opt_params.append(
                {
                    "params": [viewpoint.opt_rot_start],
                    "lr": self.lr_opt_get_all_nodes,
                    "name": "opt_rot_start_{}".format(viewpoint.uid),
                }
            )

            opt_params.append(
                {
                    "params": [viewpoint.opt_trans_start],
                    "lr": self.lr_opt_get_all_nodes,
                    "name": "opt_trans_start_{}".format(viewpoint.uid),
                }
            )
            opt_params.append(
                {
                    "params": [viewpoint.opt_rot_end],
                    "lr": self.lr_opt_get_all_nodes,
                    "name": "opt_rot_end_{}".format(viewpoint.uid),
                }
            )
            opt_params.append(
                {
                    "params": [viewpoint.opt_trans_end],
                    "lr": self.lr_opt_get_all_nodes,
                    "name": "opt_trans_end_{}".format(viewpoint.uid),
                }
            )

            if video_idx != 0 and self.is_exposure:
                opt_params.append(
                    {
                        "params": [viewpoint.exposure_a],
                        "lr": 0.01,
                        "name": "exposure_a_{}".format(viewpoint.uid),
                    }
                )
                opt_params.append(
                    {
                        "params": [viewpoint.exposure_b],
                        "lr": 0.01,
                        "name": "exposure_b_{}".format(viewpoint.uid),
                    }
                )
        
            self.keyframe_optimizers = torch.optim.Adam(opt_params)

            random_viewpoint_kf_idxs = []
            random_viewpoint_kf_idxs.append(video_idx)

            for _ in range(iter_per_frame):
                loss_mapping = 0

                rgb_stack, total_num_poses = self.render_blur(viewpoint)
 
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                rgb_stack.append(image)
                image = torch.mean(torch.stack(rgb_stack, dim=0), dim=0)

                if self.config["mapping"]["full_resolution"]:
                    depth = F.interpolate(
                        depth.unsqueeze(0), viewpoint.depth.shape, mode="bicubic"
                    ).squeeze(0)
                if not self.uncertainty_aware:
                    loss_mapping += get_loss_mapping(
                        self.config["mapping"], image, depth, viewpoint, opacity
                    )
                else:
                    train_frac = self.uncer_params["train_frac_fix"]
                    ssim_frac = self.uncer_params["train_frac_fix"]

                    (
                        current_uncertainty,
                        loss_mapping_this_frame,
                    ) = get_loss_mapping_uncertainty(
                        self.config["mapping"],
                        image,
                        depth,
                        viewpoint,
                        opacity,
                        self.uncer_network,
                        train_frac,
                        ssim_frac,
                        freeze_uncertainty_loss=self.iterations_after_densify_or_reset
                        < 200,
                    )
                    loss_mapping += loss_mapping_this_frame
                
                loss_mapping.backward()

                with torch.no_grad():
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    self.uncer_optimizer.zero_grad()
                    # Optimize the exposure compensation
                    self.keyframe_optimizers.step()
                    self.keyframe_optimizers.zero_grad(set_to_none=True)

                for kf_idx in random_viewpoint_kf_idxs:
                    self.frame_count_log[kf_idx] += 1
            
            plot_dir = self.save_dir + "/plots_get_all_nodes"
            mkdir_p(plot_dir)
            self.save_fig_everything_t(frame_idx, plot_dir, eval=True)

        self.printer.print("Get all nodes done", FontColor.MAPPER)

    @torch.no_grad()
    def _get_uncertainty_ssim_loss_vis(
        self, gt_image: torch.Tensor, rendered_img: torch.Tensor, opacity: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates and visualizes the SSIM-based uncertainty loss.
        Note that this is the Nerf-on-the-go version of SSIM. Only use this for visualization.

        Parameters:
        - gt_image (torch.Tensor): Ground truth image.
        - rendered_img (torch.Tensor): Rendered image from the model.
        - opacity (torch.Tensor): Opacity values for the rendered image.

        Returns:
        - torch.Tensor: Nerf-on-the-go version of SSIM-based uncertainty loss map (scaled).
        """
        ssim_frac = self.uncer_params["train_frac_fix"]
        l, c, s = map_utils.compute_ssim_components(
            gt_image,
            rendered_img,
            window_size=self.config["mapping"]["uncertainty_params"][
                "ssim_window_size"
            ],
        )
        ssim_loss = torch.clip(
            (100 + 900 * map_utils.compute_bias_factor(ssim_frac, 0.8))
            * (1 - l)
            * (1 - s)
            * (1 - c),
            max=5.0,
        )
        # ssim_loss = torch.clip(
        #     (100 + 900 * map_utils.compute_bias_factor(ssim_frac, 0.8)) * (1 - l * s *c),
        #     max=5.0,
        # )
        median_filter = MedianPool2d(
            kernel_size=self.config["mapping"]["uncertainty_params"][
                "ssim_median_filter_size"
            ],
            stride=1,
            padding=0,
            same=True,
        )
        ssim_loss = (
            median_filter(ssim_loss.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        )
        opacity_th = self.config["mapping"]["uncertainty_params"][
            "opacity_th_for_uncer_loss"
        ]
        ssim_loss[opacity < opacity_th] = 0

        return ssim_loss

    @torch.no_grad()
    def get_viewpoint_uncertainty_no_grad(self, viewpoint: Camera) -> torch.Tensor:
        """
        Compute the uncertainty for a given viewpoint without gradient computation.
        """
        features = viewpoint.features.to(self.device)
        with Lock():
            uncertainty = self.uncer_network(features, viewpoint.uid)

        # Process uncertainty values
        uncertainty = torch.clip(uncertainty, min=0.1) + 1e-3
        target_shape = (viewpoint.image_height, viewpoint.image_width)
        uncertainty_resized = map_utils.resample_tensor_to_shape(
            uncertainty, target_shape
        )

        # Apply data rate adjustment, the same as how we calculate the loss function
        train_frac = self.uncer_params["train_frac_fix"]
        data_rate = 1 + map_utils.compute_bias_factor(train_frac, 0.8)
        uncertainty_adjusted = (uncertainty_resized - 0.1) * data_rate + 0.1

        return uncertainty_adjusted ** 2
    
    @torch.no_grad()
    def save_fig_everything_t(self, keyframe_idx: int, plot_dir: str, suffix: str = "", depth_max: float = 10.0, eval=False):
        """
        Saves various visualizations for a specific keyframe.

        This function renders the scene from a given viewpoint, compares the rendered image
        and depth to ground truth, calculates quality metrics, and saves visualizations.
        If uncertainty-aware mode is enabled, it also includes uncertainty visualizations.

        Parameters:
        - keyframe_idx (int): Index of the keyframe to visualize.
        - plot_dir (str): Directory where the visualizations will be saved.
        - suffix (str, optional): Additional string to append to the saved filename. Defaults to "".
                                  The saved image will be named as "{keyframe_idx}{suffix}.png".
        - depth_max (float, optional): Maximum depth value for visualization. Defaults to 10.0.
        """
        viewpoint = self.cameras[keyframe_idx]

        rgb_stack = []

        rgb_stack, total_num_poses = self.render_blur(viewpoint)

        render_pkg = render(
            viewpoint, self.gaussians, self.pipeline_params, self.background
        )
        (
            image,
            viewspace_point_tensor,
            visibility_filter,
            radii,
            rendered_depth,
            opacity,
            n_touched,
        ) = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
            render_pkg["depth"].detach(),
            render_pkg["opacity"],
            render_pkg["n_touched"],
        )

        rgb_stack.append(image)
        rendered_img = torch.mean(torch.stack(rgb_stack, dim=0), dim=0).detach()

        if self.config["mapping"]["full_resolution"]:
            rendered_depth = F.interpolate(
                rendered_depth.unsqueeze(0), viewpoint.depth.shape, mode="bicubic"
            ).squeeze(0)
        gt_image = viewpoint.original_image
        gt_depth = viewpoint.depth
        
        rendered_img = (torch.exp(viewpoint.exposure_a)) * rendered_img + viewpoint.exposure_b # exp
        rendered_img = torch.clamp(rendered_img, 0.0, 1.0)
        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = (rendered_img.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        mask = gt_image > 0
        psnr_score = psnr(
            (rendered_img[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0)
        )

        if eval:
            ssim_score = ssim((rendered_img)[None],
            (gt_image)[None],
            ).item()
            full_mask = np.ones_like(gt[:, :, :1])
            lpips_score = compute_lpips(
                self.lpips_fn, gt, pred, full_mask, device=self.device
            ).item()
            self.psnr_list.append(psnr_score.item())
            self.ssim_list.append(ssim_score)
            self.lpips_list.append(lpips_score)
            with open(f"{self.save_dir}/psnr.txt", "a+") as f:
                f.write(f"{psnr_score.item()} {str(np.mean(self.psnr_list))}\n")
            with open(f"{self.save_dir}/ssim.txt", "a+") as f:
                f.write(f"{ssim_score} {str(np.mean(self.ssim_list))}\n")
            with open(f"{self.save_dir}/lpips.txt", "a+") as f:
                f.write(f"{lpips_score} {str(np.mean(self.lpips_list))}\n")

        diff_rgb=np.abs(gt - pred)

        if gt_depth is not None:
            diff_depth_l1 = torch.abs(rendered_depth.detach().cpu() - gt_depth)
            diff_depth_l1 = diff_depth_l1 * (gt_depth > 0)
            depth_l1 = diff_depth_l1.sum() / (gt_depth > 0).sum()
            diff_depth_l1 = diff_depth_l1.cpu().squeeze(0)
        else:
            diff_depth_l1 = torch.zeros_like(rendered_depth).cpu().squeeze(0)
            gt_depth = np.zeros_like(diff_rgb[:, :, 0])
            depth_l1 = 1.

        if self.uncertainty_aware:
            # Add plotting 2x4 grid with additional figures for uncertainty
            # Estimated uncertainty map
            uncertainty_map = self.get_viewpoint_uncertainty_no_grad(viewpoint)
            uncertainty_map = uncertainty_map.cpu().squeeze(0)

            # SSIM loss
            opacity = render_pkg["opacity"].detach().squeeze()
            ssim_loss = self._get_uncertainty_ssim_loss_vis(
                gt_image, rendered_img, opacity
            )
            ssim_loss = ssim_loss.cpu().squeeze(0)
        else:
            # All white
            uncertainty_map = torch.ones_like(rendered_img)
            ssim_loss = torch.ones_like(rendered_img)

        # Make the plot
        # Determine Plot Aspect Ratio
        aspect_ratio = gt_image.shape[2] / gt_image.shape[1]
        fig_height = 8
        fig_width = 11
        fig_width = fig_width * aspect_ratio

        # Plot the Ground Truth and Rasterized RGB & Depth, along with Diff Depth & Silhouette
        fig, axs = plt.subplots(2, 4, figsize=(fig_width, fig_height))
        axs[0, 0].imshow(gt_image.cpu().permute(1, 2, 0))
        axs[0, 0].set_title("Ground Truth RGB", fontsize=16)
        axs[0, 1].imshow(gt_depth, cmap='jet', vmin=0, vmax=depth_max)
        axs[0, 1].set_title(f"Metric Depth, vmax:{depth_max:.2f}", fontsize=16)
        axs[1, 0].imshow(rendered_img.cpu().permute(1, 2, 0))
        axs[1, 0].set_title(f"{int(total_num_poses)} Rendered RGB EXP, PSNR: {psnr_score.item():.2f}", fontsize=16)
        axs[1, 1].imshow(rendered_depth[0, :, :].cpu(), cmap='jet', vmin=0, vmax=depth_max)
        axs[1, 1].set_title("Rendered Depth, L1: {:.2f}".format(depth_l1), fontsize=16)
        axs[0, 2].imshow(diff_rgb, cmap='jet', vmin=0, vmax=diff_rgb.max())
        axs[0, 2].set_title(f"Diff RGB L1, vmax:{diff_rgb.max():.2f}", fontsize=16)
        axs[1, 2].imshow(diff_depth_l1, cmap='jet', vmin=0, vmax=depth_max/5.0)
        axs[1, 2].set_title(f"Diff Depth L1, vmax:{depth_max/5.0:.2f}", fontsize=16)
        axs[0, 3].imshow(uncertainty_map, cmap='jet', vmin=0, vmax=5)
        axs[0, 3].set_title("Uncertainty", fontsize=16)
        axs[1, 3].imshow(ssim_loss, cmap='jet', vmin=0, vmax=5)
        axs[1, 3].set_title("ssim_loss", fontsize=16)
        
        for i in range(2):
            for j in range(4):
                axs[i, j].axis('off')
                axs[i, j].grid(False)

        frame_idx = int(viewpoint.uid)
        fig.suptitle(f"Key Frame idx ({keyframe_idx}), Frame idx ({frame_idx}), Plot{suffix}", y=0.95, fontsize=20)
        fig.tight_layout()

        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, f"video_idx_{keyframe_idx}_kf_idx_{frame_idx}{suffix}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    @torch.no_grad()
    def save_fig_everything(self, keyframe_idx: int, plot_dir: str, suffix: str = "", depth_max: float = 10.0):
        """
        Saves various visualizations for a specific keyframe.

        This function renders the scene from a given viewpoint, compares the rendered image
        and depth to ground truth, calculates quality metrics, and saves visualizations.
        If uncertainty-aware mode is enabled, it also includes uncertainty visualizations.

        Parameters:
        - keyframe_idx (int): Index of the keyframe to visualize.
        - plot_dir (str): Directory where the visualizations will be saved.
        - suffix (str, optional): Additional string to append to the saved filename. Defaults to "".
                                  The saved image will be named as "{keyframe_idx}{suffix}.png".
        - depth_max (float, optional): Maximum depth value for visualization. Defaults to 10.0.
        """
        viewpoint = self.cameras[keyframe_idx]

        rgb_stack = []
        if self.is_blur:
            rgb_stack, total_num_poses = self.render_blur(viewpoint)

        render_pkg = render(
            viewpoint, self.gaussians, self.pipeline_params, self.background
        )
        (
            image,
            viewspace_point_tensor,
            visibility_filter,
            radii,
            rendered_depth,
            opacity,
            n_touched,
        ) = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
            render_pkg["depth"].detach(),
            render_pkg["opacity"],
            render_pkg["n_touched"],
        )

        rgb_stack.append(image)
        rendered_img = torch.mean(torch.stack(rgb_stack, dim=0), dim=0).detach()

        if self.config["mapping"]["full_resolution"]:
            rendered_depth = F.interpolate(
                rendered_depth.unsqueeze(0), viewpoint.depth.shape, mode="bicubic"
            ).squeeze(0)
        gt_image = viewpoint.original_image
        gt_depth = viewpoint.depth
        
        rendered_img = (torch.exp(viewpoint.exposure_a)) * rendered_img + viewpoint.exposure_b # exp
        rendered_img = torch.clamp(rendered_img, 0.0, 1.0)
        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = (rendered_img.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        mask = gt_image > 0
        psnr_score = psnr(
            (rendered_img[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0)
        )
        diff_rgb=np.abs(gt - pred)

        if gt_depth is not None:
            diff_depth_l1 = torch.abs(rendered_depth.detach().cpu() - gt_depth)
            diff_depth_l1 = diff_depth_l1 * (gt_depth > 0)
            depth_l1 = diff_depth_l1.sum() / (gt_depth > 0).sum()
            diff_depth_l1 = diff_depth_l1.cpu().squeeze(0)
        else:
            diff_depth_l1 = torch.zeros_like(rendered_depth).cpu().squeeze(0)
            gt_depth = np.zeros_like(diff_rgb[:, :, 0])
            depth_l1 = 1.

        if self.uncertainty_aware:
            # Add plotting 2x4 grid with additional figures for uncertainty
            # Estimated uncertainty map
            uncertainty_map = self.get_viewpoint_uncertainty_no_grad(viewpoint)
            uncertainty_map = uncertainty_map.cpu().squeeze(0)

            # SSIM loss
            opacity = render_pkg["opacity"].detach().squeeze()
            ssim_loss = self._get_uncertainty_ssim_loss_vis(
                gt_image, rendered_img, opacity
            )
            ssim_loss = ssim_loss.cpu().squeeze(0)
        else:
            # All white
            uncertainty_map = torch.ones_like(rendered_img[0]).cpu()
            ssim_loss = torch.ones_like(rendered_img[0]).cpu()

        # Make the plot
        # Determine Plot Aspect Ratio
        aspect_ratio = gt_image.shape[2] / gt_image.shape[1]
        fig_height = 8
        fig_width = 11
        fig_width = fig_width * aspect_ratio

        # Plot the Ground Truth and Rasterized RGB & Depth, along with Diff Depth & Silhouette
        fig, axs = plt.subplots(2, 4, figsize=(fig_width, fig_height))
        axs[0, 0].imshow(gt_image.cpu().permute(1, 2, 0))
        axs[0, 0].set_title("Ground Truth RGB", fontsize=16)
        axs[0, 1].imshow(gt_depth, cmap='jet', vmin=0, vmax=depth_max)
        axs[0, 1].set_title(f"Metric Depth, vmax:{depth_max:.2f}", fontsize=16)
        axs[1, 0].imshow(rendered_img.cpu().permute(1, 2, 0))
        axs[1, 0].set_title(f"{int(total_num_poses)} Rendered RGB EXP, PSNR: {psnr_score.item():.2f}", fontsize=16)
        axs[1, 1].imshow(rendered_depth[0, :, :].cpu(), cmap='jet', vmin=0, vmax=depth_max)
        axs[1, 1].set_title("Rendered Depth, L1: {:.2f}".format(depth_l1), fontsize=16)
        axs[0, 2].imshow(diff_rgb, cmap='jet', vmin=0, vmax=diff_rgb.max())
        axs[0, 2].set_title(f"Diff RGB L1, vmax:{diff_rgb.max():.2f}", fontsize=16)
        axs[1, 2].imshow(diff_depth_l1, cmap='jet', vmin=0, vmax=depth_max/5.0)
        axs[1, 2].set_title(f"Diff Depth L1, vmax:{depth_max/5.0:.2f}", fontsize=16)
        axs[0, 3].imshow(uncertainty_map, cmap='jet', vmin=0, vmax=5)
        axs[0, 3].set_title("Uncertainty", fontsize=16)
        axs[1, 3].imshow(ssim_loss, cmap='jet', vmin=0, vmax=5)
        axs[1, 3].set_title("ssim_loss", fontsize=16)
        
        for i in range(2):
            for j in range(4):
                axs[i, j].axis('off')
                axs[i, j].grid(False)

        frame_idx = int(self.video.timestamp[keyframe_idx])
        fig.suptitle(f"Key Frame idx ({keyframe_idx}), Frame idx ({frame_idx}), Plot{suffix}", y=0.95, fontsize=20)
        fig.tight_layout()

        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, f"video_idx_{keyframe_idx}_kf_idx_{frame_idx}{suffix}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def save_all_kf_figs(
        self,
        save_dir: str,
        iteration: Union[str, float] ="after_refine",
    ): 
        """
        Save figures for all keyframes in the specified directory.

        This function saves figures for each keyframe in the video sequence,
        creates a directory for plots, and generates a gif from the saved figures.

        Args:
            save_dir (str): The base directory where figures will be saved.
            iteration (Union[str, float]): A string or float representing the 
                                        iteration or stage of the process. 
                                        Default is "after_refine".
        """
        video_idxs = self.video_idxs

        plot_dir = os.path.join(save_dir, "plots_" + iteration)
        mkdir_p(plot_dir)

        for kf_idx in tqdm(video_idxs):
            self.save_fig_everything(kf_idx, plot_dir)
        # Create gif
        create_gif_from_directory(plot_dir, plot_dir + '/output.gif', online=True)

    @torch.no_grad()
    def _vis_uncertainty_mask_all(self, n_rows=8, n_cols=8, is_final=False):
        """Used to inspect the uncertainty"""
        assert (
            n_rows % 2 == 0
        )  # one row for uncertainty, one for imgs, the other for uncertainty

        n_img = int(n_rows * n_cols / 2)
        if n_img >= len(self.cameras):
            keyframe_idxs = list(self.cameras.keys())
        else:
            keyframe_idxs = (
                list(self.cameras.keys())[:n_cols]
                + list(self.cameras.keys())[-(n_img - n_cols) :]
            )

        h = self.cameras[keyframe_idxs[0]].image_height
        w = self.cameras[keyframe_idxs[0]].image_width
        all_white = (np.ones((h, w, 3)) * 255).astype(np.uint8)

        aspect_ratio = w / h
        fig_height = 8
        fig_width = 8.5
        fig, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * fig_width * aspect_ratio, n_rows * fig_height),
        )
        for i in range(0, n_rows // 2):
            for j in range(n_cols):
                idx = i * n_cols + j
                if idx >= len(keyframe_idxs):
                    axs[2 * i, j].imshow(all_white)
                    axs[2 * i + 1, j].imshow(all_white)
                else:
                    viewpoint = self.cameras[keyframe_idxs[idx]]
                    rgb = viewpoint.original_image.cpu().permute(1, 2, 0).numpy()
                    rgb = (rgb * 255.0).astype(np.uint8)
                    uncer_resized = self.get_viewpoint_uncertainty_no_grad(viewpoint)
                    uncer_resized = uncer_resized.cpu().squeeze(0)

                    axs[2 * i, j].imshow(rgb)
                    if keyframe_idxs[idx] in self.current_window:
                        # used for highlight
                        rect = patches.Rectangle(
                            (0, 0),
                            1,
                            1,
                            linewidth=30,
                            edgecolor="red",
                            facecolor="none",
                            transform=axs[2 * i, j].transAxes,
                        )
                        axs[2 * i, j].add_patch(rect)
                    axs[2 * i + 1, j].imshow(
                        uncer_resized ** 2, cmap="jet", vmin=0, vmax=5
                    )
                    axs[2 * i + 1, j].grid(False)
                axs[2 * i, j].axis("off")
                axs[2 * i + 1, j].axis("off")

        fig.tight_layout()
        cur_idx = self.current_window[np.array(self.current_window).argmax()]
        os.makedirs(os.path.join(self.save_dir, "online_uncer"), exist_ok=True)
        if is_final:
            save_path = os.path.join(
                self.save_dir, "online_uncer", f"after_final_refine.png"
            )
        else:
            save_path = os.path.join(self.save_dir, "online_uncer", f"{cur_idx}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def render_blur(self, viewpoint):
        t_s = self.t_s
        r_s = self.r_s

        rgb_stack = []
        total_num_poses = 1
        pts = self.gaussians.get_xyz
        gs_rotation = self.gaussians.get_rotation
        mu = pts

        pts_ones = torch.ones(pts.shape[0], 1, device=mu.device, dtype=torch.float32)
        pts4 = torch.cat((pts, pts_ones), dim=1)

        opt_cam_rot_start = viewpoint.opt_rot_start
        opt_cam_trans_start = viewpoint.opt_trans_start
        opt_cam_rot_end = viewpoint.opt_rot_end
        opt_cam_trans_end = viewpoint.opt_trans_end
        opt_cam_rot_start = F.normalize(opt_cam_rot_start, dim = -1)
        opt_cam_rot_end = F.normalize(opt_cam_rot_end, dim = -1)

        quat_start = opt_cam_rot_start

        quat_end = opt_cam_rot_end

        with torch.no_grad():

            dr = 2 * torch.acos(torch.abs(opt_cam_rot_start[0]))

            dt = (opt_cam_trans_start[0]**2 + 
                opt_cam_trans_start[1]**2 + 
                opt_cam_trans_start[2]**2).sqrt()

            num_poses = max(min((torch.ceil(dt / t_s)).int(), 4), min((torch.ceil(dr / r_s)).int(), 4))
            if self.fix_num_poses != 0:
                num_poses = self.fix_num_poses
        if num_poses >= 1:
            if num_poses > 1:
                qt_start = torch.cat([quat_start.squeeze(0), opt_cam_trans_start])


                qt_end = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=qt_start.device)

                qts = SplineN_linear_qt(qt_start, qt_end, num_poses, mu.device)

                qts = qts[0:-1]

                qs = qts[:, :4]  # [N, 4]
                ts = qts[:, 4:]  # [N, 3]
            else:
                qs = quat_start.unsqueeze(0)
                ts = opt_cam_trans_start.unsqueeze(0)


            rotation_matrices = quaternion_to_matrix_optimized(F.normalize(qs))

            rel_transforms = torch.eye(4, device=mu.device, dtype=torch.float32).repeat(len(qs), 1, 1)
            rel_transforms[:, :3, :3] = rotation_matrices
            rel_transforms[:, :3, 3] = ts


            pts4_expanded = pts4.unsqueeze(0).repeat(len(qs), 1, 1)  # [N, num_points, 4]
            transformed_pts_all = torch.bmm(rel_transforms, pts4_expanded.transpose(1, 2)).transpose(1, 2)[:, :, :3]


            _rotations_all = multiply_quaternions(
                gs_rotation.unsqueeze(0).repeat(len(qs), 1, 1), 
                qs.unsqueeze(-2)
            )
            total_num_poses += len(qs)
            for i in range(len(qs)):
                rgb_stack.append(    # render
                    render(
                        viewpoint, self.gaussians, self.pipeline_params, self.background, override_means_3d=transformed_pts_all[i], override_rotations=_rotations_all[i]
                    )["render"]
                )


        with torch.no_grad():

            dr = 2 * torch.acos(torch.abs(opt_cam_rot_end[0]))

            dt = (opt_cam_trans_end[0]**2 + 
                opt_cam_trans_end[1]**2 + 
                opt_cam_trans_end[2]**2).sqrt()
            num_poses = max(min((torch.ceil(dt / t_s)).int(), 4), min((torch.ceil(dr / r_s)).int(), 4))
            if self.fix_num_poses != 0:
                num_poses = self.fix_num_poses
        if num_poses >= 1:
            if num_poses > 1:
                qt_end = torch.cat([quat_end.squeeze(0), opt_cam_trans_end])

                qt_start = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=qt_end.device)

                qts = SplineN_linear_qt(qt_start, qt_end, num_poses, mu.device)
                qts = qts[1:]

                qs = qts[:, :4]  # [N, 4]
                ts = qts[:, 4:]  # [N, 3]

            else:
                qs = quat_end.unsqueeze(0)
                ts = opt_cam_trans_end.unsqueeze(0)

            rotation_matrices = quaternion_to_matrix_optimized(F.normalize(qs))
            rel_transforms = torch.eye(4, device=mu.device, dtype=torch.float32).repeat(len(qs), 1, 1)
            rel_transforms[:, :3, :3] = rotation_matrices
            rel_transforms[:, :3, 3] = ts

            pts4_expanded = pts4.unsqueeze(0).repeat(len(qs), 1, 1)  # [N, num_points, 4]
            transformed_pts_all = torch.bmm(rel_transforms, pts4_expanded.transpose(1, 2)).transpose(1, 2)[:, :, :3]

            _rotations_all = multiply_quaternions(
                gs_rotation.unsqueeze(0).repeat(len(qs), 1, 1), 
                qs.unsqueeze(-2)
            )

            total_num_poses += len(qs)
            for i in range(len(qs)):
                rgb_stack.append(    # render
                    render(
                        viewpoint, self.gaussians, self.pipeline_params, self.background, override_means_3d=transformed_pts_all[i], override_rotations=_rotations_all[i]
                    )["render"]
                )
        return rgb_stack, total_num_poses