import os
import torch
import numpy as np
import time
from collections import OrderedDict
import torch.multiprocessing as mp
from munch import munchify

from src.modules.droid_net import DroidNet
from src.depth_video import DepthVideo
from src.trajectory_filler import PoseTrajectoryFiller
from src.utils.common import setup_seed, update_cam
from src.utils.Printer import Printer, FontColor
from src.utils.eval_traj import kf_traj_eval, full_traj_eval
from src.utils.datasets import BaseDataset
from src.tracker import Tracker
from src.mapper import Mapper
from src.backend import Backend
from src.utils.dyn_uncertainty.uncertainty_model import generate_uncertainty_mlp
from src.utils.datasets import RGB_NoPose
from src.gui import gui_utils, slam_gui
from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel

from src.utils.datasets import load_img_feature
from src.utils.camera_utils import Camera
from tqdm import tqdm

from thirdparty.gaussian_splatting.utils.general_utils import (
    multiply_quaternions,
)

import lpips

class SLAM:
    def __init__(self, cfg, stream: BaseDataset):
        super(SLAM, self).__init__()
        self.cfg = cfg
        self.device = cfg["device"]
        self.verbose: bool = cfg["verbose"]
        self.logger = None
        self.save_dir = cfg["data"]["output"] + "/" + cfg["scene"]

        os.makedirs(self.save_dir, exist_ok=True)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = update_cam(cfg)

        self.droid_net: DroidNet = DroidNet()

        self.printer = Printer(
            len(stream)
        )  # use an additional process for printing all the info

        self.load_pretrained(cfg)
        self.droid_net.to(self.device).eval()
        self.droid_net.share_memory()

        self.num_running_thread = torch.zeros((1)).int()
        self.num_running_thread.share_memory_()
        self.all_trigered = torch.zeros((1)).int()
        self.all_trigered.share_memory_()

        if self.cfg["mapping"]["uncertainty_params"]["activate"]:
            n_features = self.cfg["mapping"]["uncertainty_params"]["feature_dim"]
            self.uncer_network = generate_uncertainty_mlp(n_features) # , time_dim=1
            self.uncer_network.share_memory()
        else:
            self.uncer_network = None
            if self.cfg["tracking"]["uncertainty_params"]["activate"]:
                raise ValueError(
                    "uncertainty estimation cannot be activated on tracking while not on mapping"
                )

        self.video = DepthVideo(cfg, self.printer, uncer_network=self.uncer_network)
        self.ba = Backend(self.droid_net, self.video, self.cfg)

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(
            cfg=cfg,
            net=self.droid_net,
            video=self.video,
            printer=self.printer,
            device=self.device,
        )

        self.tracker: Tracker = None
        self.mapper: Mapper = None
        self.stream = stream

        self.test_training_time_mode = cfg["test_training_time_mode"]

    def load_pretrained(self, cfg):
        droid_pretrained = cfg["tracking"]["pretrained"]
        state_dict = OrderedDict(
            [
                (k.replace("module.", ""), v)
                for (k, v) in torch.load(droid_pretrained, weights_only=True).items()
            ]
        )
        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]
        self.droid_net.load_state_dict(state_dict)
        self.droid_net.eval()
        self.printer.print(
            f"Load droid pretrained checkpoint from {droid_pretrained}!", FontColor.INFO
        )

    def tracking(self, pipe):
        self.tracker = Tracker(self, pipe)
        self.printer.print("Tracking Triggered!", FontColor.TRACKER)
        self.all_trigered += 1

        os.makedirs(f"{self.save_dir}/mono_priors/depths", exist_ok=True)
        os.makedirs(f"{self.save_dir}/mono_priors/features", exist_ok=True)

        while self.all_trigered < self.num_running_thread:
            pass
        self.printer.pbar_ready()
        self.tracker.run(self.stream)
        self.printer.print("Tracking Done!", FontColor.TRACKER)

    def mapping(self, pipe, q_main2vis, q_vis2main):
        if self.cfg["mapping"]["uncertainty_params"]["activate"]:
            self.mapper = Mapper(self, pipe, self.uncer_network, q_main2vis, q_vis2main)
        else:
            self.mapper = Mapper(self, pipe, None, q_main2vis, q_vis2main)
        self.printer.print("Mapping Triggered!", FontColor.MAPPER)

        self.all_trigered += 1
        setup_seed(self.cfg["setup_seed"])

        while self.all_trigered < self.num_running_thread:
            pass
        self.mapper.run()
        self.printer.print("Mapping Done!", FontColor.MAPPER)

        self.terminate()

    def backend(self):
        self.printer.print("Final Global BA Triggered!", FontColor.TRACKER)

        metric_depth_reg_activated = self.video.metric_depth_reg
        if metric_depth_reg_activated:
            self.video.metric_depth_reg = False

        self.ba = Backend(self.droid_net, self.video, self.cfg)
        torch.cuda.empty_cache()
        self.ba.dense_ba(7)
        torch.cuda.empty_cache()
        self.ba.dense_ba(12)
        self.printer.print("Final Global BA Done!", FontColor.TRACKER)

        if metric_depth_reg_activated:
            self.video.metric_depth_reg = True

    def terminate(self):
        """fill poses for non-keyframe images and evaluate"""

        if (
            self.cfg["tracking"]["backend"]["final_ba"]
            and self.cfg["mapping"]["eval_before_final_ba"]
        ):
            self.video.save_video(f"{self.save_dir}/video.npz")
            if not isinstance(self.stream, RGB_NoPose):
                try:
                    ate_statistics, global_scale, r_a, t_a = kf_traj_eval(
                        f"{self.save_dir}/video.npz",
                        f"{self.save_dir}/traj/before_final_ba",
                        "kf_traj",
                        self.stream,
                        self.logger,
                        self.printer,
                    )
                except Exception as e:
                    self.printer.print(e, FontColor.ERROR)

        if not self.test_training_time_mode:
            self.mapper.save_all_kf_figs(
                self.save_dir,
                iteration="before_refine",
            )

        if self.cfg["tracking"]["backend"]["final_ba"]:
            self.backend()

        self.video.save_video(f"{self.save_dir}/video.npz")
        if not isinstance(self.stream, RGB_NoPose):
            try:
                ate_statistics, global_scale, r_a, t_a = kf_traj_eval(
                    f"{self.save_dir}/video.npz",
                    f"{self.save_dir}/traj",
                    "kf_traj",
                    self.stream,
                    self.logger,
                    self.printer,
                )
            except Exception as e:
                self.printer.print(e, FontColor.ERROR)

        if not self.test_training_time_mode:
            self.mapper.gaussians.save_ply(f"{self.save_dir}/final_gs_before.ply")
            if self.cfg["mapping"]["uncertainty_params"]["activate"]:
                torch.save(
                    self.mapper.uncer_network.state_dict(),
                    self.save_dir + "/uncertainty_mlp_weight_before.pth",
                )
        if self.cfg["tracking"]["backend"]["final_refine"]:
            self.mapper.final_refine(
                iters=self.cfg["mapping"]["final_refine_iters"]
            )  # this performs a set of optimizations with RGBD loss to correct

        opt_starts = []
        opt_ends = []
        exposures = []
        is_kfs = []
        for video_idx, viewpoint in self.mapper.cameras.items():
            opt_starts.append(torch.cat([viewpoint.opt_rot_start.data, viewpoint.opt_trans_start.data], dim = -1))
            opt_ends.append(torch.cat([viewpoint.opt_rot_end.data, viewpoint.opt_trans_end.data], dim = -1))
            exposures.append([viewpoint.exposure_a, viewpoint.exposure_b])
            is_kfs.append(self.mapper.is_kf[video_idx])

        data = {"opt_starts": opt_starts, "opt_ends": opt_ends, "exposures": exposures, "is_kfs": is_kfs}

        torch.save(data, f"{self.save_dir}/viewpoints.ckpt")

        # Evaluate the metrics
        if not self.test_training_time_mode:
            self.mapper.save_all_kf_figs(
                self.save_dir,
                iteration="after_refine",
            )

        ## Not used, see head comments of the function
        # self._eval_depth_all(ate_statistics, global_scale, r_a, t_a)

        # Regenerate feature extractor for non-keyframes
        self.traj_filler.setup_feature_extractor()
        full_traj_eval(
            self.traj_filler,
            self.mapper,
            f"{self.save_dir}/traj",
            "full_traj",
            self.stream,
            self.logger,
            self.printer,
            self.cfg['fast_mode'],
        )

        self.mapper.gaussians.save_ply(f"{self.save_dir}/final_gs.ply")

        if self.cfg["mapping"]["uncertainty_params"]["activate"]:
            torch.save(
                self.mapper.uncer_network.state_dict(),
                self.save_dir + "/uncertainty_mlp_weight.pth",
            )

        self.printer.print("Metrics Evaluation Done!", FontColor.EVAL)

    def _eval_depth_all(self, ate_statistics, global_scale, r_a, t_a):
        """From Splat-SLAM. Not used in ru4d-SLAM evaluation, but might be useful in the future."""
        # Evaluate depth error
        self.printer.print(
            "Evaluate sensor depth error with per frame alignment", FontColor.EVAL
        )
        depth_l1, depth_l1_max_4m, coverage = self.video.eval_depth_l1(
            f"{self.save_dir}/video.npz", self.stream
        )
        self.printer.print("Depth L1: " + str(depth_l1), FontColor.EVAL)
        self.printer.print("Depth L1 mask 4m: " + str(depth_l1_max_4m), FontColor.EVAL)
        self.printer.print("Average frame coverage: " + str(coverage), FontColor.EVAL)

        self.printer.print(
            "Evaluate sensor depth error with global alignment", FontColor.EVAL
        )
        depth_l1_g, depth_l1_max_4m_g, _ = self.video.eval_depth_l1(
            f"{self.save_dir}/video.npz", self.stream, global_scale
        )
        self.printer.print("Depth L1: " + str(depth_l1_g), FontColor.EVAL)
        self.printer.print(
            "Depth L1 mask 4m: " + str(depth_l1_max_4m_g), FontColor.EVAL
        )

        # save output data to dict
        # File path where you want to save the .txt file
        file_path = f"{self.save_dir}/depth_stats.txt"
        integers = {
            "depth_l1": depth_l1,
            "depth_l1_global_scale": depth_l1_g,
            "depth_l1_mask_4m": depth_l1_max_4m,
            "depth_l1_mask_4m_global_scale": depth_l1_max_4m_g,
            "Average frame coverage": coverage,  # How much of each frame uses depth from droid (the rest from Omnidata)
            "traj scaling": global_scale,
            "traj rotation": r_a,
            "traj translation": t_a,
            "traj stats": ate_statistics,
        }
        # Write to the file
        with open(file_path, "w") as file:
            for label, number in integers.items():
                file.write(f"{label}: {number}\n")

        self.printer.print(f"File saved as {file_path}", FontColor.EVAL)

    def run(self):
        m_pipe, t_pipe = mp.Pipe()

        q_main2vis = mp.Queue() if self.cfg['gui'] else None
        q_vis2main = mp.Queue() if self.cfg['gui'] else None

        processes = [
            mp.Process(target=self.tracking, args=(t_pipe,)),
            mp.Process(target=self.mapping, args=(m_pipe,q_main2vis,q_vis2main)),
        ]
        self.num_running_thread[0] += len(processes)
        for p in processes:
            p.start()

        if self.cfg['gui']:
            time.sleep(5)
            pipeline_params = munchify(self.cfg["mapping"]["pipeline_params"])
            bg_color = [0, 0, 0]
            background = torch.tensor(
                bg_color, dtype=torch.float32, device=self.device
            )
            gaussians = GaussianModel(self.cfg['mapping']['model_params']['sh_degree'], config=self.cfg)

            params_gui = gui_utils.ParamsGUI(
                pipe=pipeline_params,
                background=background,
                gaussians=gaussians,
                q_main2vis=q_main2vis,
                q_vis2main=q_vis2main,
            )
            gui_process = mp.Process(target=slam_gui.run, args=(params_gui,))
            gui_process.start()
            self.num_running_thread[0] += 1


        for p in processes:
            p.join()

        self.printer.terminate()

        for process in mp.active_children():
            process.terminate()
            process.join()

    def final_refine(self):
        # return
        m_pipe, t_pipe = mp.Pipe()

        q_main2vis = mp.Queue() if self.cfg['gui'] else None
        q_vis2main = mp.Queue() if self.cfg['gui'] else None

        pipe = m_pipe
        if self.cfg["mapping"]["uncertainty_params"]["activate"]:
            self.mapper = Mapper(self, pipe, self.uncer_network, q_main2vis, q_vis2main)

        if os.path.exists(os.path.join(self.save_dir, "uncertainty_mlp_weight_before.pth")):
            weight = torch.load(os.path.join(self.save_dir, "uncertainty_mlp_weight_before.pth"))
        else:
            weight = torch.load(os.path.join(self.save_dir, "uncertainty_mlp_weight.pth"))
        self.mapper.uncer_network.load_state_dict(weight)
        if os.path.exists(os.path.join(self.save_dir, "final_gs_before.ply")):
            self.mapper.gaussians.load_ply(os.path.join(self.save_dir, "final_gs_before.ply"))
        elif os.path.exists(os.path.join(self.save_dir, "final_gs_t.ply")):
            self.mapper.gaussians.load_ply(os.path.join(self.save_dir, "final_gs_t.ply"))
        else:
            self.mapper.gaussians.load_ply(os.path.join(self.save_dir, "final_gs.ply"))
        
        self.mapper.gaussians.training_setup(self.mapper.opt_params)

        video_data = np.load(os.path.join(self.save_dir, "video.npz"))
        timestamps = video_data["timestamps"].astype(int)
        depths = video_data['depths']
        poses = torch.from_numpy(video_data['poses']).to(self.device)
       
        self.mapper.iteration_count = 0
        self.mapper.iterations_after_densify_or_reset = 0
        self.mapper.cameras = {}
        self.mapper.is_kf = {}
        self.mapper.frame_count_log = {}

        self.mapper.video_idxs = []
        for video_idx, (timestamp, depth, pose) in enumerate(zip(timestamps, depths, poses)):
            pose = pose.inverse()

            if self.mapper.config["mapping"]["full_resolution"]:
                color = (
                    self.mapper.frame_reader.get_color_full_resol(timestamp)
                    .to(self.device)
                    .squeeze()
                )
                load_feature_suffix = "full"
            else:
                color = self.mapper.frame_reader.get_color(timestamp).to(self.device).squeeze()
                load_feature_suffix = ""

            # Load features if uncertainty-aware
            if self.mapper.uncertainty_aware:
                features = load_img_feature(
                    timestamp, self.save_dir, suffix=load_feature_suffix
                ).to(self.device)
            else:
                features = None

            # Get estimated depth and camera pose


            # Prepare data dictionary for Camera initialization
            camera_data = {
                "idx": video_idx,
                "gt_color": color,
                "est_depth": depth,
                "est_pose": pose,
                "features": features,
            }

            # Initialize Camera object
            viewpoint = Camera.init_from_dataset(
                self.mapper.frame_reader,
                camera_data,
                self.mapper.projection_matrix,
                full_resol=self.mapper.config["mapping"]["full_resolution"],
            )

            # Update camera pose and compute gradient mask
            # The Camera class is based on MonoGS and
            # init_from_dataset function only updates the ground truth pose
            viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
            viewpoint.compute_grad_mask(self.mapper.config)
            
            self.mapper.cameras[video_idx] = viewpoint
            self.mapper.video_idxs.append(video_idx)

            self.mapper.is_kf[video_idx] = True
            self.mapper.frame_count_log[video_idx] = 0

        viewpoints_data = torch.load(os.path.join(self.save_dir, "viewpoints.ckpt"))

        opt_starts = viewpoints_data["opt_starts"]
        opt_ends = viewpoints_data["opt_ends"]
        exposures = viewpoints_data["exposures"]

        for video_idx,(opt_start, opt_end, exposure) in enumerate(zip(opt_starts, opt_ends, exposures)):

            viewpoint = self.mapper.cameras[video_idx]

            viewpoint.opt_rot_start.data = opt_start[:4]
            viewpoint.opt_trans_start.data = opt_start[4:]
            viewpoint.opt_rot_end.data = opt_end[:4]
            viewpoint.opt_trans_end.data = opt_end[4:]
                    
            viewpoint.exposure_a.data = exposure[0].data
            viewpoint.exposure_b.data = exposure[1].data

        iters = self.cfg["mapping"]["final_refine_iters"]
        iters = 2000
        self.mapper.final_refine(
                iters=iters, just_gs=True
            )
        
        self.mapper.gaussians.save_ply(f"{self.save_dir}/final_gs_t.ply")

        self.mapper.save_all_kf_figs(
            self.save_dir,
            iteration="after_nth_refine",
        )


    def get_all_nodes(self, get_all=False, iter_per_frame=30):
        m_pipe, t_pipe = mp.Pipe()

        q_main2vis = mp.Queue() if self.cfg['gui'] else None
        q_vis2main = mp.Queue() if self.cfg['gui'] else None
        
        pipe = m_pipe
        
        if self.cfg["mapping"]["uncertainty_params"]["activate"]:
            self.mapper = Mapper(self, pipe, self.uncer_network, q_main2vis, q_vis2main)
        
        pipe = t_pipe
        self.tracker = None

        weight = torch.load(os.path.join(self.save_dir, "uncertainty_mlp_weight.pth"))
        self.mapper.uncer_network.load_state_dict(weight)

        if os.path.exists(os.path.join(self.save_dir, "final_gs_t.ply")):
            self.mapper.gaussians.load_ply(os.path.join(self.save_dir, "final_gs_t.ply"))
        else:
            self.mapper.gaussians.load_ply(os.path.join(self.save_dir, "final_gs.ply"))
            
        video_data = np.load(os.path.join(self.save_dir, "video.npz"))
        timestamps = video_data["timestamps"].astype(int)
        poses = video_data["poses"]
        valid_depth_masks = video_data['valid_depth_masks']
        for i, valid_depth_mask in enumerate(valid_depth_masks):
            num = np.count_nonzero(valid_depth_mask == False)
            if num > 0:
                print(f"video {i} has {num} invalid depth pixels")

        full_traj_path = os.path.join(self.save_dir, "traj/est_poses_full.txt")
        pose_data = np.loadtxt(full_traj_path, delimiter=' ', dtype=np.unicode_)
        pose_vecs = pose_data[:, 1:].astype(np.float64)
        def pose_matrix_from_quaternion(pvec):
            """ convert 4x4 pose matrix to (t, q) """
            from scipy.spatial.transform import Rotation

            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
            pose[:3, 3] = pvec[:3]
            return pose
        
        full_poses = []
        for i in range(len(pose_vecs)):
            full_poses.append(pose_matrix_from_quaternion(pose_vecs[i]))

        self.mapper.iteration_count = 30000
        self.mapper.iterations_after_densify_or_reset = 0
        self.mapper.cameras = {}
        self.mapper.is_kf = {}
        self.mapper.frame_count_log = {}

        self.mapper.video_idxs = []

        from src.utils.mono_priors.img_feature_extractors import predict_img_features

        for video_idx in tqdm(range(len(self.mapper.frame_reader))):
            pose = torch.from_numpy(full_poses[video_idx]).to(self.device).inverse()

            timestamp = video_idx
            if self.mapper.config["mapping"]["full_resolution"]:
                color = (
                    self.mapper.frame_reader.get_color_full_resol(timestamp)
                    .to(self.device)
                    .squeeze()
                )
                load_feature_suffix = "full"
            else:
                color = self.mapper.frame_reader.get_color(timestamp).to(self.device).squeeze()
                load_feature_suffix = ""

            # Load features if uncertainty-aware
            if self.mapper.uncertainty_aware:
                feat_path = f"{self.save_dir}/mono_priors/features/{timestamp:05d}{load_feature_suffix}.npy"
                if os.path.exists(feat_path):
                    features = load_img_feature(
                        timestamp, self.save_dir, suffix=load_feature_suffix
                    ).to(self.device)
                else:
                    if self.tracker is None:
                        self.tracker = Tracker(self, pipe)
                    features = predict_img_features(self.tracker.motion_filter.feat_extractor,timestamp,color.unsqueeze(0),self.cfg,self.device)
            else:
                features = None

            # Get estimated depth and camera pose


            # Prepare data dictionary for Camera initialization
            camera_data = {
                "idx": video_idx,
                "gt_color": color,
                "est_depth": None,
                "est_pose": pose,
                "features": features,
            }

            # Initialize Camera object
            viewpoint = Camera.init_from_dataset(
                self.mapper.frame_reader,
                camera_data,
                self.mapper.projection_matrix,
                full_resol=self.mapper.config["mapping"]["full_resolution"],
            )

            # Update camera pose and compute gradient mask
            # The Camera class is based on MonoGS and
            # init_from_dataset function only updates the ground truth pose
            viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
            viewpoint.compute_grad_mask(self.mapper.config)
            
            self.mapper.cameras[video_idx] = viewpoint
            self.mapper.video_idxs.append(video_idx)
            self.mapper.is_kf[video_idx] = False
            self.mapper.frame_count_log[video_idx] = 0

        self.printer.terminate()

        for process in mp.active_children():
            process.terminate()
            process.join()

        if os.path.exists(f"{self.save_dir}/viewpoints_all.ckpt") and 0: # ttt
            viewpoints_data = torch.load(os.path.join(self.save_dir, "viewpoints_all.ckpt"))
            T_starts = viewpoints_data["opt_starts"]
            T_ends = viewpoints_data["opt_ends"]
            exposures = viewpoints_data["exposures"]
            for T_start, T_end, exposure, viewpoint in tqdm(zip(T_starts, T_ends, exposures, self.mapper.cameras.values())):
                viewpoint.T_start = T_start[:3, 3]
                viewpoint.T_end = T_end[:3, 3]
                viewpoint.R_start = T_start[:3, :3]
                viewpoint.R_end = T_end[:3, :3]
                viewpoint.exposure_a.data = exposure[0].data
                viewpoint.exposure_b.data = exposure[1].data
        else:
            viewpoints_data = torch.load(os.path.join(self.save_dir, "viewpoints.ckpt"))

            opt_starts = viewpoints_data["opt_starts"]
            opt_ends = viewpoints_data["opt_ends"]
            exposures = viewpoints_data["exposures"]

            for video_idx,(opt_start, opt_end, exposure) in enumerate(zip(opt_starts, opt_ends, exposures)):
                timestamp = timestamps[video_idx]
                viewpoint = self.mapper.cameras[timestamp]

                viewpoint.opt_rot_start.data = opt_start[:4]
                viewpoint.opt_trans_start.data = opt_start[4:]
                viewpoint.opt_rot_end.data = opt_end[:4]
                viewpoint.opt_trans_end.data = opt_end[4:]
                        
                viewpoint.exposure_a.data = exposure[0].data
                viewpoint.exposure_b.data = exposure[1].data

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
            
            kf_idx = timestamps
            t = 0
            for viewpoint in tqdm(self.mapper.cameras.values()):
                video_idx = viewpoint.uid

                if video_idx not in timestamps:
                    if video_idx == kf_idx[t]:
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
                    else:
                        device = viewpoint.device
                        low, high = 0.001, 0.003

                        poses                
                        R_cur = viewpoint.R.float().to(device)
                        R_last = (torch.from_numpy(poses[t])[:3, :3]).to(device)
                        T_cur = viewpoint.T.float().to(device)
                        T_last = (torch.from_numpy(poses[t])[:3, 3]).to(device)
                        R_rel = R_cur @ (R_last).T 
                        T_rel = T_cur - T_last 

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

                if t + 1 < len(kf_idx) and video_idx == kf_idx[t+1]:
                    t += 1

        for param in self.uncer_network.parameters():
            param.requires_grad_(False)

        self.mapper.gaussians._xyz.requires_grad_(False)
        self.mapper.gaussians._features_dc.requires_grad_(False)
        self.mapper.gaussians._features_rest.requires_grad_(False)
        self.mapper.gaussians._opacity.requires_grad_(False)
        self.mapper.gaussians._scaling.requires_grad_(False)
        self.mapper.gaussians._rotation.requires_grad_(False)

        if get_all:
            outlier_timestamps = []
        else:
            outlier_timestamps = timestamps
        
        self.mapper.lpips_fn = lpips.LPIPS(net="alex", spatial=True).to(self.mapper.device)
        self.mapper.get_all_nodes(iter_per_frame=iter_per_frame, outlier_timestamps=outlier_timestamps)
        

        opt_starts_list = []
        opt_ends_list = []
        exposures = []
        is_kfs = []
        for video_idx, viewpoint in self.mapper.cameras.items():
            opt_starts_list.append(torch.cat([viewpoint.opt_rot_start.data, viewpoint.opt_trans_start.data]))
            opt_ends_list.append(torch.cat([viewpoint.opt_rot_end.data, viewpoint.opt_trans_end.data]))

            exposures.append([viewpoint.exposure_a.data, viewpoint.exposure_b.data])
            is_kfs.append(self.mapper.is_kf[video_idx])

        data = {"opt_starts": opt_starts_list, "opt_ends": opt_ends_list, "exposures": exposures, "is_kfs": is_kfs}

        torch.save(data, f"{self.save_dir}/viewpoints_all.ckpt")

    def get_all_nodes_o(self, get_all=False, iter_per_frame=30):
        m_pipe, t_pipe = mp.Pipe()

        q_main2vis = mp.Queue() if self.cfg['gui'] else None
        q_vis2main = mp.Queue() if self.cfg['gui'] else None

        pipe = m_pipe
        if self.cfg["mapping"]["uncertainty_params"]["activate"]:
            self.mapper = Mapper(self, pipe, self.uncer_network, q_main2vis, q_vis2main)
        
        pipe = t_pipe
        self.tracker = None

        weight = torch.load(os.path.join(self.save_dir, "uncertainty_mlp_weight.pth"))
        self.mapper.uncer_network.load_state_dict(weight)

        if os.path.exists(os.path.join(self.save_dir, "final_gs_t.ply")):
            self.mapper.gaussians.load_ply(os.path.join(self.save_dir, "final_gs_t.ply"))
        else:
            self.mapper.gaussians.load_ply(os.path.join(self.save_dir, "final_gs.ply"))

        video_data = np.load(os.path.join(self.save_dir, "video.npz"))
        timestamps = video_data["timestamps"].astype(int)
        depths = video_data['depths']
        valid_depth_masks = video_data['valid_depth_masks']
        for i, valid_depth_mask in enumerate(valid_depth_masks):
            num = np.count_nonzero(valid_depth_mask == False)
            if num > 0:
                print(f"video {i} has {num} invalid depth pixels")

        # poses = torch.from_numpy(video_data['poses']).to(self.device)

        full_traj_path = os.path.join(self.save_dir, "traj/est_poses_full.txt")
        pose_data = np.loadtxt(full_traj_path, delimiter=' ', dtype=np.unicode_)
        pose_vecs = pose_data[:, 1:].astype(np.float64)
        def pose_matrix_from_quaternion(pvec):
            """ convert 4x4 pose matrix to (t, q) """
            from scipy.spatial.transform import Rotation

            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
            pose[:3, 3] = pvec[:3]
            return pose
        
        full_poses = []
        for i in range(len(pose_vecs)):
            full_poses.append(pose_matrix_from_quaternion(pose_vecs[i]))

        self.mapper.iteration_count = 30000
        self.mapper.iterations_after_densify_or_reset = 0
        self.mapper.cameras = {}
        self.mapper.is_kf = {}
        self.mapper.frame_count_log = {}

        self.mapper.video_idxs = []

        from src.utils.mono_priors.img_feature_extractors import predict_img_features, get_feature_extractor

        for video_idx in tqdm(range(len(self.mapper.frame_reader))):
            pose = torch.from_numpy(full_poses[video_idx]).to(self.device).inverse()

            timestamp = video_idx
            if self.mapper.config["mapping"]["full_resolution"]:
                color = (
                    self.mapper.frame_reader.get_color_full_resol(timestamp)
                    .to(self.device)
                    .squeeze()
                )
                load_feature_suffix = "full"
            else:
                color = self.mapper.frame_reader.get_color(timestamp).to(self.device).squeeze()
                load_feature_suffix = ""

            # Load features if uncertainty-aware
            if self.mapper.uncertainty_aware:
                feat_path = f"{self.save_dir}/mono_priors/features/{timestamp:05d}{load_feature_suffix}.npy"
                if os.path.exists(feat_path):
                    features = load_img_feature(
                        timestamp, self.save_dir, suffix=load_feature_suffix
                    ).to(self.device)
                else:
                    if self.tracker is None:
                        self.tracker = Tracker(self, pipe)
                    features = predict_img_features(self.tracker.motion_filter.feat_extractor,timestamp,color.unsqueeze(0),self.cfg,self.device)
            else:
                features = None

            # Get estimated depth and camera pose


            # Prepare data dictionary for Camera initialization
            camera_data = {
                "idx": video_idx,
                "gt_color": color,
                "est_depth": None,
                "est_pose": pose,
                "features": features,
            }

            # Initialize Camera object
            viewpoint = Camera.init_from_dataset(
                self.mapper.frame_reader,
                camera_data,
                self.mapper.projection_matrix,
                full_resol=self.mapper.config["mapping"]["full_resolution"],
            )

            # Update camera pose and compute gradient mask
            # The Camera class is based on MonoGS and
            # init_from_dataset function only updates the ground truth pose
            viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
            viewpoint.compute_grad_mask(self.mapper.config)
            
            self.mapper.cameras[video_idx] = viewpoint
            self.mapper.video_idxs.append(video_idx)
            self.mapper.is_kf[video_idx] = False
            self.mapper.frame_count_log[video_idx] = 0

        if os.path.exists(f"{self.save_dir}/viewpoints_all.ckpt") and 0: # ttt
            viewpoints_data = torch.load(os.path.join(self.save_dir, "viewpoints_all.ckpt"))
            T_starts = viewpoints_data["opt_starts"]
            T_ends = viewpoints_data["opt_ends"]
            exposures = viewpoints_data["exposures"]
            for T_start, T_end, exposure, viewpoint in tqdm(zip(T_starts, T_ends, exposures, self.mapper.cameras.values())):
                viewpoint.T_start = T_start[:3, 3]
                viewpoint.T_end = T_end[:3, 3]
                viewpoint.R_start = T_start[:3, :3]
                viewpoint.R_end = T_end[:3, :3]
                viewpoint.exposure_a.data = exposure[0].data
                viewpoint.exposure_b.data = exposure[1].data
        else:
            viewpoints_data = torch.load(os.path.join(self.save_dir, "viewpoints.ckpt"))

            opt_starts = viewpoints_data["opt_starts"]
            opt_ends = viewpoints_data["opt_ends"]
            exposures = viewpoints_data["exposures"]

            for video_idx,(opt_start, opt_end, exposure) in enumerate(zip(opt_starts, opt_ends, exposures)):
                timestamp = timestamps[video_idx]
                viewpoint = self.mapper.cameras[timestamp]

                viewpoint.opt_rot_start.data = opt_start[:4]
                viewpoint.opt_trans_start.data = opt_start[4:]
                viewpoint.opt_rot_end.data = opt_end[:4]
                viewpoint.opt_trans_end.data = opt_end[4:]
                        
                viewpoint.exposure_a.data = exposure[0].data
                viewpoint.exposure_b.data = exposure[1].data

            for viewpoint in tqdm(self.mapper.cameras.values()):
                video_idx = viewpoint.uid

                if video_idx in timestamps:
                    if get_all:
                        ttt=0
                    else:
                        continue

                else:
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

        for param in self.uncer_network.parameters():
            param.requires_grad_(False)

        self.mapper.gaussians._xyz.requires_grad_(False)
        self.mapper.gaussians._features_dc.requires_grad_(False)
        self.mapper.gaussians._features_rest.requires_grad_(False)
        self.mapper.gaussians._opacity.requires_grad_(False)
        self.mapper.gaussians._scaling.requires_grad_(False)
        self.mapper.gaussians._rotation.requires_grad_(False)

        if get_all:
            outlier_timestamps = []
        else:
            outlier_timestamps = timestamps
        
        self.mapper.lpips_fn = lpips.LPIPS(net="alex", spatial=True).to(self.mapper.device)
        self.mapper.get_all_nodes(iter_per_frame=iter_per_frame, outlier_timestamps=outlier_timestamps)
        

        opt_starts_list = []
        opt_ends_list = []
        exposures = []
        is_kfs = []
        for video_idx, viewpoint in self.mapper.cameras.items():
            opt_starts_list.append(torch.cat([viewpoint.opt_rot_start.data, viewpoint.opt_trans_start.data]))
            opt_ends_list.append(torch.cat([viewpoint.opt_rot_end.data, viewpoint.opt_trans_end.data]))

            exposures.append([viewpoint.exposure_a.data, viewpoint.exposure_b.data])
            is_kfs.append(self.mapper.is_kf[video_idx])

        data = {"opt_starts": opt_starts_list, "opt_ends": opt_ends_list, "exposures": exposures, "is_kfs": is_kfs}

        torch.save(data, f"{self.save_dir}/viewpoints_all.ckpt")

        self.printer.terminate()

        for process in mp.active_children():
            process.terminate()
            process.join()

def gen_pose_matrix(R, T):
    pose = np.eye(4)
    pose[0:3, 0:3] = R.cpu().numpy()
    pose[0:3, 3] = T.cpu().numpy()
    return pose
