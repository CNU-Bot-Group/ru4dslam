import torch
import os.path as osp
import logging
from omegaconf import OmegaConf

from lib_prior.prior_loading import Saved2D
from lib_moca.moca import moca_solve
from lib_moca.camera import MonocularCameras

from ru4d_evaluate import test_tum_cam, test_sintel_cam

from data_utils.iphone_helpers import load_iphone_gt_poses
from data_utils.nvidia_helpers import load_nvidia_gt_pose, get_nvidia_dummy_test

from recon_utils import (
    seed_everything,
    setup_recon_ws,
    auto_get_depth_dir_tap_mode,
    SEED,
)

import numpy as np
from src.utils.datasets import get_dataset
from thirdparty.gaussian_splatting.utils.general_utils import (
    rotation_matrix_to_quaternion,
    quaternion_multiply,
    build_rotation,
)
import cv2
import glob

def load_gt_cam(ws, fit_cfg):
    mode = getattr(fit_cfg, "mode", "iphone")
    logging.info(f"Loading gt camera poses in mode {mode}")
    if mode == "iphone":
        return load_iphone_gt_poses(ws, t_subsample=getattr(fit_cfg, "t_subsample", 1))
    elif mode == "nvidia":
        (gt_training_cam_T_wi, gt_training_fov, gt_training_cxcy_ratio) = (
            load_nvidia_gt_pose(osp.join(ws, "poses_bounds.npy"))
        )

        (
            gt_testing_cam_T_wi_list,
            gt_testing_tids_list,
            gt_testing_fns_list,
            gt_testing_fov_list,
            gt_testing_cxcy_ratio_list,
        ) = get_nvidia_dummy_test(gt_training_cam_T_wi, gt_training_fov)
        return (
            gt_training_cam_T_wi,
            gt_testing_cam_T_wi_list,
            gt_testing_tids_list,
            gt_testing_fns_list,
            gt_training_fov,
            gt_testing_fov_list,
            gt_training_cxcy_ratio,
            gt_testing_cxcy_ratio_list,
        )
    else:
        raise RuntimeError(f"Unknown mode: {mode}")
    return


def static_reconstruct(ws, log_path, fit_cfg):
    seed_everything(SEED)
    DEPTH_DIR, TAP_MODE = auto_get_depth_dir_tap_mode(ws, fit_cfg)
    DEPTH_BOUNDARY_TH = getattr(fit_cfg, "depth_boundary_th_fit", 1.0)
    INIT_GT_CAMERA_FLAG = getattr(fit_cfg, "init_gt_camera", False)
    DEP_MEDIAN = getattr(fit_cfg, "dep_median", 1.0)

    EPI_TH = getattr(fit_cfg, "ba_epi_th", getattr(fit_cfg, "epi_th", 1e-3))
    logging.info(f"Static BA with EPI_TH={EPI_TH}")

    TEST_TRAINING_TIME_MODE = getattr(fit_cfg, "test_training_time_mode", False)

    print(f"Static BA with EPI_TH={EPI_TH}")
    device = torch.device("cuda:0")

    video_file_name = fit_cfg["data"]["output"] + "/" + fit_cfg["scene"] + "/video.npz"
    npz_path = video_file_name
    offline_video = dict(np.load(npz_path))
    video_timestamps = offline_video['timestamps']

    s2d: Saved2D = (
        Saved2D(ws)
        .load_epi()
        .load_dep(DEPTH_DIR, DEPTH_BOUNDARY_TH, npz_path=npz_path if fit_cfg["dep_mode"] == "ru4d" else None) # 'depthcrafter_depth', 0.3 # , cfg = fit_cfg, npz_path=npz_path
        .normalize_depth(median_depth=DEP_MEDIAN)
        .recompute_dep_mask(depth_boundary_th=DEPTH_BOUNDARY_TH)
        .load_track(
            f"*uniform*{TAP_MODE}",
            min_valid_cnt=getattr(fit_cfg, "tap_loading_min_valid_cnt", 4),
        )
        .load_vos()
    )

    H_out, W_out = fit_cfg['cam']['H_out'], fit_cfg['cam']['W_out']
    H_edge, W_edge = fit_cfg['cam']['H_edge'], fit_cfg['cam']['W_edge']
    H_out_with_edge, W_out_with_edge = H_out + H_edge * 2, W_out + W_edge * 2

    H, W = fit_cfg['cam']['H'], fit_cfg['cam']['W']
    fx, fy, cx, cy = fit_cfg['cam']['fx'], fit_cfg['cam']['fy'], fit_cfg['cam']['cx'], fit_cfg['cam']['cy']
    intrinsic = torch.as_tensor([fx, fy, cx, cy]).float()
    intrinsic[0] *= W_out_with_edge / W
    intrinsic[1] *= H_out_with_edge / H
    intrinsic[2] *= W_out_with_edge / W
    intrinsic[3] *= H_out_with_edge / H
    intrinsic[2] -= W_edge
    intrinsic[3] -= H_edge
    fx = intrinsic[0].item()
    fy = intrinsic[1].item()
    cx = intrinsic[2].item()
    cy = intrinsic[3].item()

    L = min(W_out, H_out)
    fx1 = fx/L*2
    fy1 = fy/L*2

    fx1 = np.rad2deg(2 * np.arctan(1/fx1))
    fy1 = np.rad2deg(2 * np.arctan(1/fy1))

    fx, fy = fx1, fy1

    cx = cx/W_out
    cy = cy/H_out

    video_file_name = fit_cfg["data"]["output"] + "/" + fit_cfg["scene"] + "/video.npz"

    npz_path = video_file_name
    offline_video = dict(np.load(npz_path))
    video_timestamps = offline_video['timestamps']
    timestamps = []

    for i in range(video_timestamps.shape[0]):
        timestamp = int(video_timestamps[i])
        timestamps.append(video_timestamps[i])

    from evo.core.trajectory import PoseTrajectory3D

    data = np.load(video_file_name)
    poses = data['poses']
    poses_o = poses.copy()
    timestamps = data['timestamps']

    from evo.core import lie_algebra as lie

    traj_recon_poses = poses
    traj_recon = PoseTrajectory3D(poses_se3=traj_recon_poses,timestamps=timestamps)
    traj_recon.scale(s2d.scale_nw)
    
    poses = traj_recon.poses_se3

    if INIT_GT_CAMERA_FLAG:

        cams = MonocularCameras(
            n_time_steps=s2d.T,
            default_H=s2d.H,
            default_W=s2d.W,
            fxfycxcy=[fx, fy, cx, cy],
            delta_flag=False,
            init_camera_pose=poses,
            iso_focal=getattr(fit_cfg, "iso_focal", False),
        )

        viewpoints_path = fit_cfg["data"]["output"] + "/" + fit_cfg["scene"] + "/viewpoints.ckpt"
        viepoints_data = torch.load(viewpoints_path)
        opt_starts = viepoints_data["opt_starts"]
        opt_ends = viepoints_data["opt_ends"]
        exposures = viepoints_data["exposures"]
        rot_starts_list = []
        rot_ends_list = []
        trans_starts_list = []
        trans_ends_list = []
        exposures_a_list = []
        exposures_b_list = []

        for i in range(len(timestamps)):
            rot_starts_list.append(opt_starts[i][:4])
            rot_ends_list.append(opt_ends[i][:4])
            trans_starts_list.append(opt_starts[i][4:])
            trans_ends_list.append(opt_ends[i][4:])

            exposures_a_list.append(exposures[i][0])
            exposures_b_list.append(exposures[i][1])

        rot_starts = torch.stack(rot_starts_list, dim=0).to(device)
        rot_ends = torch.stack(rot_ends_list, dim=0).to(device)
        trans_starts = torch.stack(trans_starts_list, dim=0).to(device)
        trans_ends = torch.stack(trans_ends_list, dim=0).to(device)
        
        R_starts = build_rotation(rot_starts)
        R_ends = build_rotation(rot_ends)

        poses = torch.from_numpy(poses_o).to(device)
        if poses.dim() == 2:
            poses = poses.unsqueeze(-1) 

        batch_size = R_starts.shape[0]
        T_starts = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        T_starts[:, :3, :3] = R_starts
        T_starts[:, :3, 3] = trans_starts

        w2c_starts = torch.bmm(T_starts.inverse(), poses)
        
        T_ends = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        T_ends[:, :3, :3] = R_ends
        T_ends[:, :3, 3] = trans_ends

        w2c_ends = torch.bmm(T_ends.inverse(), poses)

        w2c_starts_np = w2c_starts.detach().cpu().numpy()
        w2c_ends_np = w2c_ends.detach().cpu().numpy()

        traj_start = PoseTrajectory3D(poses_se3=w2c_starts_np,timestamps=timestamps)
        traj_start.scale(s2d.scale_nw)

        traj_end = PoseTrajectory3D(poses_se3=w2c_ends_np,timestamps=timestamps)
        traj_end.scale(s2d.scale_nw)

        w2c_starts_scaled = torch.from_numpy(np.stack(traj_start.poses_se3, axis=0)).to(device)
        w2c_ends_scaled = torch.from_numpy(np.stack(traj_end.poses_se3, axis=0)).to(device)

        w2c_cams = torch.eye(4, device=device, dtype=torch.float64).unsqueeze(0).repeat(batch_size, 1, 1)
        w2c_cams[:, :3, :3] = build_rotation(cams.q_wc.data)
        w2c_cams[:, :3, 3] = cams.t_wc.data

        T_starts = w2c_cams @ w2c_starts_scaled.inverse()
        T_ends = w2c_cams @ w2c_ends_scaled.inverse()

        q, t = cams.__get_init_qt__(T_starts)
        cams.opt_rot_start = torch.nn.Parameter(q, requires_grad=True)
        cams.opt_trans_start = torch.nn.Parameter(t, requires_grad=True)

        q, t = cams.__get_init_qt__(T_ends)
        cams.opt_rot_end = torch.nn.Parameter(q, requires_grad=True)
        cams.opt_trans_end = torch.nn.Parameter(t, requires_grad=True)

        exposures_a_list = torch.stack(exposures_a_list, dim=0).view(-1, 1)
        exposures_b_list = torch.stack(exposures_b_list, dim=0).view(-1, 1)

        cams.exposures_a = torch.nn.Parameter(torch.tensor(exposures_a_list), requires_grad=False)
        cams.exposures_b = torch.nn.Parameter(torch.tensor(exposures_b_list), requires_grad=False)

        uncer_weight_dir = osp.join(ws, "uncertainty")
        uncer_weight = torch.load(osp.join(uncer_weight_dir, "uncer_weight.ckpt"))

        if isinstance (uncer_weight, list):
            uncer_weight = torch.stack(uncer_weight, dim=0)
        cams.uncer_weight = torch.nn.Parameter(uncer_weight, requires_grad=False)
        
        motion_mask_dir = osp.join(ws, "motion_mask")

        motion_mask_paths = glob.glob(osp.join(motion_mask_dir, "*.png"))
        motion_mask_paths.sort()

        motion_mask_list = []
        for motion_mask_path in motion_mask_paths:
            motion_mask = cv2.imread(motion_mask_path, cv2.IMREAD_GRAYSCALE)
            motion_mask = (motion_mask > 0)
            motion_mask_list.append(motion_mask)
        motion_mask_list = np.stack(motion_mask_list, 0)
        
        motion_mask_list = torch.from_numpy(np.stack(motion_mask_list)).cuda()  # T,H,W
        cams.motion_masks = torch.nn.Parameter(motion_mask_list, requires_grad=False)

        cams.scale_nw = torch.nn.Parameter(torch.tensor(s2d.scale_nw), requires_grad=False)
    else:
        cams = None

    logging.info("*" * 20 + "MoCa BA" + "*" * 20)
    cams, s2d, _ = moca_solve(
        ws=log_path,
        s2d=s2d,
        device=device,
        epi_th=EPI_TH,
        ba_total_steps= 0 if fit_cfg["dep_mode"] == "ru4d" else getattr(fit_cfg, "ba_total_steps", 2000),
        ba_switch_to_ind_step=getattr(fit_cfg, "ba_switch_to_ind_step", 500),
        ba_depth_correction_after_step=getattr(
            fit_cfg, "ba_depth_correction_after_step", 500
        ),
        ba_max_frames_per_step=32,
        static_id_mode="motion" if getattr(fit_cfg, "use_motion_mask", False) else "raft" if s2d.has_epi else "track",
        # * robust setting
        robust_depth_decay_th=getattr(fit_cfg, "robust_depth_decay_th", 2.0),
        robust_depth_decay_sigma=getattr(fit_cfg, "robust_depth_decay_sigma", 1.0),
        robust_std_decay_th=getattr(fit_cfg, "robust_std_decay_th", 0.2),
        robust_std_decay_sigma=getattr(fit_cfg, "robust_std_decay_sigma", 0.2),
        #
        gt_cam=cams,
        iso_focal=getattr(fit_cfg, "iso_focal", False),
        rescale_gt_cam_transl=getattr(fit_cfg, "rescale_gt_cam_transl", False),
        ba_lr_cam_f=getattr(fit_cfg, "ba_lr_cam_f", 0.0003) if cams is None else 0.0,
        ba_lr_dep_c=getattr(fit_cfg, "ba_lr_dep_c", 0.001),
        ba_lr_dep_s=getattr(fit_cfg, "ba_lr_dep_s", 0.001),
        ba_lr_cam_q=getattr(fit_cfg, "ba_lr_cam_q", 0.0003),
        ba_lr_cam_t=getattr(fit_cfg, "ba_lr_cam_t", 0.0003),
        #
        ba_lambda_flow=getattr(fit_cfg, "ba_lambda_flow", 1.0),
        ba_lambda_depth=getattr(fit_cfg, "ba_lambda_depth", 0.1),
        ba_lambda_small_correction=getattr(fit_cfg, "ba_lambda_small_correction", 0.03),
        ba_lambda_cam_smooth_trans=getattr(fit_cfg, "ba_lambda_cam_smooth_trans", 0.0),
        ba_lambda_cam_smooth_rot=getattr(fit_cfg, "ba_lambda_cam_smooth_rot", 0.0),
        #
        depth_filter_th=getattr(fit_cfg, "ba_depth_remove_th", -1.0),
        init_cam_with_optimal_fov_results=getattr(
            fit_cfg, "init_cam_with_optimal_fov_results", True
        ),
        # fov
        fov_search_fallback=getattr(fit_cfg, "ba_fov_search_fallback", 53.0),
        fov_search_N=getattr(fit_cfg, "ba_fov_search_N", 100),
        fov_search_start=getattr(fit_cfg, "ba_fov_search_start", 30.0),
        fov_search_end=getattr(fit_cfg, "ba_fov_search_end", 90.0),
        viz_valid_ba_points=getattr(fit_cfg, "ba_viz_valid_points", False),
        TEST_TRAINING_TIME_MODE=TEST_TRAINING_TIME_MODE,
    )  # ! S2D is changed becuase the depth is re-scaled

    datamode = getattr(fit_cfg, "mode", "iphone")
    if datamode == "sintel":
        test_func = test_sintel_cam
    elif datamode == "tum":
        test_func = test_tum_cam
    else:
        test_func = None
    if test_func is not None:
        test_func(
            cam_pth_fn=osp.join(log_path, "bundle", "bundle_cams.pth"),
            ws=ws,
            save_path=osp.join(log_path, "cam_metrics_ba.txt"),
        )

    return s2d


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("MoCa Reconstruction Camera Only")
    parser.add_argument("--ws", type=str, help="Source folder", required=True)
    parser.add_argument("--cfg", type=str, help="profile yaml file path", required=True)
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.load(args.cfg)
    cli_cfg = OmegaConf.from_dotlist([arg.lstrip("--") for arg in unknown])
    cfg = OmegaConf.merge(cfg, cli_cfg)

    logdir = setup_recon_ws(args.ws, fit_cfg=cfg)

    static_reconstruct(args.ws, logdir, cfg)
