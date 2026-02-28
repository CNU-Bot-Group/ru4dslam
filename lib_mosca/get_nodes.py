import torch
import os, os.path as osp
import logging
import numpy as np
from tqdm import tqdm
from typing import Literal, Optional, Tuple
from pytorch3d.ops import knn_points

from lib_moca.camera import MonocularCameras
from lib_mosca.dynamic_gs import DynSCFGaussian
from lib_mosca.static_gs import StaticGaussian

from eval_utils.eval_nvidia import eval_nvidia_dir
from eval_utils.eval_dyncheck import eval_dycheck
from eval_utils.eval_sintel_cam import eval_sintel_campose
from eval_utils.eval_tum_cam import eval_metrics as eval_tum_campose
from eval_utils.eval_tum_cam import c2w_to_tumpose, load_traj as load_tum_traj

from eval_utils.campose_alignment import align_ate_c2b_use_a2b
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion

import imageio
from omegaconf import OmegaConf
from data_utils.iphone_helpers import load_iphone_gt_poses
from data_utils.nvidia_helpers import load_nvidia_gt_pose, get_nvidia_dummy_test

from lib_render.render_helper import render, render_cam_pcl
from tqdm import tqdm
import imageio
from matplotlib import pyplot as plt
import cv2 as cv
from lib_render.render_helper import GS_BACKEND
import time
import torch.nn.functional as F
from lib_mosca.misc import seed_everything
import torch.nn.functional as F
from torch import nn
from thirdparty.gaussian_splatting.utils.general_utils import (
    rotation_matrix_to_quaternion,
    quaternion_multiply,
    build_rotation,
    multiply_quaternions,
)

from gs_utils.loss_helper import (
    compute_rgb_loss,
    compute_dep_loss,
)

from lib_render.render_helper import render, render_blur
from lib_prior.prior_loading import laplacian_filter_depth

from eval_utils.dycheck_metrics import compute_psnr, compute_ssim, compute_lpips
import cv2
SEED = 12345

from scipy.spatial.transform import Rotation as R

logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)
from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel
from munch import munchify

import lpips

from eval_utils.dycheck_metrics import compute_psnr, compute_ssim, compute_lpips
from lib_mosca.gs_utils.ssim_helper import ssim


def save_ply(gs_param, path, s_model, cfg):
    if torch.is_tensor(gs_param[0]):  # direct 5 tuple
        mu, fr, s, o, sph = gs_param
    else:
        mu, fr, s, o, sph = gs_param[0]
        for i in range(1, len(gs_param)):
            mu = torch.cat([mu, gs_param[i][0]], 0)
            fr = torch.cat([fr, gs_param[i][1]], 0)
            s = torch.cat([s, gs_param[i][2]], 0)
            o = torch.cat([o, gs_param[i][3]], 0)
            sph = torch.cat([sph, gs_param[i][4]], 0)

    dc = sph[:, :3].view(sph.shape[0], 3, 1)
    rest = sph[:, 3:].view(sph.shape[0], 3, -1)
    sph = torch.cat([dc, rest], dim=-1)
    n = sph.shape[-1]
    if n < 16:
        sph = F.pad(sph, (0, 16-n), mode='constant', value=0)
    # sph = sph.reshape(sph.shape[0], 3, -1)
    gaussians = GaussianModel(s_model.max_sph_order)
    gaussians.init_lr(6.0)
    gaussians.training_setup(munchify(cfg["mapping"]["opt_params"]))
    gaussians.extend_from_pcd(mu, sph, s_model.s_inv_act(s), matrix_to_quaternion(fr),  s_model.o_inv_act(o), 0)
    # n = fr.shape[0]
    # unit_quaternion = torch.zeros(n, 4).to(fr)
    # unit_quaternion[:, 3] = 1.0  # 设置第一个元素为1
    # gaussians.extend_from_pcd(mu, sph, s_model.s_inv_act(s), unit_quaternion,  s_model.o_inv_act(o), 0)
    gaussians.save_ply(path)

def get_all_nodes(ws, log_path, cfg):
    seed_everything(SEED)
    is_blur = getattr(cfg, "is_blur", True)
    is_exposure = getattr(cfg, "is_exposure", True)
    # is_blur = True
    # is_blur = False
    use_ru4d_nodes = False
    is_train = True # True
    save_img = True
    lr_opt = 0.0003
    data_root = ws
    if data_root.endswith("/"):
        data_root = data_root[:-1]
    
    saved_dir = log_path
    img_save_dir = osp.join(saved_dir, "blur_imgs")
    os.makedirs(img_save_dir, exist_ok=True)
    device = torch.device("cuda")

    lpips_fn = lpips.LPIPS(net="alex", spatial=True).to(device)

    dataset_mode = getattr(cfg, "mode", "iphone")
    # max_sph_order = getattr(cfg, "max_sph_order", 1)
    logging.info(f"Dataset mode: {dataset_mode}")

    ######################################################################
    ######################################################################
    if os.path.exists(osp.join(saved_dir, "nodes_cam.pth")) and not is_train:
        cams = MonocularCameras.load_from_ckpt(
            torch.load(osp.join(saved_dir, "nodes_cam.pth"))
        ).to(device)
    else:
        cams = MonocularCameras.load_from_ckpt(
            torch.load(osp.join(saved_dir, "photometric_cam.pth"))
        ).to(device)
    s_model = StaticGaussian.load_from_ckpt(
        torch.load(
            osp.join(saved_dir, f"photometric_s_model_{GS_BACKEND.lower()}.pth")
        ),
        device=device,
    )
    d_model = DynSCFGaussian.load_from_ckpt(
        torch.load(
            osp.join(saved_dir, f"photometric_d_model_{GS_BACKEND.lower()}.pth")
        ),
        device=device,
    )

    cams.to(device)
    cams.eval()
    d_model.to(device)
    d_model.eval()
    s_model.to(device)
    s_model.eval()
    d_model.is_opa_control = getattr(cfg, "is_opa_control", True)
    
    for name, param in d_model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}, grad={param.grad is not None}")
        param.requires_grad = False
    for name, param in s_model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}, grad={param.grad is not None}")
        param.requires_grad = False

    H_out, W_out = cfg['cam']['H_out'], cfg['cam']['W_out']
    H_edge, W_edge = cfg['cam']['H_edge'], cfg['cam']['W_edge']
    H_out_with_edge, W_out_with_edge = H_out + H_edge * 2, W_out + W_edge * 2

    H, W = cfg['cam']['H'], cfg['cam']['W']
    fx, fy, cx, cy = cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
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

    # fx1 = fx/W_out_with_edge*2
    # fy1 = fy/H_out_with_edge*2

    L = min(W_out, H_out)
    fx1 = fx/L*2
    fy1 = fy/L*2

    fx1 = np.rad2deg(2 * np.arctan(1/fx1))
    fy1 = np.rad2deg(2 * np.arctan(1/fy1))
    # fy1 = fx1

    fx0 = np.rad2deg(2 * np.arctan(W_out/2.0/fx))
    fy0 = np.rad2deg(2 * np.arctan(H_out/2.0/fy))

    fx, fy = fx1, fy1

    cx = cx/W_out
    cy = cy/H_out
    
    # fx, fy, cx, cy = 53.13, 53.13, 0.5, 0.5

    video_file_name = cfg["data"]["output"] + "/" + cfg["scene"] + "/video.npz"
    from src.utils.datasets import get_dataset
    stream = get_dataset(cfg)

    npz_path = video_file_name
    offline_video = dict(np.load(npz_path))
    traj_ref = []
    traj_est = []
    video_traj = offline_video['poses']
    video_timestamps = offline_video['timestamps']
    timestamps = []

    for i in range(video_timestamps.shape[0]):
        timestamp = int(video_timestamps[i])
        # val = stream.poses[timestamp].sum()
        # if np.isnan(val) or np.isinf(val):
        #     print(f'Nan or Inf found in gt poses, skipping {i}th pose!')
        #     continue
        # traj_est.append(video_traj[i])
        # traj_ref.append(stream.poses[timestamp])
        # timestamps.append(video_timestamps[i])
        timestamps.append(timestamp)

    from evo.core.trajectory import PoseTrajectory3D

    # traj_est =PoseTrajectory3D(poses_se3=traj_est,timestamps=timestamps)
    # traj_ref =PoseTrajectory3D(poses_se3=traj_ref,timestamps=timestamps)

    # from evo.core import sync

    # traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    # r_a, t_a, s = traj_est.align(traj_ref, correct_scale=True)

    # from evo.core import metrics
    # data = (traj_ref, traj_est)
    # ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    # ape_metric.process_data(data)
    # ape_statistics = ape_metric.get_all_statistics()
    # print(ape_statistics['rmse'])

    tto_flag = False

    # gt_training_cam_T_wi = cams.T_wc_list().detach().cpu()
    # gt_testing_cam_T_wi_list = torch.from_numpy(np.stack(traj_ref.poses_se3, axis=0))

    # gt_testing_cam_T_wi_list = cams.T_wc_list().detach().cpu()
    full_traj_path = cfg["data"]["output"] + "/" + cfg["scene"] + "/traj/est_poses_full.txt"
    pose_data = np.loadtxt(full_traj_path, delimiter=' ', dtype=np.unicode_)
    pose_vecs = pose_data[:, 1:].astype(np.float64)
    def pose_matrix_from_quaternion(pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose
    
    gt_testing_cam_T_wi_list = []
    for i in range(len(pose_vecs)):
        gt_testing_cam_T_wi_list.append(pose_matrix_from_quaternion(pose_vecs[i]))


    kf_poses = []
    for i in range(video_timestamps.shape[0]):
        kf_poses.append(video_traj[i][:2, 3])
    kf_poses = np.stack(kf_poses, axis=0)

    full_poses = []
    for i in range(len(stream)):
        full_poses.append(gt_testing_cam_T_wi_list[i][:2, 3])
    full_poses = np.stack(full_poses, axis=0)

    # plt.figure(figsize=(10, 10))
    # plt.plot(kf_poses[:, 0], kf_poses[:, 1], 'b-', label='kf', linewidth=2)
    # plt.plot(full_poses[:, 0], full_poses[:, 1], 'r-', label='full', linewidth=2)
    # plt.legend()
    # plt.title('Trajectory Comparison')
    # plt.xlabel('X Position')
    # plt.ylabel('Y Position')   
    # plt.axis('equal')
    # plt.grid(True)
    # plt.savefig('trajectory_comparison.png')

    traj_est = np.stack(gt_testing_cam_T_wi_list, axis=0)
    poses_o = traj_est.copy()

    traj_est = PoseTrajectory3D(poses_se3=traj_est,timestamps=range(len(stream)))
    # cams.scale_nw = 0.58046954870224
    s = cams.scale_nw.detach().cpu()
    traj_est.scale(s)
    traj_est = np.stack(traj_est.poses_se3, axis=0)

    opt_rot_start = []
    opt_trans_start = []
    opt_rot_end = []
    opt_trans_end = []
    exposures_a_list = []
    exposures_b_list = []
    
    psnr_list = []
    ssim_list = []
    lpips_list = []

    if use_ru4d_nodes:
        if os.path.exists(cfg["data"]["output"] + "/" + cfg["scene"] + "/viewpoints_all_t.ckpt"):
            viewpoints_path = cfg["data"]["output"] + "/" + cfg["scene"] + "/viewpoints_all.ckpt"
            viepoints_data = torch.load(viewpoints_path)
        elif os.path.exists(cfg["data"]["output"] + "/" + cfg["scene"] + "/viewpoints_all.ckpt"):
            viewpoints_path = cfg["data"]["output"] + "/" + cfg["scene"] + "/viewpoints_all.ckpt"
            viepoints_data = torch.load(viewpoints_path)

        opt_starts = viepoints_data["opt_starts"]
        opt_ends = viepoints_data["opt_ends"]
        exposures = viepoints_data["exposures"]
        
        for timestamp in range(len(stream)):
            opt_rot_start.append(opt_starts[timestamp][:4])
            opt_trans_start.append(opt_starts[timestamp][4:])

            opt_rot_end.append(opt_ends[timestamp][:4])
            opt_trans_end.append(opt_ends[timestamp][4:])

            exposures_a_list.append(exposures[timestamp][0])
            exposures_b_list.append(exposures[timestamp][1])

        rot_starts = torch.stack(opt_rot_start, dim=0)
        rot_ends = torch.stack(opt_rot_end, dim=0)
        trans_starts = torch.stack(opt_trans_start, dim=0)
        trans_ends = torch.stack(opt_trans_end, dim=0)
        # for i, timestamp in enumerate(timestamps):
        #     timestamp = int(timestamp)
        #     opt_rot_start[timestamp] = cams.opt_rot_start.data[i]
        #     opt_trans_start[timestamp] = cams.opt_trans_start.data[i]
        #     opt_rot_end[timestamp] = cams.opt_rot_end.data[i]
        #     opt_trans_end[timestamp] = cams.opt_trans_end.data[i]

        #     exposures_a_list[timestamp] = cams.exposures_a.data[i]
        #     exposures_b_list[timestamp] = cams.exposures_b.data[i]

        R_starts = build_rotation(rot_starts)
        R_ends = build_rotation(rot_ends)

        poses = torch.from_numpy(poses_o).to(device)
        if poses.dim() == 2:
            poses = poses.unsqueeze(-1)  # 添加最后一个维度

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
        traj_start.scale(s)

        traj_end = PoseTrajectory3D(poses_se3=w2c_ends_np,timestamps=timestamps)
        traj_end.scale(s)

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

        exposures_a_list = torch.stack(exposures_a_list, dim=0)
        exposures_b_list = torch.stack(exposures_b_list, dim=0)

        # param_cam_q_start, param_cam_t_end = cams.__get_init_qt__(T_starts_list)
        # param_cam_q_end, param_cam_t_start = cams.__get_init_qt__(T_ends_list)

        # cams.q_wc_start = torch.nn.Parameter(param_cam_q_start, requires_grad=False)
        # cams.t_wc_start = torch.nn.Parameter(param_cam_t_start, requires_grad=False)
        # cams.q_wc_end = torch.nn.Parameter(param_cam_q_end, requires_grad=False)
        # cams.t_wc_end = torch.nn.Parameter(param_cam_t_end, requires_grad=False)
        cams.exposures_a = torch.nn.Parameter(torch.tensor(exposures_a_list), requires_grad=False)
        cams.exposures_b = torch.nn.Parameter(torch.tensor(exposures_b_list), requires_grad=False)
    elif not is_train:
        for i in range(cams.T):
            opt_rot_start.append(cams.opt_rot_start.data[i])
            opt_trans_start.append(cams.opt_trans_start.data[i])
            opt_rot_end.append(cams.opt_rot_end.data[i])
            opt_trans_end.append(cams.opt_trans_end.data[i])

            exposures_a_list.append(cams.exposures_a.data[i])
            exposures_b_list.append(cams.exposures_b.data[i])
    else:
        opt_cam_rot_start = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        opt_cam_trans_start = torch.tensor([0.0, 0.0, 0.0], device=device)

        opt_cam_rot_end = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        opt_cam_trans_end = torch.tensor([0.0, 0.0, 0.0], device=device)

        def rotation_matrix_to_axis_angle(R):
            angle = torch.acos((torch.trace(R) - 1) / 2)
            if angle < 1e-6:
                return torch.zeros(3, device=device)
            rx = R[2,1] - R[1,2]
            ry = R[0,2] - R[2,0]
            rz = R[1,0] - R[0,1]
            axis = torch.tensor([rx, ry, rz], device=device)
            axis = axis / (2 * torch.sin(angle))
            return axis * angle
            
        t = 0
        kf_idx=timestamps
        param_cam_q, param_cam_t = cams.__get_init_qt__(traj_est)
        cams.q_wc = nn.Parameter(param_cam_q)
        cams.t_wc = nn.Parameter(param_cam_t)
        cams.T = len(pose_vecs)
        for timestamp in range(len(pose_vecs)):
            if timestamp == kf_idx[t]:
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
                
                opt_rot_start.append(multiply_quaternions(opt_cam_rot_start.unsqueeze(0), perturbation1_rot.unsqueeze(0)).squeeze(0))
                opt_trans_start.append(opt_cam_trans_start + rand1_trans)
    
                opt_rot_end.append(multiply_quaternions(opt_cam_rot_end.unsqueeze(0), perturbation2_rot.unsqueeze(0)).squeeze(0))
                opt_trans_end.append(opt_cam_trans_end + rand2_trans)
            else:
                R_cur = (cams.T_cw(timestamp)[:3, :3]).to(device)
                R_last = (cams.T_cw(kf_idx[t])[:3, :3]).to(device)
                T_cur = (cams.T_cw(timestamp)[:3, 3]).to(device)
                T_last = (cams.T_cw(kf_idx[t])[:3, 3]).to(device)
                R_rel = R_cur @ (R_last).T  # 相对旋转矩阵
                T_rel = T_cur - T_last  # 相对平移向量
                rot_axis_angle = rotation_matrix_to_axis_angle(R_rel)
                rot_axis_angle = rot_axis_angle / (torch.norm(rot_axis_angle) + 1e-8)  # 归一化

                # 计算相对平移方向
                trans_direction = T_rel / (torch.norm(T_rel) + 1e-8)  # 归一化

                # 生成沿相对位姿方向的随机扰动
                trans_magnitude = (high - low) * torch.rand(1) + low
                rand1_trans = trans_direction * trans_magnitude.to(device)
                rand1_trans = rand1_trans.to(device)
                rand2_trans = -rand1_trans

                # 对于旋转，沿相对旋转轴方向添加扰动
                rot_magnitude = (high - low) * torch.rand(1) + low
                rand1_rot = rot_axis_angle * rot_magnitude.to(device)
                rand1_rot = rand1_rot.to(device)
                rand2_rot = -rand1_rot

                # 构建旋转扰动四元数
                perturbation1_rot = torch.cat([
                    torch.ones_like(rand1_rot[..., :1]),
                    rand1_rot
                ], dim=-1).to(device)

                perturbation2_rot = torch.cat([
                    torch.ones_like(rand2_rot[..., :1]),
                    rand2_rot
                ], dim=-1).to(device)

                # 应用扰动到初始位姿（opt_rot_start是单位四元数，trans_start是零向量）
                opt_rot_start.append(multiply_quaternions(
                    opt_cam_rot_start.unsqueeze(0),
                    perturbation1_rot.unsqueeze(0)
                ).squeeze(0))
                opt_trans_start.append(opt_cam_trans_start + rand1_trans)

                # 应用扰动到结束位姿
                opt_rot_end.append(multiply_quaternions(
                    opt_cam_rot_end.unsqueeze(0),
                    perturbation2_rot.unsqueeze(0)
                ).squeeze(0))
                opt_trans_end.append(opt_cam_trans_end + rand2_trans)
                
            if t + 1 < len(kf_idx) and timestamp == kf_idx[t+1]:
                t += 1
        ### exp use this
        # for timestamp in range(len(stream)):
        #     low, high = 0.001, 0.003
        #     rand1_trans = (high - low) * torch.rand(3) + low
        #     rand1_trans = rand1_trans.to(device)
            
        #     rand2_trans = -rand1_trans

        #     rand1_rot = (high - low) * torch.rand(3) + low
        #     rand1_rot = rand1_rot.to(device)
        #     perturbation1_rot = torch.cat([
        #         torch.ones_like(rand1_rot[..., :1]),
        #         rand1_rot
        #     ], dim=-1).to(device)
            
        #     rand2_rot = -rand1_rot
        #     perturbation2_rot = torch.cat([
        #         torch.ones_like(rand2_rot[..., :1]),
        #         rand2_rot
        #     ], dim=-1).to(device)
            
        #     opt_rot_start.append(multiply_quaternions(opt_cam_rot_start.unsqueeze(0), perturbation1_rot.unsqueeze(0)).squeeze(0))
        #     opt_trans_start.append(opt_cam_trans_start + rand1_trans)

        #     opt_rot_end.append(multiply_quaternions(opt_cam_rot_end.unsqueeze(0), perturbation2_rot.unsqueeze(0)).squeeze(0))
        #     opt_trans_end.append(opt_cam_trans_end + rand2_trans)
        ###

            exposures_a_list.append(torch.tensor([0.0], device=device))
            exposures_b_list.append(torch.tensor([0.0], device=device))

        for i, timestamp in enumerate(timestamps):
            timestamp = int(timestamp)
            opt_rot_start[timestamp] = cams.opt_rot_start.data[i]
            opt_trans_start[timestamp] = cams.opt_trans_start.data[i]
            opt_rot_end[timestamp] = cams.opt_rot_end.data[i]
            opt_trans_end[timestamp] = cams.opt_trans_end.data[i]

            exposures_a_list[timestamp] = cams.exposures_a.data[i]
            exposures_b_list[timestamp] = cams.exposures_b.data[i]

    opt_rot_start = torch.stack(opt_rot_start, dim=0)
    opt_trans_start = torch.stack(opt_trans_start, dim=0)
    opt_rot_end = torch.stack(opt_rot_end, dim=0)
    opt_trans_end = torch.stack(opt_trans_end, dim=0)

    exposures_a_list = torch.stack(exposures_a_list, dim=0)
    exposures_b_list = torch.stack(exposures_b_list, dim=0)

    param_cam_q, param_cam_t = cams.__get_init_qt__(traj_est)
    cams.q_wc = nn.Parameter(param_cam_q)
    cams.t_wc = nn.Parameter(param_cam_t)

    cams.opt_rot_start = nn.Parameter(opt_rot_start, requires_grad=True)
    cams.opt_trans_start = nn.Parameter(opt_trans_start, requires_grad=True)
    cams.opt_rot_end = nn.Parameter(opt_rot_end, requires_grad=True)
    cams.opt_trans_end = nn.Parameter(opt_trans_end, requires_grad=True)

    cams.exposures_a = torch.nn.Parameter(torch.tensor(exposures_a_list), requires_grad=True)
    cams.exposures_b = torch.nn.Parameter(torch.tensor(exposures_b_list), requires_grad=True)
    
    H, W = int(cams.default_H), int(cams.default_W)

    focal = cams.rel_focal

    cxcy_ratio = cams.cxcy_ratio

    L = min(H, W)
    f = focal * L / 2.0
    fx = f[0]
    fy = f[1]
    cx = W * cxcy_ratio[0]
    cy = H * cxcy_ratio[1]
    K = torch.eye(3).to(device)
    K[0, 0] = K[0, 0] * 0 + fx
    K[1, 1] = K[1, 1] * 0 + fy
    K[0, 2] = K[0, 2] * 0 + cx
    K[1, 2] = K[1, 2] * 0 + cy




    # random_bg=getattr(fit_cfg, "photo_random_bg", True),
    default_bg_color = getattr(cfg, "photo_default_bg_color", [1.0, 1.0, 1.0])
    bg_color = default_bg_color
    
    if GS_BACKEND in ["native_add3"]:
        # the render internally has another protection, because if not set, the grad has bug
        bg_color += [0.0, 0.0, 0.0]

    rgb_sup_mask = torch.ones((H, W), dtype=torch.float64, device=device)
    depth_boundary_th = getattr(cfg, "depth_boundary_th_fit", 1.0)
    depth_min=1e-3
    depth_max=1000.0

    cams = cams.to(device)
    with open (saved_dir + "/psnr.txt", "w+") as f:
        pass
    with open (saved_dir + "/ssim.txt", "w+") as f:
        pass
    with open (saved_dir + "/lpips.txt", "w+") as f:
        pass



    # gs5 = [s_model(), d_model(0)]

    # render_dict = render(
    #     gs5,
    #     H,
    #     W,
    #     K,
    #     T_cw=cams.T_cw(0).to(device),
    # )

    # rendered = torch.clamp(render_dict["rgb"], 0.0, 1.0)
    # rendered_np = (rendered.detach().permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
    # rendered_np = cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR)

    # cv2.imwrite("t0.png", rendered_np)

    # d_model_t = d_model(test_camera_tid[0])
    # for t in tqdm(range(4, 50, 5)):
    #     working_t = test_camera_tid[t]

    #     mu, fr, s, o, sph = d_model_t
    #     gs_param = d_model(working_t)

    #     mu = torch.cat([mu, gs_param[0]], 0)
    #     fr = torch.cat([fr, gs_param[1]], 0)
    #     s = torch.cat([s, gs_param[2]], 0)
    #     o = torch.cat([o, gs_param[3]], 0)
    #     sph = torch.cat([sph, gs_param[4]], 0)
    #     d_model_t = [mu, fr, s, o, sph]

    # gs5 = [s_model(), d_model_t]

    # save_ply(gs5, "./point_cloud.ply", s_model, cfg)

    # T_wc0 = cams.T_wc(0).to(device)

    # while 1:
    #     x = input("x:")
    #     y = input("y:")
    #     z = input("z:")
    #     rx = input("rx:")
    #     ry = input("ry:")
    #     rz = input("rz:")

    #     t = T_wc0[:3, 3] + torch.tensor([float(x), float(y), float(z)], device=device)
    #     euler = [float(rx), float(ry), float(rz)]
    #     dr = R.from_euler('xyz', euler, degrees=True).as_matrix()
    #     rot = torch.matmul(torch.tensor(dr, dtype=torch.float32, device=device), T_wc0[:3, :3])
    #     T_wc0[:3, :3] = rot
    #     T_wc0[:3, 3] = t

    #     T_cw0 = torch.inverse(T_wc0)

    #     render_dict = render(
    #         gs5,
    #         H,
    #         W,
    #         K,
    #         T_cw=T_cw0,
    #     )

    #     rendered = torch.clamp(render_dict["rgb"], 0.0, 1.0)
    #     rendered_np = (rendered.detach().permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
    #     rendered_np = cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR)

    #     cv2.imwrite("t.png", rendered_np)
    
    cams.T = len(cams.exposures_a)
    test_camera_tid = np.arange(cams.T)
    t = 0
    iter_per_frame = 30
    kf_idx = timestamps

    for i in tqdm(range(len(stream))):
        # pre_time = time.time()
        with torch.no_grad():
            if i == kf_idx[t]:
                working_t = test_camera_tid[t]

                gs5 = [s_model(), d_model(working_t)]

            elif t + 1 < len(kf_idx) and i == kf_idx[t+1]:
                t += 1
                working_t = test_camera_tid[t]

                gs5 = [s_model(), d_model(working_t)]

            elif t + 1 < len(kf_idx):
                dt = (i - kf_idx[t])/(kf_idx[t+1] - kf_idx[t])
                working_t = test_camera_tid[t]
                params_0 = d_model(working_t)
                working_t = test_camera_tid[t+1]
                params_1 = d_model(working_t)

                R0 = params_0[1]
                R1 = params_1[1]
                q0 = matrix_to_quaternion(R0)
                q1 = matrix_to_quaternion(R1)

                dot = (q0 * q1).sum(dim=-1)
                q1 = torch.where(dot.unsqueeze(-1) < 0, -q1, q1)
                dot = torch.abs(dot)

                theta = torch.acos(torch.clamp(dot, -1.0 + 1e-7, 1.0 - 1e-7))

                sin_theta = torch.sin(theta)
                mask = sin_theta > 1e-5
                w0 = torch.where(mask, torch.sin((1 - dt) * theta) / sin_theta, 1 - dt)
                w1 = torch.where(mask, torch.sin(dt * theta) / sin_theta, dt)

                q_interp = w0.unsqueeze(-1) * q0 + w1.unsqueeze(-1) * q1

                q_interp_norm = F.normalize(q_interp, p=2, dim=-1)

                rot = quaternion_to_matrix(q_interp_norm)

                mu = params_0[0] * (1 - dt) + params_1[0] * dt
                
                o = params_0[3] * (1 - dt) + params_1[3] * dt

                d_params =[mu, rot, params_0[2], o, params_0[4]]

                gs5 = [s_model(), d_params]
                
            elif t + 1 > len(kf_idx):
                dt = (kf_idx[t] - kf_idx[t-1])/(i - kf_idx[t])
                working_t = test_camera_tid[t-1]
                params_0 = d_model(working_t)
                working_t = test_camera_tid[t]
                params_1 = d_model(working_t)

                R0 = params_0[1]
                R1 = params_1[1]
                q0 = matrix_to_quaternion(R0)
                q1 = matrix_to_quaternion(R1)

                dot = (q0 * q1).sum(dim=-1)
                q1 = torch.where(dot.unsqueeze(-1) < 0, -q1, q1)
                dot = torch.abs(dot)

                theta = torch.acos(torch.clamp(dot, -1.0 + 1e-7, 1.0 - 1e-7))

                sin_theta = torch.sin(theta)
                mask = sin_theta > 1e-5
                w0 = torch.where(mask, torch.sin((1 - dt) * theta) / sin_theta, 1 - dt)
                w1 = torch.where(mask, torch.sin(dt * theta) / sin_theta, dt)

                q_interp = w0.unsqueeze(-1) * q0 + w1.unsqueeze(-1) * q1

                q_interp_norm = F.normalize(q_interp, p=2, dim=-1)

                rot = quaternion_to_matrix(q_interp_norm)

                mu = params_1[0] + (params_1[0] - params_0[0]) * dt 

                o = params_0[3] * (1 - dt) + params_1[3] * dt

                d_params =[mu, rot, params_0[2], o, params_0[4]]

                gs5 = [s_model(), d_params]
            
            if is_blur:
                gs_param=gs5
                if torch.is_tensor(gs_param[0]):  # direct 5 tuple
                    mu, fr, s, o, sph = gs_param
                else:
                    # mu, fr, s, o, sph = gs_param[0]
                    override_rotation = matrix_to_quaternion(gs_param[0][1])
                    for j in range(1, len(gs_param)):
                        override_rotation = torch.cat([override_rotation, matrix_to_quaternion(gs_param[j][1])])
            # print(f"get gs time: {time.time() - pre_time}")

            _, gt_image, gt_depth, _ = stream[i]
            dep_mask, _ = laplacian_filter_depth(gt_depth.detach().cpu().numpy(), depth_boundary_th, 5)
            dep_mask = torch.from_numpy(dep_mask).to(device)
            gt_depth = gt_depth.to(device)
            gt_image = gt_image.squeeze(0).permute(1, 2, 0).to(device)
            dep_sup_mask = dep_mask.squeeze(-1) * (gt_depth > depth_min) * (gt_depth < depth_max)
            gt_depth = torch.clamp(gt_depth, depth_min, depth_max)

            opt_cam_rot_start = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, requires_grad=True))
            opt_cam_trans_start = nn.Parameter(torch.tensor([0.0, 0.0, 0.0], device=device, requires_grad=True))
            opt_cam_rot_end = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, requires_grad=True))
            opt_cam_trans_end = nn.Parameter(torch.tensor([0.0, 0.0, 0.0], device=device, requires_grad=True))

            exposure_a = nn.Parameter(torch.tensor([0.0], device=device, requires_grad=True))
            exposure_b = nn.Parameter(torch.tensor([0.0], device=device, requires_grad=True))

            opt_cam_rot_start.data = cams.opt_rot_start[i].data.detach().clone()
            opt_cam_trans_start.data = cams.opt_trans_start[i].data.detach().clone()
            opt_cam_rot_end.data = cams.opt_rot_end[i].data.detach().clone()
            opt_cam_trans_end.data = cams.opt_trans_end[i].data.detach().clone()
        if i != 0 and is_train:
            exposure_a.data = cams.exposures_a[i-1].data.detach().clone()
            exposure_b.data = cams.exposures_b[i-1].data.detach().clone()
        if is_blur and not use_ru4d_nodes and is_train:

            cam_param_list = []
            cam_param_list.append(
                {
                    "params": [opt_cam_rot_start],
                    "lr": lr_opt,
                    "name": "opt_cam_rot_start",
                }
            )
            cam_param_list.append(
                {
                    "params": [opt_cam_trans_start],
                    "lr": lr_opt,
                    "name": "opt_cam_trans_start",
                }
            )
            cam_param_list.append(
                {
                    "params": [opt_cam_rot_end],
                    "lr": lr_opt,
                    "name": "opt_cam_rot_end",
                }
            )
            cam_param_list.append(
                {
                    "params": [opt_cam_trans_end],
                    "lr": lr_opt,
                    "name": "opt_cam_trans_end",
                }
            )

            if i != 0 and is_exposure:
                cam_param_list.append(
                    {
                        "params": [exposure_a],
                        "lr": 0.01,
                        "name": "exposures_a",
                    }
                )
                cam_param_list.append(
                    {
                        "params": [exposure_b],
                        "lr": 0.01,
                        "name": "exposures_b",
                    }
                )
            optimizer = torch.optim.Adam(cam_param_list)
        elif not use_ru4d_nodes:
            cam_param_list = []
            cam_param_list.append(
                {
                    "params": [exposure_a],
                    "lr": 0.01,
                    "name": "exposures_a",
                }
            )
            cam_param_list.append(
                {
                    "params": [exposure_b],
                    "lr": 0.01,
                    "name": "exposures_b",
                }
            )
            optimizer = torch.optim.Adam(cam_param_list)
        if not is_train:
            opt_cam_rot_start.requires_grad_(False)
            opt_cam_trans_start.requires_grad_(False)
            opt_cam_rot_end.requires_grad_(False)
            opt_cam_trans_end.requires_grad_(False)
            exposure_a.requires_grad_(False)
            exposure_b.requires_grad_(False)
            optimizer = torch.optim.Adam([torch.tensor(0., requires_grad=True)])  

        # print(f"total prev time: {time.time() - pre_time}")
        if optimizer is not None:
            for j in range(iter_per_frame):
                # total_time = time.time()
                loss = 0.0
                image_stack = []
                if is_blur:
                    render_time = time.time()
                    image_stack = render_blur(
                        gs5,
                        H,
                        W,
                        K,
                        cams.T_cw(i),
                        bg_color=bg_color,
                        opt_cam_rot_start=opt_cam_rot_start,
                        opt_cam_trans_start=opt_cam_trans_start,
                        opt_cam_rot_end=opt_cam_rot_end,
                        opt_cam_trans_end=opt_cam_trans_end,
                        override_rotation=override_rotation,
                        scale_nw=cams.scale_nw,
                    )
                    # print(f"render time: {time.time() - render_time}")
                
                render_dict = render(
                    gs5,
                    H,
                    W,
                    K,
                    T_cw=cams.T_cw(i).to(device),
                )
                image_stack.append(render_dict["rgb"])
                
                image_stack = torch.stack(image_stack, 0)

                render_dict["rgb"] = image_stack.mean(dim=0)
                
                if i != 0:
                    render_dict["rgb"] = torch.exp(exposure_a) * image_stack.mean(dim=0) + exposure_b
                
                # loss_time = time.time()
                _l_rgb, _, _, _ = compute_rgb_loss(
                    gt_image.detach().clone(), render_dict, rgb_sup_mask
                )
                # dep_sup_mask = rgb_sup_mask * dep_sup_mask
                # _l_dep, _, _, _ = compute_dep_loss(
                #     gt_depth.detach().clone(),
                #     render_dict,
                #     dep_sup_mask,
                #     st_invariant=True,
                # )

                # loss = _l_rgb + _l_dep
                
                loss = _l_rgb 

                # print(f"loss time: {time.time() - loss_time}")

                # backward_time = time.time()

                loss.backward()

                # print(f"backward time: {time.time() - backward_time}")
                step_time = time.time()
                optimizer.step()
                # print(f"step time: {time.time() - step_time}")
                optimizer.zero_grad(set_to_none=True)
                cams.zero_grad()
                # print(f"total time: {time.time() - total_time}")

            if is_blur:
                with torch.no_grad():
                    # print(cams.opt_rot_start[i].data, opt_cam_rot_start.data)
                    # cams.opt_rot_start[i].data = opt_cam_rot_start.data.clone()
                    # print(cams.opt_rot_start[i].data)

                    # print(cams.opt_trans_start[i].data, opt_cam_trans_start.data)
                    # cams.opt_trans_start[i].data = opt_cam_trans_start.data.clone()
                    # print(cams.opt_trans_start[i].data)
                    
                    # cams.opt_rot_end[i].data = opt_cam_rot_end.data.clone()
                    # cams.opt_trans_end[i].data = opt_cam_trans_end.data.clone()

                    # print(cams.opt_rot_start[i].data, opt_cam_rot_start.data)
                    cams.opt_rot_start[i].copy_(opt_cam_rot_start)
                    # print(cams.opt_rot_start[i].data)
                    cams.opt_trans_start[i].copy_(opt_cam_trans_start)
                    cams.opt_rot_end[i].copy_(opt_cam_rot_end)
                    cams.opt_trans_end[i].copy_(opt_cam_trans_end)
            if i != 0:
                with torch.no_grad():
                    # cams.exposures_a[i].data = exposure_a.data.clone()
                    # cams.exposures_b[i].data = exposure_b.data.clone()
                    cams.exposures_a[i].copy_(exposure_a)
                    cams.exposures_b[i].copy_(exposure_b)
        
        # render_psnr_time = time.time()

        # print(f"render psnr time: {time.time() - render_psnr_time}")

        with torch.no_grad():
            image_stack = []
            if is_blur:
                image_stack = render_blur(
                        gs5,
                        H,
                        W,
                        K,
                        cams.T_cw(i),
                        bg_color=bg_color,
                        opt_cam_rot_start=opt_cam_rot_start,
                        opt_cam_trans_start=opt_cam_trans_start,
                        opt_cam_rot_end=opt_cam_rot_end,
                        opt_cam_trans_end=opt_cam_trans_end,
                        scale_nw=cams.scale_nw,
                    )
            # render_o_time = time.time()
            render_dict = render(
                gs5,
                H,
                W,
                K,
                T_cw=cams.T_cw(i).to(device),
            )
            # print(f"render o time: {time.time() - render_o_time}")
            image_stack.append(render_dict["rgb"])
            
            image_stack = torch.stack(image_stack, 0)

            render_dict["rgb"] = image_stack.mean(dim=0)
            if i != 0:
                render_dict["rgb"] = torch.exp(exposure_a) * image_stack.mean(dim=0) + exposure_b

            # print(is_train, opt_cam_rot_start, opt_cam_trans_start, opt_cam_rot_end, opt_cam_trans_end, exposure_a.item(), exposure_b.item())
            # psnr_time = time.time()
            pred = torch.clamp(render_dict["rgb"], 0.0, 1.0).detach().permute(1, 2, 0).cpu().numpy()
            gt = gt_image.detach().cpu().numpy()
            full_mask = np.ones_like(gt[:, :, :1])
            
            # # SOM data has 4 channels images
            # gt = gt[..., :3]
            # pred = pred[..., :3]

            tmp_psnr = compute_psnr(gt, pred, full_mask).item()
            # tmp_ssim = compute_ssim(gt, pred, full_mask).item()
            tmp_ssim = ssim((torch.clamp(render_dict["rgb"], 0.0, 1.0))[None],
            (gt_image.permute(2, 0, 1))[None],
            ).item()
            tmp_lpips = compute_lpips(
                lpips_fn, gt, pred, full_mask, device=device
            ).item()
            psnr_list.append(tmp_psnr)
            ssim_list.append(tmp_ssim)
            lpips_list.append(tmp_lpips)

            with open (saved_dir + "/psnr.txt", "a+") as f:
                f.write(f"{str(tmp_psnr)} {str(np.mean(psnr_list))}")
                f.write("\n")
            with open (saved_dir + "/ssim.txt", "a+") as f:
                f.write(f"{str(tmp_ssim)} {str(np.mean(ssim_list))}")
                f.write("\n")
            with open (saved_dir + "/lpips.txt", "a+") as f:
                f.write(f"{str(tmp_lpips)} {str(np.mean(lpips_list))}")
                f.write("\n")

            # print(f"psnr time: {time.time() - psnr_time}")
            
            
            rendered = torch.clamp(render_dict["rgb"], 0.0, 1.0)
            if save_img:
                
                rendered_np = (rendered.detach().permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
                gt_np = (gt_image.detach().cpu().numpy()*255).astype(np.uint8)
                # save_time = time.time()
                fig, axes = plt.subplots(1, 3, figsize=(12, 6))
                axes[0].imshow(rendered_np)
                axes[0].set_title(f"{len(image_stack)} Rendered {i}, PSNR: {tmp_psnr:.2f}, SSIM: {tmp_ssim:.3f}, LPIPS: {tmp_lpips:.3f}")
                axes[1].imshow(gt_np)
                axes[1].set_title(f"GT {i}, psnr_mean{np.mean(psnr_list)}")
                
                rendered_gray = cv2.cvtColor(rendered_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
                gt_gray = cv2.cvtColor(gt_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
                diff_rgb = np.abs(rendered_gray - gt_gray)/255.0
                axes[2].imshow(diff_rgb, cmap="jet", vmin=0.0, vmax=1.0)
                axes[2].set_title(f"Diff {i}")

                for ax in axes:
                    ax.axis("off")
                if is_blur:
                    prefix = "_blur"
                else:
                    prefix = "_noblur"
                plt.savefig(osp.join(img_save_dir, f"{i:03d}{prefix}.png"))
                plt.close()
                # print(f"save time: {time.time() - save_time}")

    print("PSNR: ", np.mean(psnr_list))
    print("SSIM: ", np.mean(ssim_list))
    print("LPIPS: ", np.mean(lpips_list))

    # s_save_fn = osp.join(
    #     log_path, f"nodes_s_model_{GS_BACKEND.lower()}.pth"
    # )
    # torch.save(s_model.state_dict(), s_save_fn)

    torch.save(cams.state_dict(), osp.join(log_path, f"nodes_cam.pth"))

    # d_save_fn = osp.join(
    #     log_path, f"nodes_d_model_{GS_BACKEND.lower()}.pth"
    # )
    # torch.save(d_model.state_dict(), d_save_fn)


