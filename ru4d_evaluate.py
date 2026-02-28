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

from lib_render.render_helper import render, render_cam_pcl, render_blur
from tqdm import tqdm
import imageio
from matplotlib import pyplot as plt
import cv2 as cv
from lib_render.render_helper import GS_BACKEND
import time
import torch.nn.functional as F
from src.utils.datasets import get_dataset

logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)


####################
# * DyCheck PCK Eval
####################

# def blur(gs):


def compute_pck(
    kps0: np.ndarray,
    kps1: np.ndarray,
    img_wh: Tuple[int, int],
    ratio: float = 0.05,
    reduce: Optional[Literal["mean"]] = "mean",
) -> np.ndarray:
    """Compute PCK between two sets of keypoints given the threshold ratio.

    Canonical Surface Mapping via Geometric Cycle Consistency.
        Kulkarni et al., ICCV 2019.
        https://arxiv.org/abs/1907.10043

    Args:
        kps0 (jnp.ndarray): A set of keypoints of shape (J, 2) in float32.
        kps1 (jnp.ndarray): A set of keypoints of shape (J, 2) in float32.
        img_wh (Tuple[int, int]): Image width and height.
        ratio (float): A threshold ratios. Default: 0.05.
        reduce (Optional[Literal["mean"]]): Reduction method. Default: "mean".

    Returns:
        jnp.ndarray:
            if reduce == "mean", PCK of shape();
            if reduce is None, corrects of shape (J,).
    """
    dists = np.linalg.norm(kps0 - kps1, axis=-1)
    thres = ratio * max(img_wh)
    corrects = dists < thres
    if reduce == "mean":
        return corrects.mean()
    elif reduce is None:
        return corrects


def eval_pck(gt_list, pred_list, image_size, ratio):

    N = len(gt_list)
    assert N == len(pred_list)
    metrics = []
    for i in tqdm(range(N)):
        common_corrects = compute_pck(
            gt_list[i],
            pred_list[i],
            image_size,
            ratio,
            reduce=None,
        )
        metrics.append(common_corrects)
    mean_pck = np.mean(
        [it.mean() for it in metrics]
    )  # ! the teddy scene verified the mean is in this way, not the cat all mean, but first mean across all points and then across all paris
    return mean_pck, metrics


def load_gt_pck_data(gt_data_dict):
    gt_dst_pixel_list = [it["dst_pixel_gt"] for it in gt_data_dict]
    gt_src_pixel_list = [it["src_pixel"] for it in gt_data_dict]
    src_t_list = [it["src_t"] for it in gt_data_dict]
    dst_t_list = [it["dst_t"] for it in gt_data_dict]
    img_wh = gt_data_dict[0]["img_wh"]
    ratio = gt_data_dict[0]["ratio"]
    for it in gt_data_dict:
        assert (it["img_wh"] == img_wh).all()
        assert it["ratio"] == ratio
    return (
        gt_src_pixel_list,
        gt_dst_pixel_list,
        src_t_list,
        dst_t_list,
        img_wh,
        ratio,
    )


#########
# test helper
#########

@torch.no_grad()
def render_test_wild(
    H,
    W,
    cams: MonocularCameras,
    s_model: StaticGaussian,
    d_model: DynSCFGaussian,
    train_camera_T_wi,
    test_camera_T_wi,
    test_camera_tid,
    save_dir=None,
    fn_list=None,
    focal=None,
    cxcy_ratio=None,
    # cover_factor=0.3,
    kf_idx=None,
    stream=None,
):
    # prior2d: Prior2D = self.prior2d
    # device = self.device
    device = s_model.device

    # first align the camera
    solved_cam_T_wi = test_camera_T_wi
    # solved_cam_T_wi = torch.stack([cams.T_wc(i) for i in range(cams.T)], 0)
    # aligned_test_camera_T_wi = align_ate_c2b_use_a2b(
    #     traj_a=train_camera_T_wi,
    #     traj_b=solved_cam_T_wi.detach().cpu(),
    #     traj_c=test_camera_T_wi,
    # )
    
    # aligned_test_camera_T_wi_np = aligned_test_camera_T_wi.detach().cpu().numpy()
    # train_camera_T_wi_np = train_camera_T_wi.detach().cpu().numpy()
    # from evo.core.trajectory import PoseTrajectory3D
    # traj_est =PoseTrajectory3D(poses_se3=aligned_test_camera_T_wi_np,timestamps=test_camera_tid)
    # traj_ref = PoseTrajectory3D(poses_se3=train_camera_T_wi_np,timestamps=test_camera_tid)

    # from evo.core import metrics
    # data = (traj_ref, traj_est)
    # ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    # ape_metric.process_data(data)
    # ape_statistics = ape_metric.get_all_statistics()
    # print(ape_statistics['rmse'])


    # render
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if focal is None:
        focal = cams.rel_focal
    if cxcy_ratio is None:
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

    test_ret = []
    # for i in tqdm(range(len(test_camera_tid))):
    #     working_t = test_camera_tid[i]
    #     render_dict = render(
    #         [s_model(), d_model(working_t)],
    #         H,
    #         W,
    #         K,
    #         T_cw=cams.T_cw[i].to(device),
    #     )
    #     rgb = render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
    #     rgb = np.clip(rgb, 0, 1)  # ! important
    #     rgb = (rgb*255).astype(np.uint8)
    #     test_ret.append(rgb)
    #     # if save_dir:
    #     #     imageio.imwrite(osp.join(save_dir, f"{fn_list[i]}.png"), rgb)
    t = 0
    is_blur = False
    for i in tqdm(range(len(stream))):
        if i == kf_idx[t]:
            working_t = test_camera_tid[t]

            gs5 = [s_model(), d_model(working_t)]
            image_stack = []
            render_dict = render(
                gs5,
                H,
                W,
                K,
                T_cw=cams.T_cw[i].to(device),
            )
            image_stack.append(render_dict["rgb"])
            if is_blur:
                render_dict_t = render(
                    gs5,
                    H,
                    W,
                    K,
                    T_cw=cams.T_cw_start(i).to(device),
                )
                image_stack.append(render_dict_t["rgb"])

                render_dict_t = render(
                    gs5,
                    H,
                    W,
                    K,
                    T_cw=cams.T_cw_end(i).to(device),
                )
                image_stack.append(render_dict_t["rgb"])
            image_stack = torch.stack(image_stack, 0)

            render_dict["rgb"] = torch.exp(cams.exposures_a[i]) * image_stack.mean(dim=0) + cams.exposures_b[i]

        elif t + 1 < len(kf_idx) and i == kf_idx[t+1]:
            t += 1
            working_t = test_camera_tid[t]

            gs5 = [s_model(), d_model(working_t)]
            image_stack = []
            render_dict = render(
                gs5,
                H,
                W,
                K,
                T_cw=cams.T_cw[i].to(device),
            )
            image_stack.append(render_dict["rgb"])
            if is_blur:
                render_dict_t = render(
                    gs5,
                    H,
                    W,
                    K,
                    T_cw=cams.T_cw_start(i).to(device),
                )
                image_stack.append(render_dict_t["rgb"])

                render_dict_t = render(
                    gs5,
                    H,
                    W,
                    K,
                    T_cw=cams.T_cw_end(i).to(device),
                )
                image_stack.append(render_dict_t["rgb"])
            image_stack = torch.stack(image_stack, 0)

            render_dict["rgb"] = torch.exp(cams.exposures_a[i]) * image_stack.mean(dim=0) + cams.exposures_b[i]
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

            d_params =[mu, rot, params_0[2], params_0[3], params_0[4]]

            gs5 = [s_model(), d_params]
            image_stack = []
            render_dict = render(
                gs5,
                H,
                W,
                K,
                T_cw=cams.T_cw[i].to(device),
            )
            image_stack.append(render_dict["rgb"])

            if is_blur:
                render_dict_t = render(
                    gs5,
                    H,
                    W,
                    K,
                    T_cw=cams.T_cw_start(i).to(device),
                )
                image_stack.append(render_dict_t["rgb"])

                render_dict_t = render(
                    gs5,
                    H,
                    W,
                    K,
                    T_cw=cams.T_cw_end(i).to(device),
                )
                image_stack.append(render_dict_t["rgb"])

            image_stack = torch.stack(image_stack, 0)

            render_dict["rgb"] = torch.exp(cams.exposures_a[i]) * image_stack.mean(dim=0) + cams.exposures_b[i]
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

            d_params =[mu, rot, params_0[2], params_0[3], params_0[4]]

            gs5 = [s_model(), d_params]
            image_stack = []
            render_dict = render(
                gs5,
                H,
                W,
                K,
                T_cw=cams.T_cw[i].to(device),
            )
            image_stack.append(render_dict["rgb"])

            if is_blur:
                render_dict_t = render(
                    gs5,
                    H,
                    W,
                    K,
                    T_cw=cams.T_cw_start(i).to(device),
                )
                image_stack.append(render_dict_t["rgb"])

                render_dict_t = render(
                    gs5,
                    H,
                    W,
                    K,
                    T_cw=cams.T_cw_end(i).to(device),
                )
                image_stack.append(render_dict_t["rgb"])
            image_stack = torch.stack(image_stack, 0)

            render_dict["rgb"] = torch.exp(cams.exposures_a[i]) * image_stack.mean(dim=0) + cams.exposures_b[i]

        rgb = render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
        rgb = np.clip(rgb, 0, 1)  # ! important
        rgb = (rgb*255).astype(np.uint8)
        test_ret.append(rgb)
        if save_dir:
            imageio.imwrite(osp.join(save_dir, f"{fn_list[i]}.png"), rgb)
    return test_ret


@torch.no_grad()
def render_test(
    H,
    W,
    cams: MonocularCameras,
    s_model: StaticGaussian,
    d_model: DynSCFGaussian,
    train_camera_T_wi,
    test_camera_T_wi,
    test_camera_tid,
    save_dir=None,
    fn_list=None,
    focal=None,
    cxcy_ratio=None,
    # cover_factor=0.3,
):
    # prior2d: Prior2D = self.prior2d
    # device = self.device
    device = s_model.device

    # first align the camera
    solved_cam_T_wi = torch.stack([cams.T_wc(i) for i in range(cams.T)], 0)
    aligned_test_camera_T_wi = align_ate_c2b_use_a2b(
        traj_a=train_camera_T_wi,
        traj_b=solved_cam_T_wi.detach().cpu(),
        traj_c=test_camera_T_wi,
    )
    
    # aligned_test_camera_T_wi_np = aligned_test_camera_T_wi.detach().cpu().numpy()
    # train_camera_T_wi_np = train_camera_T_wi.detach().cpu().numpy()
    # from evo.core.trajectory import PoseTrajectory3D
    # traj_est =PoseTrajectory3D(poses_se3=aligned_test_camera_T_wi_np,timestamps=test_camera_tid)
    # traj_ref = PoseTrajectory3D(poses_se3=train_camera_T_wi_np,timestamps=test_camera_tid)

    # from evo.core import metrics
    # data = (traj_ref, traj_est)
    # ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    # ape_metric.process_data(data)
    # ape_statistics = ape_metric.get_all_statistics()
    # print(ape_statistics['rmse'])


    # render
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if focal is None:
        focal = cams.rel_focal
    if cxcy_ratio is None:
        cxcy_ratio = cams.cxcy_ratio

    L = min(H, W)
    fx = focal * L / 2.0
    fy = focal * L / 2.0
    cx = W * cxcy_ratio[0]
    cy = H * cxcy_ratio[1]
    K = torch.eye(3).to(device)
    fx = fx.to(device)
    fy = fy.to(device)
    cx = cx.to(device)
    cy = cy.to(device)
    K[0, 0] = K[0, 0] * 0 + fx
    K[1, 1] = K[1, 1] * 0 + fy
    K[0, 2] = K[0, 2] * 0 + cx
    K[1, 2] = K[1, 2] * 0 + cy

    test_ret = []
    for i in tqdm(range(len(test_camera_tid))):
        working_t = test_camera_tid[i]
        render_dict = render(
            [s_model(), d_model(working_t)],
            H,
            W,
            K,
            T_cw=torch.linalg.inv(aligned_test_camera_T_wi[i]).to(device),
        )
        rgb = render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
        rgb = np.clip(rgb, 0, 1)  # ! important
        test_ret.append(rgb)
        if save_dir:
            imageio.imwrite(osp.join(save_dir, f"{fn_list[i]}.png"), rgb)
    return test_ret


def render_test_tto(
    H,
    W,
    cams: MonocularCameras,
    s_model: StaticGaussian,
    d_model: DynSCFGaussian,
    train_camera_T_wi,
    test_camera_T_wi,
    test_camera_tid,
    gt_rgb_dir,
    save_pose_fn,
    ##
    tto_steps=25,
    decay_start=15,
    lr_p=0.003,
    lr_q=0.003,
    lr_final=0.0001,
    ###
    gt_mask_dir=None,
    save_dir=None,
    fn_list=None,
    focal=None,
    cxcy_ratio=None,
    # dbg
    use_sgd=False,
    loss_type="psnr",
    # boost
    initialize_from_previous_camera=True,
    initialize_from_previous_step_factor=10,
    initialize_from_previous_lr_factor=0.1,
    fg_mask_th=0.1,
):
    # * Optimize the test camera pose, nost simply do the global sim(3) alignment
    s_model.eval()
    d_model.eval()

    assert gt_mask_dir is None, "THIS IS NOT CORRECT, SHOULD NOT USE GT MASK DURING TTO"

    device = s_model.device

    # first align the camera
    with torch.no_grad():
        solved_cam_T_wi = torch.stack([cams.T_wc(i) for i in range(cams.T)], 0)
        aligned_test_camera_T_wi = align_ate_c2b_use_a2b(
            traj_a=train_camera_T_wi,
            traj_b=solved_cam_T_wi.detach().cpu(),
            traj_c=test_camera_T_wi,
        )

    # render
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    if focal is None:
        focal = cams.rel_focal
    if cxcy_ratio is None:
        cxcy_ratio = cams.cxcy_ratio

    L = min(H, W)
    fx = focal * L / 2.0
    fy = focal * L / 2.0
    cx = W * cxcy_ratio[0]
    cy = H * cxcy_ratio[1]
    cam_K = torch.eye(3).to(device)
    cam_K[0, 0] = cam_K[0, 0] * 0 + float(fx)
    cam_K[1, 1] = cam_K[1, 1] * 0 + float(fy)
    cam_K[0, 2] = cam_K[0, 2] * 0 + float(cx)
    cam_K[1, 2] = cam_K[1, 2] * 0 + float(cy)

    test_ret = []
    solved_pose_list = []
    for i in tqdm(range(len(test_camera_tid))):
        if initialize_from_previous_camera and i == 0:
            step_factor = initialize_from_previous_step_factor
            lr_factor = 1.0
        else:
            step_factor = 1
            lr_factor = initialize_from_previous_lr_factor

        working_t = test_camera_tid[i]
        # load gt rgb and mask
        gt_rgb = imageio.imread(osp.join(gt_rgb_dir, f"{fn_list[i]}.png")) / 255.0
        gt_rgb = gt_rgb[..., :3]
        if gt_mask_dir is None:
            gt_mask = np.ones_like(gt_rgb[..., 0])
        else:
            raise RuntimeError("Should not use this during TTO!!")
            gt_mask = imageio.imread(osp.join(gt_mask_dir, f"{fn_list[i]}.png")) / 255.0
        gt_rgb = torch.tensor(gt_rgb, device=device).float()
        gt_mask = torch.tensor(gt_mask, device=device).float()
        gt_mask_sum = gt_mask.sum()

        T_cw_init = torch.linalg.inv(aligned_test_camera_T_wi[i]).to(device)
        T_bottom = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
        t_init = torch.nn.Parameter(T_cw_init[:3, 3].detach())
        q_init = torch.nn.Parameter(matrix_to_quaternion(T_cw_init[:3, :3]).detach())
        if use_sgd:
            optimizer_type = torch.optim.SGD
        else:
            optimizer_type = torch.optim.Adam
        optimizer = optimizer_type(
            [
                {"params": t_init, "lr": lr_p * lr_factor},
                {"params": q_init, "lr": lr_q * lr_factor},
            ]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=tto_steps * step_factor - decay_start,
            eta_min=lr_final * lr_factor,
        )

        loss_list = []

        with torch.no_grad():
            gs5 = [s_model(), d_model(working_t)]  # ! this does not change
        for _step in range(tto_steps * step_factor):
            optimizer.zero_grad()
            _T_cw = torch.cat([quaternion_to_matrix(q_init), t_init[:, None]], 1)
            T_cw = torch.cat([_T_cw, T_bottom[None]], 0)
            render_dict = render(gs5, H, W, cam_K, T_cw=T_cw)
            pred_rgb = render_dict["rgb"].permute(1, 2, 0)
            rendered_mask = render_dict["alpha"].squeeze(-1).squeeze(0) > fg_mask_th

            if loss_type == "abs":
                raise RuntimeError("Should not use this")
                rgb_loss_i = torch.abs(pred_rgb - gt_rgb) * gt_mask[..., None]
                rgb_loss = rgb_loss_i.sum() / gt_mask_sum
            elif loss_type == "psnr":
                mse = ((pred_rgb - gt_rgb) ** 2)[rendered_mask].mean()
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                rgb_loss = -psnr

            else:
                raise ValueError(f"Unknown loss tyoe {loss_type}")

            loss = rgb_loss
            loss.backward()
            optimizer.step()
            if _step >= decay_start:
                scheduler.step()

            loss_list.append(loss.item())

        solved_T_cw = torch.cat([quaternion_to_matrix(q_init), t_init[:, None]], 1)
        solved_T_cw = torch.cat([solved_T_cw, T_bottom[None]], 0)
        solved_pose_list.append(solved_T_cw.detach().cpu().numpy())
        with torch.no_grad():

            render_dict = render(
                [s_model(), d_model(working_t)], H, W, cam_K, T_cw=T_cw
            )
            rgb = render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
            rgb = np.clip(rgb, 0, 1)  # ! important
            test_ret.append(rgb)
        if save_dir:
            imageio.imwrite(osp.join(save_dir, f"{fn_list[i]}.png"), rgb)
        logging.info(f"TTO {fn_list[i]}: {loss_list[0]:.3f}->{loss_list[-1]:.3f}")
        if initialize_from_previous_camera and i < len(test_camera_tid) - 1:
            aligned_test_camera_T_wi[i + 1] = torch.linalg.inv(solved_T_cw).to(
                aligned_test_camera_T_wi
            )
    np.savez(save_pose_fn, poses=solved_pose_list)
    return test_ret

def eval_wild(
    save_dir,
    gt_rgb_dir,
    gt_mask_dir,
    pred_dir,
    strict_eval_all_gt_flag=False,
    eval_non_masked=False,
    save_prefix="",
    viz_interval=50,
    stream=None,
    cfg=None,
    timestamps=None,
):

    import lpips
    # from ml-pgdvs
    # https://github.com/apple/ml-pgdvs/blob/13f9875eb0b7de6068452ec9c37231eda48581df/pgdvs/engines/trainer_pgdvs.py#L89
    # https://github.com/apple/ml-pgdvs/blob/13f9875eb0b7de6068452ec9c37231eda48581df/pgdvs/engines/trainer_pgdvs.py#L139
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    start_t = time.time()
    lpips_fn = lpips.LPIPS(net="alex", spatial=True).to(device)
    logging.info(f"lpips_fn init time: {time.time() - start_t:.2f}s")

    if stream is None:
        import glob
    
        dataset_type = cfg['data']['output'].split('/')[-1]

        input_folder = cfg['data']['input_folder']
        if "ROOT_FOLDER_PLACEHOLDER" in input_folder:
            input_folder = input_folder.replace("ROOT_FOLDER_PLACEHOLDER", cfg['data']['root_folder'])

        if dataset_type == "TUM_RGBD":
            color_paths = sorted(
                glob.glob(f'{input_folder}/rgb/*.png'))
        elif dataset_type == "Wild_SLAM_Mocap":
            color_paths = sorted(
                glob.glob(f'{input_folder}/rgb/frame*.png'))

        stride = cfg['stride']
        max_frames = cfg['max_frames']
        if max_frames < 0:
            max_frames = len(color_paths)

        color_paths = color_paths[:max_frames][::stride]
    else:
        gt_rgb_fns = stream
    # gt_mask_fns = sorted(os.listdir(gt_mask_dir))
    # pred_fns = sorted(os.listdir(pred_dir))
    import glob
    pred_fns = sorted(glob.glob(pred_dir + "/*.png"))
    # # debug
    # pred_fns = pred_fns[:5]

    if pred_dir.endswith("/"):
        pred_dir = pred_dir[:-1]
    eval_name = osp.basename(pred_dir)
    save_viz_dir = osp.join(save_dir, f"{eval_name}_viz")
    os.makedirs(save_viz_dir, exist_ok=True)

    assert (len(gt_rgb_fns) == len(pred_fns)), "Number of files must match"
    # if strict_eval_all_gt_flag:
    #     assert (
    #         len(gt_rgb_fns) == len(gt_mask_fns) == len(pred_fns)
    #     ), "Number of files must match"
    # else:
    #     pred_ids = [f[:-4] for f in pred_fns]
    #     if len(gt_rgb_fns) != len(pred_fns):
    #         logging.warning(
    #             f"Only eval predicted images {len(pred_ids)} < all gt {len(gt_rgb_fns)}"
    #         )
    #         assert len(gt_rgb_fns) == len(gt_mask_fns)
    #         filtered_gt_rgb_fns, filtered_gt_mask_fns = [], []
    #         for i in range(len(gt_rgb_fns)):
    #             if gt_rgb_fns[i][:-4] in pred_ids:
    #                 filtered_gt_rgb_fns.append(gt_rgb_fns[i])
    #                 filtered_gt_mask_fns.append(gt_mask_fns[i])
    #         gt_rgb_fns = filtered_gt_rgb_fns
    #         gt_mask_fns = filtered_gt_mask_fns
    #     assert (
    #         len(gt_rgb_fns) == len(gt_mask_fns) == len(pred_fns)
    #     ), "Number of files must match"

    psnr_list, ssim_list, lpips_list = [], [], []
    # mpsnr_list, mssim_list, mlpips_list = [], [], []
    for i in tqdm(range(len(gt_rgb_fns))):
    # for i in tqdm(range(len(pred_fns))):
        if stream is None:
            ht = cfg['cam']['H_out']
            wd = cfg['cam']['W_out']

            H_out, W_out = cfg['cam']['H_out'], cfg['cam']['W_out']
            H_edge, W_edge = cfg['cam']['H_edge'], cfg['cam']['W_edge']

            H_out_with_edge, W_out_with_edge = H_out + H_edge * 2, W_out + W_edge * 2
            
            down_scale = 8
            slice_h = slice(down_scale // 2 - 1, ht//down_scale*down_scale+1, down_scale)
            slice_w = slice(down_scale // 2 - 1, wd//down_scale*down_scale+1, down_scale)

            color_path = color_paths[i]
            import cv2
            color_data_fullsize = cv2.imread(color_path)
            # distortion = np.array(
            #         cfg['cam']['distortion']) if 'distortion' in cfg['cam'] else None
            # if distortion is not None:
            #     K = np.eye(3)
            #     K[0, 0], K[0, 2], K[1, 1], K[1, 2] = fx_orig, cx_orig, fy_orig, cy_orig
            #     # undistortion is only applied on color image, not depth!
            #     color_data_fullsize = cv2.undistort(color_data_fullsize, K, distortion)

            color_data = cv2.resize(color_data_fullsize, (W_out_with_edge, H_out_with_edge))
            
            if W_edge > 0:
                edge = W_edge
                color_data = color_data[:, edge:-edge, :]

            if H_edge > 0:
                edge = H_edge
                color_data = color_data[edge:-edge, :, :]

            gt = color_data.astype(float)[:, :, [2, 1, 0]] / 255.0  # bgr -> rgb, [0, 1]
        else:
            _, gt, _, _ = stream[i]
            gt = gt.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        # gt_mask = (imageio.imread(osp.join(gt_mask_dir, gt_mask_fns[i])) > 0).astype(
        #     float
        # )[..., None]
        pred = imageio.imread(pred_fns[i]).astype(float) / 255.0
        full_mask = np.ones_like(gt[:, :, :1])
        
        # SOM data has 4 channels images
        gt = gt[..., :3]
        pred = pred[..., :3]

        # pred / gt: [H, W, 3], float32, range [0, 1]
        # covis_mask: [H, W, 3], float32, range [0, 1]

        import jax

        device_cpu = jax.devices("cpu")[0]
        with jax.default_device(device_cpu):
            from eval_utils.dycheck_metrics import compute_psnr, compute_ssim, compute_lpips
            tmp_psnr = 0.0
            tmp_ssim = 0.0
            tmp_lpips = 0.0
            if eval_non_masked:
                tmp_psnr = compute_psnr(gt, pred, full_mask).item()
                # tmp_ssim = compute_ssim(gt, pred, full_mask).item()
                # tmp_lpips = compute_lpips(
                #     lpips_fn, gt, pred, full_mask, device=device
                # ).item()
            else:
                tmp_psnr = 0.0
                tmp_ssim = 0.0
                tmp_lpips = 0.0

            # with covis mask
            # tmp_mpsnr = compute_psnr(gt, pred, gt_mask).item()
            # tmp_mssim = compute_ssim(gt, pred, gt_mask).item()
            # tmp_mlpips = compute_lpips(
            #     lpips_fn, gt, pred, gt_mask, device=device
            # ).item()

        psnr_list.append(tmp_psnr)
        ssim_list.append(tmp_ssim)
        lpips_list.append(tmp_lpips)
        # mpsnr_list.append(tmp_mpsnr)
        # mssim_list.append(tmp_mssim)
        # mlpips_list.append(tmp_mlpips)

        if i % viz_interval == 0 or (timestamps is not None and i in timestamps):
            # viz
            # m_error = abs(pred - gt).max(axis=-1) * gt_mask.squeeze(-1)
            # m_error = cv.applyColorMap(
            #     (m_error * 255).astype(np.uint8), cv.COLORMAP_JET
            # )[..., [2, 1, 0]]
            error = abs(pred - gt).max(axis=-1)
            error = cv.applyColorMap((error * 255).astype(np.uint8), cv.COLORMAP_JET)[
                ..., [2, 1, 0]
            ]
            viz_img = np.concatenate(
                [gt * 255, pred * 255, error, np.zeros_like(error)], axis=1
            ).astype(np.uint8)
            imageio.imwrite(osp.join(save_viz_dir, f"{i:03d}.png"), viz_img)

    ave_psnr = np.mean(psnr_list)
    ave_ssim = np.mean(ssim_list)
    ave_lpips = np.mean(lpips_list)

    # ave_mpsnr = np.mean(mpsnr_list)
    # ave_mssim = np.mean(mssim_list)
    # ave_mlpips = np.mean(mlpips_list)

    logging.info(
        f"ave_psnr: {ave_psnr:.2f}, ave_ssim: {ave_ssim:.4f}, ave_lpips: {ave_lpips:.4f}"
    )
    # logging.info(
    #     f"ave_mpsnr: {ave_mpsnr:.2f}, ave_mssim: {ave_mssim:.4f}, ave_mlpips: {ave_mlpips:.4f}"
    # )

    import pandas as pd
    # * save and viz
    # save excel with pandas, each row is a frame
    df = pd.DataFrame(
        {
            "fn": ["AVE"],
            "psnr": [ave_psnr],
            "ssim": [ave_ssim],
            "lpips": [ave_lpips],
            # "mpsnr": [ave_mpsnr],
            # "mssim": [ave_mssim],
            # "mlpips": [ave_mlpips],
        }
    )
    for i in range(len(stream)):
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "fn": [i],
                        "psnr": [psnr_list[i]],
                        "ssim": [ssim_list[i]],
                        "lpips": [lpips_list[i]],
                        # "mpsnr": [mpsnr_list[i]],
                        # "mssim": [mssim_list[i]],
                        # "mlpips": [mlpips_list[i]],
                    }
                ),
            ],
            ignore_index=True,
        )
    df.to_excel(osp.join(save_dir, f"{save_prefix}dycheck_metrics.xlsx"), index=False)

    viz_fns = sorted(
        [f for f in os.listdir(save_viz_dir) if "tto" not in f and f.endswith("jpg")]
    )
    frames = [imageio.imread(osp.join(save_viz_dir, f)) for f in viz_fns]
    imageio.mimsave(save_viz_dir + "t.mp4", frames)
    return

def eval_wild_o(
    save_dir,
    gt_rgb_dir,
    gt_mask_dir,
    pred_dir,
    strict_eval_all_gt_flag=False,
    eval_non_masked=False,
    save_prefix="",
    viz_interval=50,
    stream=None,
):

    import lpips
    # from ml-pgdvs
    # https://github.com/apple/ml-pgdvs/blob/13f9875eb0b7de6068452ec9c37231eda48581df/pgdvs/engines/trainer_pgdvs.py#L89
    # https://github.com/apple/ml-pgdvs/blob/13f9875eb0b7de6068452ec9c37231eda48581df/pgdvs/engines/trainer_pgdvs.py#L139
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    start_t = time.time()
    lpips_fn = lpips.LPIPS(net="alex", spatial=True).to(device)
    logging.info(f"lpips_fn init time: {time.time() - start_t:.2f}s")

    gt_rgb_fns = sorted(os.listdir(gt_rgb_dir))
    gt_mask_fns = sorted(os.listdir(gt_mask_dir))
    pred_fns = sorted(os.listdir(pred_dir))

    # # debug
    # pred_fns = pred_fns[:5]

    if pred_dir.endswith("/"):
        pred_dir = pred_dir[:-1]
    eval_name = osp.basename(pred_dir)
    save_viz_dir = osp.join(save_dir, f"{eval_name}_viz")
    os.makedirs(save_viz_dir, exist_ok=True)

    if strict_eval_all_gt_flag:
        # assert (
        #     len(gt_rgb_fns) == len(gt_mask_fns) == len(pred_fns)
        # ), "Number of files must match"
        ttt=0
    else:
        pred_ids = [f[:-4] for f in pred_fns]
        if len(gt_rgb_fns) != len(pred_fns):
            logging.warning(
                f"Only eval predicted images {len(pred_ids)} < all gt {len(gt_rgb_fns)}"
            )
            assert len(gt_rgb_fns) == len(gt_mask_fns)
            filtered_gt_rgb_fns, filtered_gt_mask_fns = [], []
            for i in range(len(gt_rgb_fns)):
                if gt_rgb_fns[i][:-4] in pred_ids:
                    filtered_gt_rgb_fns.append(gt_rgb_fns[i])
                    filtered_gt_mask_fns.append(gt_mask_fns[i])
            gt_rgb_fns = filtered_gt_rgb_fns
            gt_mask_fns = filtered_gt_mask_fns
        assert (
            len(gt_rgb_fns) == len(gt_mask_fns) == len(pred_fns)
        ), "Number of files must match"

    psnr_list, ssim_list, lpips_list = [], [], []
    mpsnr_list, mssim_list, mlpips_list = [], [], []
    for i in tqdm(range(len(gt_rgb_fns))):
        gt = imageio.imread(osp.join(gt_rgb_dir, gt_rgb_fns[i])).astype(float) / 255.0
        gt_mask = (imageio.imread(osp.join(gt_mask_dir, gt_mask_fns[i])) > 0).astype(
            float
        )[..., None]
        pred = imageio.imread(osp.join(pred_dir, pred_fns[i])).astype(float) / 255.0
        full_mask = np.ones_like(gt_mask)
        
        # SOM data has 4 channels images
        gt = gt[..., :3]
        pred = pred[..., :3]

        # pred / gt: [H, W, 3], float32, range [0, 1]
        # covis_mask: [H, W, 3], float32, range [0, 1]

        import jax

        device_cpu = jax.devices("cpu")[0]
        with jax.default_device(device_cpu):
            from eval_utils.dycheck_metrics import compute_psnr, compute_ssim, compute_lpips

            if eval_non_masked:
                tmp_psnr = compute_psnr(gt, pred, full_mask).item()
                tmp_ssim = compute_ssim(gt, pred, full_mask).item()
                tmp_lpips = compute_lpips(
                    lpips_fn, gt, pred, full_mask, device=device
                ).item()
            else:
                tmp_psnr = 0.0
                tmp_ssim = 0.0
                tmp_lpips = 0.0

            # with covis mask
            tmp_mpsnr = compute_psnr(gt, pred, gt_mask).item()
            tmp_mssim = compute_ssim(gt, pred, gt_mask).item()
            tmp_mlpips = compute_lpips(
                lpips_fn, gt, pred, gt_mask, device=device
            ).item()

        psnr_list.append(tmp_psnr)
        ssim_list.append(tmp_ssim)
        lpips_list.append(tmp_lpips)
        mpsnr_list.append(tmp_mpsnr)
        mssim_list.append(tmp_mssim)
        mlpips_list.append(tmp_mlpips)

        if i % viz_interval == 0:
            # viz
            m_error = abs(pred - gt).max(axis=-1) * gt_mask.squeeze(-1)
            m_error = cv.applyColorMap(
                (m_error * 255).astype(np.uint8), cv.COLORMAP_JET
            )[..., [2, 1, 0]]
            error = abs(pred - gt).max(axis=-1)
            error = cv.applyColorMap((error * 255).astype(np.uint8), cv.COLORMAP_JET)[
                ..., [2, 1, 0]
            ]
            viz_img = np.concatenate(
                [gt * 255, pred * 255, error, m_error], axis=1
            ).astype(np.uint8)
            imageio.imwrite(osp.join(save_viz_dir, f"{gt_rgb_fns[i]}"), viz_img)

    ave_psnr = np.mean(psnr_list)
    ave_ssim = np.mean(ssim_list)
    ave_lpips = np.mean(lpips_list)

    ave_mpsnr = np.mean(mpsnr_list)
    ave_mssim = np.mean(mssim_list)
    ave_mlpips = np.mean(mlpips_list)

    logging.info(
        f"ave_psnr: {ave_psnr:.2f}, ave_ssim: {ave_ssim:.4f}, ave_lpips: {ave_lpips:.4f}"
    )
    logging.info(
        f"ave_mpsnr: {ave_mpsnr:.2f}, ave_mssim: {ave_mssim:.4f}, ave_mlpips: {ave_mlpips:.4f}"
    )

    import pandas as pd
    # * save and viz
    # save excel with pandas, each row is a frame
    df = pd.DataFrame(
        {
            "fn": ["AVE"],
            "psnr": [ave_psnr],
            "ssim": [ave_ssim],
            "lpips": [ave_lpips],
            "mpsnr": [ave_mpsnr],
            "mssim": [ave_mssim],
            "mlpips": [ave_mlpips],
        }
    )
    for i in range(len(gt_rgb_fns)):
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "fn": [gt_rgb_fns[i]],
                        "psnr": [psnr_list[i]],
                        "ssim": [ssim_list[i]],
                        "lpips": [lpips_list[i]],
                        "mpsnr": [mpsnr_list[i]],
                        "mssim": [mssim_list[i]],
                        "mlpips": [mlpips_list[i]],
                    }
                ),
            ],
            ignore_index=True,
        )
    df.to_excel(osp.join(save_dir, f"{save_prefix}dycheck_metrics.xlsx"), index=False)

    viz_fns = sorted(
        [f for f in os.listdir(save_viz_dir) if "tto" not in f and f.endswith("jpg")]
    )
    frames = [imageio.imread(osp.join(save_viz_dir, f)) for f in viz_fns]
    imageio.mimsave(save_viz_dir + ".mp4", frames)
    return

def test_main(
    cfg,
    saved_dir,
    data_root,
    device,
    tto_flag,
    eval_also_dyncheck_non_masked=False,
    skip_test_gen=False,
):
    # ! this func can be called at the end of running, or run seperately after trained

    # get cfg
    if data_root.endswith("/"):
        data_root = data_root[:-1]
    if isinstance(cfg, str):
        cfg = OmegaConf.load(cfg)
        OmegaConf.set_readonly(cfg, True)

    dataset_mode = getattr(cfg, "mode", "iphone")
    # max_sph_order = getattr(cfg, "max_sph_order", 1)
    logging.info(f"Dataset mode: {dataset_mode}")

    ######################################################################
    ######################################################################

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

    ######################################################################
    ######################################################################

    if dataset_mode == "iphone":
        (
            gt_training_cam_T_wi,
            gt_testing_cam_T_wi_list,
            gt_testing_tids_list,
            gt_testing_fns_list,
            gt_training_fov,
            gt_testing_fov_list,
            _,
            gt_testing_cxcy_ratio_list,
        ) = load_iphone_gt_poses(data_root, getattr(cfg, "t_subsample", 1))
        gt_dir = osp.join(data_root, "test_images")
        # * cfg
        tto_steps = getattr(cfg, "tto_steps", 30)
        decay_start = getattr(cfg, "tto_decay_start", 15)
        lr_p = getattr(cfg, "tto_lr_p", 0.003)
        lr_q = getattr(cfg, "tto_lr_q", 0.003)
        lr_final = getattr(cfg, "tto_lr_final", 0.0001)
        sgd_flag = False
        tto_initialize_from_previous_step_factor = 10
        tto_initialize_from_previous_lr_factor = 0.1
        tto_fg_mask_th = 0.1

    elif dataset_mode == "nvidia":
        # ! always use the first training view
        gt_training_cam_T_wi = cams.T_wc_list().detach().cpu()
        gt_training_fov = cams.fov

        (
            gt_testing_cam_T_wi_list,
            gt_testing_tids_list,
            gt_testing_fns_list,
            gt_testing_fov_list,
            gt_testing_cxcy_ratio_list,
        ) = get_nvidia_dummy_test(gt_training_cam_T_wi, gt_training_fov)
        gt_dir = osp.join(
            # "./data/robust_dynrf/results/Nvidia/gt/", osp.basename(data_root)
            "./eval_utils/nvidia_rodynrf_gt",
            osp.basename(data_root),
        )
        # * cfg
        gt_testing_fov_list[0] = gt_testing_fov_list[0][0]
        tto_steps = getattr(cfg, "tto_steps", 100)
        decay_start = getattr(cfg, "tto_decay_start", 30)
        lr_p = getattr(cfg, "tto_lr_p", 0.0003)
        lr_q = getattr(cfg, "tto_lr_q", 0.0003)
        lr_final = getattr(cfg, "tto_lr_final", 0.000001)
        sgd_flag = False
        # ！ use original
        tto_initialize_from_previous_step_factor = 1
        tto_initialize_from_previous_lr_factor = 1.0
        tto_fg_mask_th = 0.1
    elif dataset_mode == "wild":
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
            val = stream.poses[timestamp].sum()
            if np.isnan(val) or np.isinf(val):
                print(f'Nan or Inf found in gt poses, skipping {i}th pose!')
                continue
            traj_est.append(video_traj[i])
            traj_ref.append(stream.poses[timestamp])
            timestamps.append(video_timestamps[i])

        from evo.core.trajectory import PoseTrajectory3D

        traj_est =PoseTrajectory3D(poses_se3=traj_est,timestamps=timestamps)
        traj_ref =PoseTrajectory3D(poses_se3=traj_ref,timestamps=timestamps)

        from evo.core import sync

        traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
        r_a, t_a, s = traj_est.align(traj_ref, correct_scale=True)

        from evo.core import metrics
        data = (traj_ref, traj_est)
        ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
        ape_metric.process_data(data)
        ape_statistics = ape_metric.get_all_statistics()
        print(ape_statistics['rmse'])

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

        traj_est = PoseTrajectory3D(poses_se3=traj_est,timestamps=range(len(stream)))
        # cams.scale_nw = 0.58046954870224
        s = cams.scale_nw.detach().cpu()
        traj_est.scale(s)
        traj_est = np.stack(traj_est.poses_se3, axis=0)
        
        gt_testing_cam_T_wi_list = torch.from_numpy(traj_est).float()
        
        gt_training_cam_T_wi = torch.from_numpy(np.stack(traj_ref.poses_se3, axis=0))

        gt_testing_tids_list = np.arange(len(gt_testing_cam_T_wi_list))
        gt_training_fov = cams.fov
        gt_testing_fov_list = [gt_training_fov]
        gt_testing_cxcy_ratio_list =[cams.cxcy_ratio]

        # gt_testing_fns_list = [f"t{t:03d}" for t in range(len(gt_training_cam_T_wi))]
        gt_testing_fns_list = [f"t{t:03d}" for t in range(len(stream))]
        gt_dir = osp.join(
            data_root,
            "images",
        )

        viewpoints_data_path = cfg["data"]["output"] + "/" + cfg["scene"] + "/viewpoints_all.ckpt"
        viepoints_data = torch.load(viewpoints_data_path)
        T_starts = viepoints_data["T_starts"]
        T_ends = viepoints_data["T_ends"]
        exposures = viepoints_data["exposures"]
        T_starts_list = []
        T_ends_list = []
        exposures_a_list = []
        exposures_b_list = []
        for timestamp in range(len(stream)):
            timestamp = int(timestamp)
            T_start = T_starts[timestamp]
            T_end = T_ends[timestamp]
            T_starts_list.append(T_start.inverse())
            T_ends_list.append(T_end.inverse())
            exposures_a_list.append(exposures[timestamp][0])
            exposures_b_list.append(exposures[timestamp][1])

        T_starts_list = torch.stack(T_starts_list, dim=0)
        T_ends_list = torch.stack(T_ends_list, dim=0)

        ### normalize
        T_starts_list_np = T_starts_list.detach().cpu().numpy()
        T_ends_list_np = T_ends_list.detach().cpu().numpy()

        traj_start = PoseTrajectory3D(poses_se3=T_starts_list_np,timestamps=timestamps)
        traj_start.scale(s)
        traj_end = PoseTrajectory3D(poses_se3=T_ends_list_np,timestamps=timestamps)
        traj_end.scale(s)

        T_starts_list = torch.from_numpy(np.stack(traj_start.poses_se3, axis=0)).to(T_starts_list)
        T_ends_list = torch.from_numpy(np.stack(traj_end.poses_se3, axis=0)).to(T_ends_list)
        ###

        exposures_a_list = torch.stack(exposures_a_list, dim=0)
        exposures_b_list = torch.stack(exposures_b_list, dim=0)

        param_cam_q_start, param_cam_t_end = cams.__get_init_qt__(T_starts_list)
        param_cam_q_end, param_cam_t_start = cams.__get_init_qt__(T_ends_list)

        cams.q_wc_start = torch.nn.Parameter(param_cam_q_start, requires_grad=False)
        cams.t_wc_start = torch.nn.Parameter(param_cam_t_start, requires_grad=False)
        cams.q_wc_end = torch.nn.Parameter(param_cam_q_end, requires_grad=False)
        cams.t_wc_end = torch.nn.Parameter(param_cam_t_end, requires_grad=False)
        cams.exposures_a = torch.nn.Parameter(torch.tensor(exposures_a_list), requires_grad=False)
        cams.exposures_b = torch.nn.Parameter(torch.tensor(exposures_b_list), requires_grad=False)
        
        cams.T = len(cams.exposures_a)
    else:
        raise ValueError(
            f"Unknown dataset mode: {dataset_mode}, shouldn't call test funcs"
        )
    # id the image size
    sample_fn = [
        f for f in os.listdir(gt_dir) if f.endswith(".png") or f.endswith(".jpg")
    ][0]
    sample = imageio.imread(osp.join(gt_dir, sample_fn))
    H, W = sample.shape[:2]

    ######################################################################
    ######################################################################

    eval_prefix = "tto_" if tto_flag else ""

    if not skip_test_gen:
        for test_i in range(len(gt_testing_cam_T_wi_list)):
            testing_fov = gt_testing_fov_list[test_i]
            testing_focal = 1.0 / np.tan(np.deg2rad(testing_fov) / 2.0)

            if tto_flag:
                frames = render_test_tto(
                    gt_rgb_dir=gt_dir,
                    tto_steps=tto_steps,
                    decay_start=decay_start,
                    lr_p=lr_p,
                    lr_q=lr_q,
                    lr_final=lr_final,
                    use_sgd=sgd_flag,
                    #
                    H=H,
                    W=W,
                    cams=cams,
                    s_model=s_model,
                    d_model=d_model,
                    train_camera_T_wi=gt_training_cam_T_wi,
                    test_camera_T_wi=gt_testing_cam_T_wi_list[test_i],
                    test_camera_tid=gt_testing_tids_list[test_i],
                    save_dir=osp.join(saved_dir, f"tto_test"),
                    save_pose_fn=osp.join(saved_dir, f"tto_test_pose_{test_i}"),
                    fn_list=gt_testing_fns_list[test_i],
                    focal=testing_focal,
                    cxcy_ratio=gt_testing_cxcy_ratio_list[test_i],
                    #
                    initialize_from_previous_camera=True,
                    initialize_from_previous_step_factor=tto_initialize_from_previous_step_factor,
                    initialize_from_previous_lr_factor=tto_initialize_from_previous_lr_factor,
                    fg_mask_th=tto_fg_mask_th,
                )
                imageio.mimsave(
                    osp.join(saved_dir, f"tto_test_cam{test_i}.mp4"), frames
                )
            elif datamode == "wild":
                # break
                frames = render_test_wild(
                    H=H,
                    W=W,
                    cams=cams,
                    s_model=s_model,
                    d_model=d_model,
                    train_camera_T_wi=gt_training_cam_T_wi,
                    test_camera_T_wi=gt_testing_cam_T_wi_list,
                    test_camera_tid=gt_testing_tids_list,
                    save_dir=osp.join(saved_dir, "test"),
                    fn_list=gt_testing_fns_list,
                    focal=testing_focal,
                    cxcy_ratio=gt_testing_cxcy_ratio_list[0],
                    kf_idx=timestamps,
                    stream=stream,
                )
                imageio.mimsave(osp.join(saved_dir, f"test_cam{test_i}.mp4"), frames)
                break

            else:
                frames = render_test(
                    H=H,
                    W=W,
                    cams=cams,
                    s_model=s_model,
                    d_model=d_model,
                    train_camera_T_wi=gt_training_cam_T_wi,
                    test_camera_T_wi=gt_testing_cam_T_wi_list,
                    test_camera_tid=gt_testing_tids_list,
                    save_dir=osp.join(saved_dir, "test"),
                    fn_list=gt_testing_fns_list,
                    focal=testing_focal,
                    cxcy_ratio=gt_testing_cxcy_ratio_list[0],
                )
                imageio.mimsave(osp.join(saved_dir, f"test_cam{test_i}.mp4"), frames)
                # if datamode == "wild":
                #     break

    # * Test
    if dataset_mode == "iphone":
        eval_dycheck(
            save_dir=saved_dir,
            gt_rgb_dir=gt_dir,
            gt_mask_dir=osp.join(data_root, "test_covisible"),
            pred_dir=osp.join(saved_dir, f"{eval_prefix}test"),
            save_prefix=eval_prefix,
            strict_eval_all_gt_flag=True,  # ! only support full len now!!
            eval_non_masked=eval_also_dyncheck_non_masked,
        )

    elif dataset_mode == "nvidia":
        if data_root.endswith("/"):
            data_root = data_root[:-1]
        eval_nvidia_dir(
            gt_dir=gt_dir,
            pred_dir=osp.join(saved_dir, f"{eval_prefix}test"),
            report_dir=osp.join(saved_dir, f"{eval_prefix}test_report"),
        )

    elif dataset_mode == "wild":
        eval_wild(
            save_dir=saved_dir,
            gt_rgb_dir=gt_dir,
            gt_mask_dir=osp.join(data_root, "motion_mask"),
            pred_dir=osp.join(saved_dir, f"{eval_prefix}test"),
            save_prefix=eval_prefix,
            strict_eval_all_gt_flag=True,  # ! only support full len now!!
            eval_non_masked=True,
            stream=stream,
            cfg=cfg,
            timestamps=timestamps,
        )
    logging.info(f"Finished, saved to {saved_dir}")
    return


@torch.no_grad()
def test_pck(saved_dir, gt_npz_fn, device, save_fn=None):
    # laod gt
    src, dst_gt, src_t, dst_t, img_wh, ratio = load_gt_pck_data(
        np.load(gt_npz_fn, allow_pickle=True)["arr_0"]
    )

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

    H, W = int(cams.default_H), int(cams.default_W)
    assert img_wh[0] == W and img_wh[1] == H

    dst_pred = []
    for _st, _dt, _src in tqdm(zip(src_t, dst_t, src)):
        _st, _dt = int(_st), int(_dt)

        # ! use RGB to render xyz because should also work with native renderor
        # render world coordinate map
        d_gs5_src = d_model(_st)
        d_gs5_dst = d_model(_dt)
        s_gs5 = s_model()

        mu = torch.cat([s_gs5[0], d_gs5_src[0]], 0)
        fr = torch.cat([s_gs5[1], d_gs5_src[1]], 0)
        s = torch.cat([s_gs5[2], d_gs5_src[2]], 0)
        o = torch.cat([s_gs5[3], d_gs5_src[3]], 0)
        sph = torch.cat([s_gs5[4], d_gs5_src[4]], 0)

        T_cw = cams.T_cw(_st)
        R_cw, t_cw = T_cw[:3, :3], T_cw[:3, 3]
        mu_cam = torch.einsum("ij,nj->ni", R_cw, mu) + t_cw[None]
        fr_cam = torch.einsum("ij,njk->nik", R_cw, fr)

        xyz_dst = torch.cat([s_gs5[0], d_gs5_dst[0]], 0)
        mu_dst_cam = cams.trans_pts_to_cam(_dt, xyz_dst)

        render_dict = render_cam_pcl(
            mu_cam,
            fr_cam,
            s,
            o,
            sph,
            cams.default_H,
            cams.default_W,
            CAM_K=cams.K(),
            bg_color=[0.0, 0.0, 0.0],
            colors_precomp=mu_dst_cam,
        )
        dst_xyz_map = render_dict["rgb"].permute(1, 2, 0)

        rounded_src = np.round(_src).astype(int)
        rounded_src[:, 0] = np.clip(rounded_src[:, 0], 0, W - 1)
        rounded_src[:, 1] = np.clip(rounded_src[:, 1], 0, H - 1)
        index = rounded_src[:, 1] * W + rounded_src[:, 0]

        dst_xyz = dst_xyz_map.reshape(-1, 3)[index]

        dst_uv = cams.project(dst_xyz)
        dst_x = (dst_uv[:, :1] + 1.0) / 2.0 * 360.0
        dst_y = (dst_uv[:, 1:] + 480.0 / 360.0) / 2.0 * 360.0
        _dst_pred = torch.cat([dst_x, dst_y], dim=1).cpu().numpy()
        dst_pred.append(_dst_pred)

    pck005, _ = eval_pck(dst_gt, dst_pred, img_wh, ratio)
    print(f"PCK@0.05: {pck005}")
    if save_fn is not None:
        with open(save_fn, "w") as fp:
            fp.write(f"PCK@0.05: {pck005:.10f}\n")
    return pck005


def test_sintel_cam(cam_pth_fn, ws, save_path="sintel_pose_metrics.txt"):
    cams = MonocularCameras.load_from_ckpt(torch.load(cam_pth_fn))
    pose_est = cams.T_wc_list().detach().cpu().numpy()
    # gt_dir = osp.join("./data/robust_dynrf/results/Sintel", sq)
    gt_dir = osp.join(ws, "gt_cameras")
    ate, rpe_trans, rpe_rot = eval_sintel_campose(pose_est[:, :3], gt_dir=gt_dir)
    logging.info(
        f"Sintel ATE: {ate}, RPE Translation: {rpe_trans}, RPE Rotation: {rpe_rot}"
    )
    # save to txt
    with open(save_path, "w") as fp:
        fp.write(f"ATE: {ate:.10f}\n")
        fp.write(f"RPE-trans: {rpe_trans:.10f}\n")
        fp.write(f"RPE-rot: {rpe_rot:.10f}\n")
    return ate, rpe_trans, rpe_rot


def test_tum_cam(cam_pth_fn, ws, save_path="tum_pose_metrics.txt"):

    cams = MonocularCameras.load_from_ckpt(torch.load(cam_pth_fn))
    pose_est = cams.T_wc_list().detach().cpu().numpy()
    tt = np.arange(len(pose_est)).astype(float)
    tum_poses = [c2w_to_tumpose(p) for p in pose_est]
    tum_poses = np.stack(tum_poses, 0)
    pred_traj = [tum_poses, tt]

    gt_traj = load_tum_traj(
        gt_traj_file=osp.join(ws, "groundtruth_90.txt"), traj_format="tum"
    )

    ate, rpe_trans, rpe_rot = eval_tum_campose(pred_traj, gt_traj)
    # plot_trajectory(
    #     pred_traj, gt_traj, title=seq, filename=f'{save_dir}/{seq}.png'
    # )
    logging.info(
        f"TUM ATE: {ate}, RPE Translation: {rpe_trans}, RPE Rotation: {rpe_rot}"
    )
    # save to txt
    with open(save_path, "w") as fp:
        fp.write(f"ATE: {ate:.10f}\n")
        fp.write(f"RPE-trans: {rpe_trans:.10f}\n")
        fp.write(f"RPE-rot: {rpe_rot:.10f}\n")
    return ate, rpe_trans, rpe_rot


def test_fps(saved_dir, rounds=1, device=torch.device("cuda:0")):
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

    d_model.set_inference_mode()

    sample_t = [0, cams.T // 2, cams.T - 1]

    s_gs5 = s_model()
    H, W = cams.default_H, cams.default_W
    K = cams.K(H, W)

    viz = []
    for t in sample_t:
        d_gs5 = d_model(t)
        rd = render([s_gs5, d_gs5], H, W, K, T_cw=cams.T_cw(t))
        rd_sample = rd["rgb"].permute(1, 2, 0).cpu().detach().numpy()
        viz.append(rd_sample)
    viz = np.concatenate(viz, 1)
    imageio.imsave(osp.join(saved_dir, "fps_eval_samples.jpg"), viz)

    cnt = cams.T * rounds
    with torch.no_grad():
        start_t = time.time()
        for t in tqdm(range(cnt)):
            t = t % d_model.T
            d_gs5 = d_model(t)
            rd = render([s_gs5, d_gs5], H, W, K, T_cw=cams.T_cw(t))
        end_t = time.time()
    duration = end_t - start_t
    fps = cnt / duration
    logging.info(f"FPS: {fps} tested in rounds {rounds}, rendered {cnt} frames")
    with open(osp.join(saved_dir, "fps_eval.txt"), "w") as fp:
        fp.write(f"FPS: {fps : .10f}\n")
    return

def test_wild(
    cfg,
    saved_dir,
    data_root,
    device,
    tto_flag,
    eval_also_dyncheck_non_masked=False,
    skip_test_gen=False,
):
    # ! this func can be called at the end of running, or run seperately after trained

    # get cfg
    if data_root.endswith("/"):
        data_root = data_root[:-1]
    if isinstance(cfg, str):
        cfg = OmegaConf.load(cfg)
        OmegaConf.set_readonly(cfg, True)

    dataset_mode = getattr(cfg, "mode", "iphone")
    # max_sph_order = getattr(cfg, "max_sph_order", 1)
    logging.info(f"Dataset mode: {dataset_mode}")

    ######################################################################
    ######################################################################

    cams = MonocularCameras.load_from_ckpt(
        torch.load(osp.join(saved_dir, "nodes_cam.pth"))
    ).to(device)
    s_model = StaticGaussian.load_from_ckpt(
        torch.load(
            osp.join(saved_dir, f"nodes_s_model_{GS_BACKEND.lower()}.pth")
        ),
        device=device,
    )
    d_model = DynSCFGaussian.load_from_ckpt(
        torch.load(
            osp.join(saved_dir, f"nodes_d_model_{GS_BACKEND.lower()}.pth")
        ),
        device=device,
    )

    cams.to(device)
    cams.eval()
    d_model.to(device)
    d_model.eval()
    s_model.to(device)
    s_model.eval()

    ######################################################################
    ######################################################################

    eval_prefix = "tto_" if tto_flag else ""

    if datamode == "wild":
        # break
        frames = render_test_wild_blur(
            cams=cams,
            s_model=s_model,
            d_model=d_model,
            save_dir=osp.join(saved_dir, "test"),
        )
        imageio.mimsave(osp.join(saved_dir, f"test_cam.mp4"), frames)

        stream = get_dataset(cfg)
        eval_wild(
            save_dir=saved_dir,
            gt_mask_dir=osp.join(data_root, "motion_mask"),
            pred_dir=osp.join(saved_dir, f"{eval_prefix}test"),
            save_prefix=eval_prefix,
            strict_eval_all_gt_flag=True,  # ! only support full len now!!
            eval_non_masked=True,
            stream=stream,
            cfg=cfg,
        )
    logging.info(f"Finished, saved to {saved_dir}")
    return
@torch.no_grad()
def render_test_wild_blur(
    cams: MonocularCameras,
    s_model: StaticGaussian,
    d_model: DynSCFGaussian,
    save_dir=None,
):

    device = s_model.device

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if focal is None:
        focal = cams.rel_focal
    if cxcy_ratio is None:
        cxcy_ratio = cams.cxcy_ratio
    H, W = cams.default_H, cams.default_W
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

    test_ret = []
    
    t = 0
    is_blur = False
    kf_idx = np.arange(cams.T)
    test_camera_tid = kf_idx
    bg_color = [0.0, 0.0,0.0]
    fn_list = [f"t{t:03d}" for t in range(len(cams.T))]
    for i in tqdm(range(cams.T)):
        opt_cam_rot_start=cams.opt_rot_start[i]
        opt_cam_trans_start=cams.opt_trans_start[i]
        opt_cam_rot_end=cams.opt_rot_end[i]
        opt_cam_trans_end=cams.opt_trans_end[i]
    
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

            d_params =[mu, rot, params_0[2], params_0[3], params_0[4]]

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

            d_params =[mu, rot, params_0[2], params_0[3], params_0[4]]

            gs5 = [s_model(), d_params]


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
        )
        
        render_dict = render(
            gs5,
            H,
            W,
            K,
            T_cw=cams.T_cw[i].to(device),
        )
        image_stack.append(render_dict["rgb"])
        
        image_stack = torch.stack(image_stack, 0)

        render_dict["rgb"] = torch.exp(cams.exposures_a[i]) * image_stack.mean(dim=0) + cams.exposures_b[i]
        

        rgb = render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
        rgb = np.clip(rgb, 0, 1)  # ! important
        rgb = (rgb*255).astype(np.uint8)
        test_ret.append(rgb)
        if save_dir:
            imageio.imwrite(osp.join(save_dir, f"{fn_list[i]}.png"), rgb)
    return test_ret


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ws", type=str, help="Source folder")
    parser.add_argument("--cfg", type=str, help="profile yaml file path")
    parser.add_argument("--logdir", type=str, help="log dir")
    args, unknown = parser.parse_known_args()

    # args = parser.parse_args()

    # cfg = OmegaConf.load(args.cfg)
    def load_config_with_inheritance(config_path) -> OmegaConf:

        # if base_dir is None:
        #     base_dir = os.path.dirname(os.path.abspath(config_path))
        
        # 加载当前配置文件
        cfg = OmegaConf.load(config_path)
        
        # 检查是否有 inherit_from 属性
        if "inherit_from" in cfg:
            inherit_path = cfg.inherit_from
            
            
            # 递归加载父配置
            parent_cfg = load_config_with_inheritance(inherit_path)
            
            # 删除 inherit_from 属性，避免合并后仍然存在
            del cfg.inherit_from
            
            # 合并配置：子配置会覆盖父配置的同名项
            cfg = OmegaConf.merge(parent_cfg, cfg)
        
        return cfg

    # cfg = OmegaConf.load(args.cfg)
    # cli_cfg = OmegaConf.from_dotlist([arg.lstrip("--") for arg in unknown])
    # cfg = OmegaConf.merge(cfg, cli_cfg)

    cfg = load_config_with_inheritance(args.cfg)
    cli_cfg = OmegaConf.from_dotlist([arg.lstrip("--") for arg in unknown])
    cfg = OmegaConf.merge(cfg, cli_cfg)

    logdir = args.logdir

    datamode = getattr(cfg, "mode", "iphone")
    if datamode == "sintel":
        test_func = test_sintel_cam
    elif datamode == "tum":
        test_func = test_tum_cam
    else:
        test_func = None
    if test_func is not None:
        test_func(
            cam_pth_fn=osp.join(logdir, "photometric_cam.pth"),
            ws=args.ws,
            save_path=osp.join(logdir, "final_cam_eval.txt"),
        )

    # test_main(
    #     cfg,
    #     saved_dir=logdir,
    #     data_root=args.ws,
    #     device=torch.device("cuda"),
    #     tto_flag=True,
    #     eval_also_dyncheck_non_masked=False,
    #     skip_test_gen=False,
    # )

    test_wild(
        cfg,
        saved_dir=logdir,
        data_root=args.ws,
        device=torch.device("cuda"),
        tto_flag=True,
        eval_also_dyncheck_non_masked=False,
        skip_test_gen=False,
    )
