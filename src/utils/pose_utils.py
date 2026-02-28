# Copyright 2024 The MonoGS Authors.

# Licensed under the License issued by the MonoGS Authors
# available here: https://github.com/muskie82/MonoGS/blob/main/LICENSE.md

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from pytorch3d.transforms import (
    matrix_to_quaternion,
    quaternion_to_axis_angle,
    matrix_to_rotation_6d
)
def rt2mato(R, T): # TODO: remove?
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = T
    return mat


def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm


def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    # if angle < 1e-5:
    #     return I + W + 0.5 * W2
    # else:
    #     return (
    #         I
    #         + (torch.sin(angle) / angle) * W
    #         + ((1 - torch.cos(angle)) / (angle**2)) * W2
    #     )
    R_small = I + W + 0.5 * W2
    R_normal = I + (torch.sin(angle) / angle) * W + ((1 - torch.cos(angle)) / (angle**2)) * W2
    R = torch.where(angle < 1e-5, R_small, R_normal)
    return R

def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    # if angle < 1e-5:
    #     V = I + 0.5 * W + (1.0 / 6.0) * W2
    # else:
    #     V = (
    #         I
    #         + W * ((1.0 - torch.cos(angle)) / (angle**2))
    #         + W2 * ((angle - torch.sin(angle)) / (angle**3))
    #     )
    V_small = I + 0.5 * W + (1.0 / 6.0) * W2
    V_normal = I + W * ((1.0 - torch.cos(angle)) / (angle**2)) + W2 * ((angle - torch.sin(angle)) / (angle**3))
    V = torch.where(angle < 1e-5, V_small, V_normal)
    return V


def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.cat([
        torch.cat([R, t.unsqueeze(1)], dim=1),
        torch.tensor([[0, 0, 0, 1]], device=device, dtype=dtype)
    ], dim=0)
    return T

# def rt_to_se3(rt_matrix):
#     """
#     使用PyTorch3D将RT矩阵转换为se3
#     """
#     # 提取旋转矩阵和平移向量
#     R = rt_matrix[:3, :3]
#     t = rt_matrix[:3, 3]
    
#     # 方法1：通过四元数转换
#     quat = matrix_to_quaternion(R)  # (4,)
#     axis_angle = quaternion_to_axis_angle(quat)  # (3,)
    
#     # # 方法2：直接转换为6D表示
#     # rot_6d = matrix_to_rotation_6d(R)  # (6,)
    
#     # 组合成se3向量
#     se3 = torch.cat([t, axis_angle])
    
#     return se3

def rotation_to_so3(R):
    """
    将旋转矩阵转换为旋转向量
    :param R: 3x3 旋转矩阵
    :return: 旋转向量 (3,)
    """
    rot = Rotation.from_matrix(R.cpu().numpy())
    theta = torch.tensor(rot.as_rotvec(), device=R.device, dtype=R.dtype)
    return theta

def rt_to_se3(rt_matrix):
    """
    将 RT 矩阵转换为李代数 se(3)
    :param rt_matrix: 4x4 变换矩阵
    :return: 李代数向量 tau = [rho, theta]
    """
    R = rt_matrix[:3, :3]
    t = rt_matrix[:3, 3]

    # 旋转部分：旋转矩阵 -> 旋转向量
    theta = rotation_to_so3(R)

    # 平移部分：计算 rho = V(theta)^{-1} @ t
    V_mat = V(theta)
    rho = torch.linalg.inv(V_mat) @ t

    # 拼接为李代数向量
    tau = torch.cat([rho, theta])
    return tau
# def rt_to_se3(rt_matrix):
#     """
#     使用 torch.linalg.logm 将 RT 矩阵转换为李代数
#     """
#     se3_matrix = torch.linalg.logm(rt_matrix)
#     tau = torch.tensor([
#         se3_matrix[0, 3],
#         se3_matrix[1, 3],
#         se3_matrix[2, 3],
#         se3_matrix[2, 1],
#         se3_matrix[0, 2],
#         se3_matrix[1, 0]
#     ], device=rt_matrix.device, dtype=rt_matrix.dtype)
#     return tau

def update_pose(camera, converged_threshold=1e-4):
    tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], axis=0)

    T_w2c = torch.eye(4, device=tau.device)
    T_w2c[0:3, 0:3] = camera.R
    T_w2c[0:3, 3] = camera.T

    new_w2c = SE3_exp(tau) @ T_w2c

    new_R = new_w2c[0:3, 0:3]
    new_T = new_w2c[0:3, 3]

    converged = tau.norm() < converged_threshold
    camera.update_RT(new_R, new_T)

    camera.cam_rot_delta.data.fill_(0)
    camera.cam_trans_delta.data.fill_(0)
    return converged

def quaternion_slerp_batch(q1, q2, t_batch):
    """
    批量四元数球面线性插值
    
    参数:
        q1: 起始四元数 [4]
        q2: 终止四元数 [4]
        t_batch: 插值参数 [N]
        
    返回:
        q_batch: 插值后的四元数 [N, 4]
    """
    # 确保四元数是单位四元数
    q1 = q1 / torch.norm(q1)
    q2 = q2 / torch.norm(q2)
    
    # 计算四元数之间的点积
    dot = torch.sum(q1 * q2)
    
    # 如果点积为负，取q2的负数以确保最短路径插值
    if dot < 0:
        q2 = -q2
        dot = -dot
    
    # 如果四元数非常接近，使用线性插值以避免除以零
    if dot > 0.9995:
        result = q1.unsqueeze(0) + t_batch.unsqueeze(-1) * (q2 - q1).unsqueeze(0)
        return result / torch.norm(result, dim=-1, keepdim=True)
    
    # 计算插值角度
    theta_0 = torch.acos(torch.clamp(dot, -1.0, 1.0))
    sin_theta_0 = torch.sin(theta_0)
    
    # 计算插值系数
    theta_batch = theta_0 * t_batch
    sin_theta_batch = torch.sin(theta_batch)
    
    # 计算插值权重
    s0 = torch.cos(theta_batch) - dot * sin_theta_batch / sin_theta_0
    s1 = sin_theta_batch / sin_theta_0
    
    # 计算插值结果
    q_batch = s0.unsqueeze(-1) * q1.unsqueeze(0) + s1.unsqueeze(-1) * q2.unsqueeze(0)
    
    # 归一化四元数
    q_batch = q_batch / torch.norm(q_batch, dim=-1, keepdim=True)
    
    return q_batch