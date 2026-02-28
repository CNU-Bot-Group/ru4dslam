import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import math
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from sh_utils import eval_sh
import time
import torch
import numpy as np
import torch.nn.functional as F

from pytorch3d.transforms import matrix_to_quaternion

def render_cam_pcl(
    xyz,
    frame,
    scale,
    opacity,
    color_feat,
    H,
    W,
    # Multiple way to specify camera
    CAM_K=None,
    fx=None,
    fy=None,
    cx=None,
    cy=None,
    verbose=False,
    # active_sph_order=0,
    bg_color=[1.0, 1.0, 1.0],
    add_buffer=None,
    colors_precomp=None,
    T_cw = torch.eye(4),
    theta=torch.zeros(3),
    rho=torch.zeros(3),
):


    device = xyz.device
    screenspace_points = (
        torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device=xyz.device)
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    
    fx, fy, cx, cy = CAM_K[0, 0], CAM_K[1, 1], CAM_K[0, 2], CAM_K[1, 2]


    FoVx = focal2fov(fx, W)
    FoVy = focal2fov(fy, H)
    tanfovx = math.tan(FoVx * 0.5)
    tanfovy = math.tan(FoVy * 0.5)

    viewmatrix = getWorld2View2_torch(T_cw[:3, :3], T_cw[:3, 3]).transpose(0, 1).to(device)
    projection_matrix = (
        getProjectionMatrix2(znear=0.0001, zfar=1.0, cx=cx, cy=cy, fx=fx, fy=fy, W=W, H=H)
        .transpose(0, 1)
        .to(device)
    )
    full_proj_transform = (
        viewmatrix.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera_center = viewmatrix.inverse()[3, :3]

    means3D = xyz
    means2D = screenspace_points

    if scale.shape[-1] == 1:
        scales = scale.repeat(1, 3)
    else:
        scales = scale

    # rotations = matrix_to_quaternion(frame)
    if frame.shape[-1] == 4:
        rotations = frame
    else:
        rotations = matrix_to_quaternion(frame)
    cov3D_precomp = None

    colors_precomp = None
    shs = color_feat

    sh_degree = int(((shs.shape[-1])/3)**(1/2) - 1)
    shs = shs.view(-1, (sh_degree+1)**2, 3)

    raster_settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.tensor(bg_color, dtype=torch.float32, device=device),
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=full_proj_transform,
        projmatrix_raw=projection_matrix,
        sh_degree=sh_degree,  # ! use pre-compute color!
        campos=camera_center,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    start_time = time.time()
    rendered_image, radii, depth, alpha, n_touched = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        theta=theta,
        rho=rho,
    )

    if verbose:
        print(
            f"render time: {(time.time() - start_time)*1000:.3f}ms",
        )
    ret = {
        "rgb": rendered_image,
        "dep": depth,
        "alpha": alpha,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
    
    return ret


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def getWorld2View2_torch(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
    translate = translate.to(R.device)
    Rt = torch.zeros((4, 4), device=R.device)
    # Rt[:3, :3] = R.transpose()
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.linalg.inv(C2W)
    return Rt


def getProjectionMatrix2(znear, zfar, cx, cy, fx, fy, W, H):
    left = ((2 * cx - W) / W - 1.0) * W / 2.0
    right = ((2 * cx - W) / W + 1.0) * W / 2.0
    top = ((2 * cy - H) / H + 1.0) * H / 2.0
    bottom = ((2 * cy - H) / H - 1.0) * H / 2.0
    left = znear / fx * left
    right = znear / fx * right
    top = znear / fy * top
    bottom = znear / fy * bottom
    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P