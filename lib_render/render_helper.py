import os, sys, os.path as osp
import torch

sys.path.append(osp.dirname(osp.abspath(__file__)))

try:
    GS_BACKEND = os.environ["GS_BACKEND"]
except:
    print(f"No GS_BACKEND env var specified, for now use native_add3 backend")
    GS_BACKEND = "native_add3"

GS_BACKEND = GS_BACKEND.lower()
print(f"GS_BACKEND: {GS_BACKEND.lower()}")

if GS_BACKEND == "native":
    from gauspl_renderer_native import render_cam_pcl
elif GS_BACKEND == "gof":
    from gauspl_renderer_gof import render_cam_pcl
elif GS_BACKEND == "native_add3":
    from gauspl_renderer_native_add3 import render_cam_pcl
elif GS_BACKEND == "w_pose":
    from gauspl_renderer_w_pose import render_cam_pcl
else:
    raise ValueError(f"Unknown GS_BACKEND: {GS_BACKEND.lower()}")
from sh_utils import RGB2SH, SH2RGB

from thirdparty.gaussian_splatting.utils.general_utils import (
    quaternion_multiply,
    build_rotation,
    multiply_quaternions,
    quaternion_to_matrix_optimized,
)

from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion

from src.utils.Spline import SplineN_linear_qt

import torch.nn.functional as F

def render_o(
    gs_param,
    H,
    W,
    K,
    T_cw,
    bg_color=[1.0, 1.0, 1.0],
    scale_factor=1.0,
    opa_replace=None,
    bg_cache_dict=None,
    colors_precomp=None,
    add_buffer=None,
    theta=torch.zeros(3),
    rho=torch.zeros(3),
):
    # * Core render interface
    # prepare gs5 param in world system
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
    if opa_replace is not None:
        assert isinstance(opa_replace, float)
        o = torch.ones_like(o) * opa_replace
    s = s * scale_factor

    # cvt to cam frame
    assert T_cw.ndim == 2 and T_cw.shape[1] == T_cw.shape[0] == 4
    if GS_BACKEND == "w_pose":
        mu_cam, fr_cam = mu, fr
    else:
        R_cw, t_cw = T_cw[:3, :3], T_cw[:3, 3]
        mu_cam = torch.einsum("ij,nj->ni", R_cw, mu) + t_cw[None]
        fr_cam = torch.einsum("ij,njk->nik", R_cw, fr)
    

    # render
    render_dict = render_cam_pcl(
        mu_cam,
        fr_cam,
        s,
        o,
        sph,
        H,
        W,
        CAM_K=K,
        bg_color=bg_color,
        colors_precomp=colors_precomp,
        add_buffer=add_buffer,
        T_cw = T_cw,
        theta=theta,
        rho=rho,
    )
    if bg_cache_dict is not None:
        render_dict = fast_bg_compose_render(bg_cache_dict, render_dict, bg_color)
    return render_dict

def render(
    gs_param,
    H,
    W,
    K,
    T_cw,
    bg_color=[1.0, 1.0, 1.0],
    scale_factor=1.0,
    opa_replace=None,
    bg_cache_dict=None,
    colors_precomp=None,
    add_buffer=None,
    opt_cam_rot_start=None,
    opt_cam_trans_start=None,
    opt_cam_rot_end=None,
    opt_cam_trans_end=None,
):
    # * Core render interface
    # prepare gs5 param in world system
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
    if opa_replace is not None:
        assert isinstance(opa_replace, float)
        o = torch.ones_like(o) * opa_replace
    s = s * scale_factor

    # cvt to cam frame
    assert T_cw.ndim == 2 and T_cw.shape[1] == T_cw.shape[0] == 4
    if GS_BACKEND == "w_pose":
        mu_cam, fr_cam = mu, fr 
    else:
        R_cw, t_cw = T_cw[:3, :3], T_cw[:3, 3]
        mu_cam = torch.einsum("ij,nj->ni", R_cw, mu) + t_cw[None]
        fr_cam = torch.einsum("ij,njk->nik", R_cw, fr)
    

    if opt_cam_rot_start is None:
        # render
        render_dict = render_cam_pcl(
            mu_cam,
            fr_cam,
            s,
            o,
            sph,
            H,
            W,
            CAM_K=K,
            bg_color=bg_color,
            colors_precomp=colors_precomp,
            add_buffer=add_buffer,
            T_cw = T_cw,
        )
    else:
        rgb_stack = []
        mu = mu_cam
        fr = fr_cam
        
        gs_rotation = matrix_to_quaternion(fr)

        rotation_matrix = build_rotation(opt_cam_rot_start[None])[0]
        rel_transform_start = torch.eye(4, device=mu.device, dtype=torch.float32)
        rel_transform_start[:3, :3] = rotation_matrix
        rel_transform_start[:3, 3] = opt_cam_trans_start

        pts = mu
        pts_ones = torch.ones(pts.shape[0], 1, device=mu.device, dtype=torch.float32)
        pts4 = torch.cat((pts, pts_ones), dim=1)
        transformed_pts_start = (rel_transform_start @ pts4.T).T[:, :3]
        quat_start = opt_cam_rot_start

        _rotations_start = quaternion_multiply(gs_rotation, quat_start.unsqueeze(0)).squeeze(0)
        if GS_BACKEND == "native_add3":
            _rotations_start = quaternion_to_matrix(_rotations_start)
        rgb_stack.append(    # render
            render_cam_pcl(
                transformed_pts_start,
                _rotations_start,
                s,
                o,
                sph,
                H,
                W,
                CAM_K=K,
                bg_color=bg_color,
                colors_precomp=colors_precomp,
                add_buffer=add_buffer,
                T_cw = T_cw,
            )["rgb"]
        )

        rel_transform_end = torch.eye(4, device=mu.device, dtype=torch.float32)

        rotation_matrix_end = quaternion_to_matrix(opt_cam_rot_end)
        rel_transform_end[:3, :3] = rotation_matrix_end
        rel_transform_end[:3, 3] = opt_cam_trans_end
        transformed_pts_end = (rel_transform_end @ pts4.T).T[:, :3]

        quat_end = opt_cam_rot_end
        _rotations_end = multiply_quaternions(gs_rotation, quat_end.unsqueeze(0)).squeeze(0)
        if GS_BACKEND == "native_add3":
            _rotations_end = quaternion_to_matrix(_rotations_end)
        rgb_stack.append(    # render
            render_cam_pcl(
                transformed_pts_end,
                _rotations_end,
                s,
                o,
                sph,
                H,
                W,
                CAM_K=K,
                bg_color=bg_color,
                colors_precomp=colors_precomp,
                add_buffer=add_buffer,
                T_cw = T_cw,
            )["rgb"]
        )

        t_s = 0.01
        r_s = 0.005

        T_cur = T_cw

        w2c = torch.eye(4).to(mu.device)
        w2c[:3, :3] = quaternion_to_matrix(opt_cam_rot_start)
        w2c[:3, 3] = opt_cam_trans_start.clone().detach().cpu().to(mu.device)

        T_start = T_cur @ w2c
        T_start[-1, :] = torch.tensor([0., 0., 0., 1.], dtype=T_start.dtype, device=T_start.device)
        
        w2c = torch.eye(4).to(mu.device)
        w2c[:3, :3] = quaternion_to_matrix(opt_cam_rot_end)
        w2c[:3, 3] = opt_cam_trans_end.clone().detach().cpu().to(mu.device)

        T_end = T_cur @ w2c
        T_end[-1, :] = torch.tensor([0., 0., 0., 1.], dtype=T_end.dtype, device=T_end.device)

        c_vec = torch.tensor([0.0, 0.0, 1.0], device=mu.device)

        dt = torch.norm(opt_cam_trans_start)
        
        world_view_start = T_start[:3, :3] @ c_vec
        world_view_end = T_cur[:3, :3] @ c_vec
        world_view_start = world_view_start / torch.norm(world_view_start)
        world_view_end = world_view_end / torch.norm(world_view_end)

        cos_angle = torch.dot(world_view_start, world_view_end)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0) 
        
        dr = torch.acos(cos_angle)

        num_poses = max(min((torch.ceil(dt / t_s)).int(), 4), min((torch.ceil(dr / r_s)).int(), 4))
        num_poses = 3
        if num_poses > 2:
            
            qt_start = torch.cat([quat_start.squeeze(0), opt_cam_trans_start])

            qt_end = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=qt_start.device)

            qts = SplineN_linear_qt(qt_start, qt_end, num_poses, mu.device)
            qts = qts[1:-1]

            qs = qts[:, :4]  # [N, 4]
            ts = qts[:, 4:]  # [N, 3]

            rotation_matrices = quaternion_to_matrix(F.normalize(qs))
            rel_transforms = torch.eye(4, device=mu.device, dtype=torch.float32).repeat(len(qts), 1, 1)
            rel_transforms[:, :3, :3] = rotation_matrices
            rel_transforms[:, :3, 3] = ts

            pts4_expanded = pts4.unsqueeze(0).repeat(len(qts), 1, 1)  # [N, num_points, 4]
            transformed_pts_all = torch.bmm(rel_transforms, pts4_expanded.transpose(1, 2)).transpose(1, 2)[:, :, :3]

            _rotations_all = multiply_quaternions(
                gs_rotation.unsqueeze(0).repeat(len(qts), 1, 1), 
                qs.unsqueeze(-2)
            )
            if GS_BACKEND == "native_add3":
                _rotations_all = quaternion_to_matrix(_rotations_all)
            for i in range(len(qts)):
                rgb_stack.append(    # render
                    render_cam_pcl(
                        transformed_pts_all[i],
                        _rotations_all[i],
                        s,
                        o,
                        sph,
                        H,
                        W,
                        CAM_K=K,
                        bg_color=bg_color,
                        colors_precomp=colors_precomp,
                        add_buffer=add_buffer,
                        T_cw = T_cw,
                    )["rgb"]
                )

        dt = torch.norm(opt_cam_trans_end)
        
        world_view_start = T_cur[:3, :3] @ c_vec
        world_view_end = T_end[:3, :3] @ c_vec
        world_view_start = world_view_start / torch.norm(world_view_start)
        world_view_end = world_view_end / torch.norm(world_view_end)

        cos_angle = torch.dot(world_view_start, world_view_end)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        
        dr = torch.acos(cos_angle)

        num_poses = max(min((torch.ceil(dt / t_s)).int(), 4), min((torch.ceil(dr / r_s)).int(), 4))
        num_poses = 3
        if num_poses > 2:
            
            qt_end = torch.cat([quat_end.squeeze(0), opt_cam_trans_end])

            qt_start = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=qt_end.device)

            qts = SplineN_linear_qt(qt_start, qt_end, num_poses, mu.device)
            qts = qts[1:-1]

            qs = qts[:, :4]  # [N, 4]
            ts = qts[:, 4:]  # [N, 3]

            rotation_matrices = quaternion_to_matrix(F.normalize(qs))
            rel_transforms = torch.eye(4, device=mu.device, dtype=torch.float32).repeat(len(qts), 1, 1)
            rel_transforms[:, :3, :3] = rotation_matrices
            rel_transforms[:, :3, 3] = ts

            pts4_expanded = pts4.unsqueeze(0).repeat(len(qts), 1, 1)  # [N, num_points, 4]
            transformed_pts_all = torch.bmm(rel_transforms, pts4_expanded.transpose(1, 2)).transpose(1, 2)[:, :, :3]

            _rotations_all = multiply_quaternions(
                gs_rotation.unsqueeze(0).repeat(len(qts), 1, 1), 
                qs.unsqueeze(-2)
            )

            if GS_BACKEND == "native_add3":
                _rotations_all = quaternion_to_matrix(_rotations_all)
            for i in range(len(qts)):
                rgb_stack.append(    # render
                    render_cam_pcl(
                        transformed_pts_all[i],
                        _rotations_all[i],
                        s,
                        o,
                        sph,
                        H,
                        W,
                        CAM_K=K,
                        bg_color=bg_color,
                        colors_precomp=colors_precomp,
                        add_buffer=add_buffer,
                        T_cw = T_cw,
                    )["rgb"]
                )
    if bg_cache_dict is not None:
        render_dict = fast_bg_compose_render(bg_cache_dict, render_dict, bg_color)
    return render_dict

def render_blur(
    gs_param,
    H,
    W,
    K,
    T_cw,
    bg_color=[1.0, 1.0, 1.0],
    scale_factor=1.0,
    opa_replace=None,
    bg_cache_dict=None,
    colors_precomp=None,
    add_buffer=None,
    override_rotation=None,
    opt_cam_rot_start=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    opt_cam_trans_start=torch.tensor([0.0, 0.0, 0.0]),
    opt_cam_rot_end=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    opt_cam_trans_end=torch.tensor([0.0, 0.0, 0.0]),
    scale_nw=1.0,
):
    # * Core render interface
    # prepare gs5 param in world system

    # t = time.time()
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
    if opa_replace is not None:
        assert isinstance(opa_replace, float)
        o = torch.ones_like(o) * opa_replace
    s = s * scale_factor
    # print("prepare gs5 param in world system", time.time() - t)
    # cvt to cam frame
    assert T_cw.ndim == 2 and T_cw.shape[1] == T_cw.shape[0] == 4
    
    if GS_BACKEND == "native_add3":
        assert T_cw.ndim == 2 and T_cw.shape[1] == T_cw.shape[0] == 4
        R_cw, t_cw = T_cw[:3, :3], T_cw[:3, 3]
        mu_cam = torch.einsum("ij,nj->ni", R_cw, mu) + t_cw[None]
        fr_cam = torch.einsum("ij,njk->nik", R_cw, fr)
        mu = mu_cam
        fr = fr_cam
    rgb_stack = []

    if override_rotation is None:
        gs_rotation = matrix_to_quaternion(fr)
        # print(f"gs_rotation: {time.time() - t}")
    else:
        gs_rotation = override_rotation
    # rotation_matrix = build_rotation(opt_cam_rot_start[None])[0]
    # # rotation_matrix = quaternion_to_matrix(opt_cam_rot_start)
    # # rotation_matrix = efficient_quaternion_to_matrix(opt_cam_rot_start[None])[0]
    # rel_transform_start = torch.eye(4, device=mu.device, dtype=torch.float32)
    # rel_transform_start[:3, :3] = rotation_matrix
    # rel_transform_start[:3, 3] = opt_cam_trans_start

    pts = mu

    
    pts_ones = torch.ones(pts.shape[0], 1, device=mu.device, dtype=torch.float32)
    pts4 = torch.cat((pts, pts_ones), dim=1)
    opt_cam_rot_start = F.normalize(opt_cam_rot_start, dim=-1)
    opt_cam_rot_end = F.normalize(opt_cam_rot_end, dim=-1)

    quat_start = opt_cam_rot_start
    
    quat_end = opt_cam_rot_end
   
    t_s = 0.01*scale_nw
    r_s = 0.006

    with torch.no_grad():
        dr = 2 * torch.acos(torch.abs(opt_cam_rot_start[0]))

        dt = (opt_cam_trans_start[0]**2 + 
            opt_cam_trans_start[1]**2 + 
            opt_cam_trans_start[2]**2).sqrt()

        num_poses = max(min((torch.ceil(dt / t_s)).int(), 4), min((torch.ceil(dr / r_s)).int(), 4))
    # num_poses = 3
    if num_poses >= 1:
        if num_poses > 1:
            qt_start = torch.cat([quat_start.squeeze(0), opt_cam_trans_start])

            # q = rotation_matrix_to_quaternion(T_cur[:3, :3][None])[0]
            # t = T_cur[:3, 3]
            # qt_end = torch.cat([q, t], dim=-1)
            qt_end = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=qt_start.device)

            qts = SplineN_linear_qt(qt_start, qt_end, num_poses, mu.device)
            qts = qts[0:-1]

            qs = qts[:, :4]  # [N, 4]
            ts = qts[:, 4:]  # [N, 3]
        else:
            qs = quat_start.unsqueeze(0)
            ts = opt_cam_trans_start.unsqueeze(0)
        # print(f"spline: {time.time() - t}")
        # rotation_matrices = build_rotation(F.normalize(qs))  # [N, 3, 3]
        rotation_matrices = quaternion_to_matrix_optimized(F.normalize(qs))
        # print(f"rotation_matrices: {time.time() - t}")
        rel_transforms = torch.eye(4, device=mu.device, dtype=torch.float32).repeat(len(qs), 1, 1)
        rel_transforms[:, :3, :3] = rotation_matrices
        rel_transforms[:, :3, 3] = ts

        pts4_expanded = pts4.unsqueeze(0).repeat(len(qs), 1, 1)  # [N, num_points, 4]
        transformed_pts_all = torch.bmm(rel_transforms, pts4_expanded.transpose(1, 2)).transpose(1, 2)[:, :, :3]

        _rotations_all = multiply_quaternions(
            gs_rotation.unsqueeze(0).repeat(len(qs), 1, 1), 
            qs.unsqueeze(-2)
        )
        if GS_BACKEND == "native_add3":
            _rotations_all = quaternion_to_matrix_optimized(_rotations_all)
            # print(f"transformed_pts_all: {time.time() - t}")
        for i in range(len(qs)):
            rgb_stack.append(    # render
                render_cam_pcl(
                    transformed_pts_all[i],
                    _rotations_all[i],
                    s,
                    o,
                    sph,
                    H,
                    W,
                    CAM_K=K,
                    bg_color=bg_color,
                    colors_precomp=colors_precomp,
                    add_buffer=add_buffer,
                    T_cw = T_cw,
                )["rgb"]
            )
    with torch.no_grad():
        dr = 2 * torch.acos(torch.abs(opt_cam_rot_end[0]))

        dt = (opt_cam_trans_end[0]**2 + 
            opt_cam_trans_end[1]**2 + 
            opt_cam_trans_end[2]**2).sqrt()
        num_poses = max(min((torch.ceil(dt / t_s)).int(), 4), min((torch.ceil(dr / r_s)).int(), 4))
    # num_poses = 3

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
        # print(f"spline: {time.time() - t}")
        rotation_matrices = quaternion_to_matrix_optimized(F.normalize(qs))
        # print(f"rotation_matrices: {time.time() - t}")
        rel_transforms = torch.eye(4, device=mu.device, dtype=torch.float32).repeat(len(qs), 1, 1)
        rel_transforms[:, :3, :3] = rotation_matrices
        rel_transforms[:, :3, 3] = ts

        pts4_expanded = pts4.unsqueeze(0).repeat(len(qs), 1, 1)  # [N, num_points, 4]
        transformed_pts_all = torch.bmm(rel_transforms, pts4_expanded.transpose(1, 2)).transpose(1, 2)[:, :, :3]

        _rotations_all = multiply_quaternions(
            gs_rotation.unsqueeze(0).repeat(len(qs), 1, 1), 
            qs.unsqueeze(-2)
        )

        # print(f"multi: {time.time() - t}")

        if GS_BACKEND == "native_add3":
            _rotations_all = quaternion_to_matrix_optimized(_rotations_all)
            # print(f"transformed_pts_all: {time.time() - t}")
        for i in range(len(qs)):
            rgb_stack.append(    # render
                render_cam_pcl(
                    transformed_pts_all[i],
                    _rotations_all[i],
                    s,
                    o,
                    sph,
                    H,
                    W,
                    CAM_K=K,
                    bg_color=bg_color,
                    colors_precomp=colors_precomp,
                    add_buffer=add_buffer,
                    T_cw = T_cw,
                )["rgb"]
            )
        # print(f"render: {time.time() - t}")
    return rgb_stack

def fast_bg_compose_render(bg_cache_dict, render_dict, bg_color=[1.0, 1.0, 1.0]):
    assert GS_BACKEND == "native", "GOF does not support this now"

    # manually compose the fg
    # ! warning, be careful when use the visibility masks .etc, watch the len
    fg_rgb, bg_rgb = render_dict["rgb"], bg_cache_dict["rgb"]
    fg_alpha, bg_alpha = render_dict["alpha"], bg_cache_dict["alpha"]
    fg_dep, bg_dep = render_dict["dep"], bg_cache_dict["dep"]
    _fg_alp = torch.clamp(fg_alpha, 1e-8, 1.0)
    _bg_alp = torch.clamp(bg_alpha, 1e-8, 1.0)
    fg_dep_corr = fg_dep / _fg_alp
    bg_dep_corr = bg_dep / _bg_alp
    fg_in_front = (fg_dep_corr < bg_dep_corr).float()
    # compose alpha
    alpha_fg_front_compose = fg_alpha + (1.0 - fg_alpha) * bg_alpha
    alpha_fg_behind_compose = bg_alpha + (1.0 - bg_alpha) * fg_alpha
    alpha_composed = alpha_fg_front_compose * fg_in_front + alpha_fg_behind_compose * (
        1.0 - fg_in_front
)
    alpha_composed = torch.clamp(alpha_composed, 0.0, 1.0)
    # compose rgb
    bg_color = torch.as_tensor(bg_color, device=fg_rgb.device, dtype=fg_rgb.dtype)
    rgb_fg_front_compose = (
        fg_rgb * fg_alpha
        + bg_rgb * (1.0 - fg_alpha) * bg_alpha
        + (1.0 - fg_alpha) * (1.0 - bg_alpha) * bg_color[:, None, None]
    )
    rgb_fg_behind_compose = (
        bg_rgb * bg_alpha
        + fg_rgb * (1.0 - bg_alpha) * fg_alpha
        + (1.0 - bg_alpha) * (1.0 - fg_alpha) * bg_color[:, None, None]
    )
    rgb_composed = rgb_fg_front_compose * fg_in_front + rgb_fg_behind_compose * (
        1.0 - fg_in_front
    )
    # compose dep
    dep_fg_front_compose = (
        fg_dep_corr * fg_alpha + bg_dep_corr * (1.0 - fg_alpha) * bg_alpha
    )
    dep_fg_behind_compose = (
        bg_dep_corr * bg_alpha + fg_dep_corr * (1.0 - bg_alpha) * fg_alpha
    )
    dep_composed = dep_fg_front_compose * fg_in_front + dep_fg_behind_compose * (
        1.0 - fg_in_front
    )
    return {
        "rgb": rgb_composed,
        "dep": dep_composed,
        "alpha": alpha_composed,
        "visibility_filter": render_dict["visibility_filter"],
        "viewspace_points": render_dict["viewspace_points"],
        "radii": render_dict["radii"],
        "dyn_rgb": fg_rgb,
        "dyn_dep": fg_dep,
        "dyn_alpha": fg_alpha,
    }
