# Copyright 2024 The Splat-SLAM Authors.
# Licensed under the Apache License, Version 2.0
# available at: https://github.com/google-research/Splat-SLAM/blob/main/LICENSE
#
# Additional modifications include:
# - Added "matplotlib.use('Agg')" in the function `traj_eval_and_plot` to avoid "Could not load the Qt platform" error.
# - Added a “save_traj” function to save the estimated trajectory to tum format.
# - Refine non-keyframe-traj from the mapping component in the function `full_traj_eval`.

import numpy as np
import torch
from lietorch import SE3
from src.utils.Printer import FontColor
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from src.utils.datasets import RGB_NoPose

def align_kf_traj(npz_path,stream,return_full_est_traj=False,printer=None):
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
            printer.print(f'Nan or Inf found in gt poses, skipping {i}th pose!',FontColor.INFO)
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

    if return_full_est_traj:
        from evo.core import lie_algebra as lie
        traj_est_full = PoseTrajectory3D(poses_se3=video_traj,timestamps=video_timestamps)
        traj_est_full.scale(s)
        traj_est_full.transform(lie.se3(r_a, t_a))
        traj_est = traj_est_full

    return r_a, t_a, s, traj_est, traj_ref    

def align_full_traj(traj_est_full,stream,printer):

    timestamps = []
    traj_ref = []
    traj_est = []
    for i in range(len(stream.poses)):
        val = stream.poses[i].sum()
        if np.isnan(val) or np.isinf(val):
            printer.print(f'Nan or Inf found in gt poses, skipping {i}th pose!',FontColor.INFO)
            continue
        traj_est.append(traj_est_full[i])
        traj_ref.append(stream.poses[i])
        timestamps.append(float(i))
    
    from evo.core.trajectory import PoseTrajectory3D

    traj_est =PoseTrajectory3D(poses_se3=traj_est,timestamps=timestamps)
    traj_ref =PoseTrajectory3D(poses_se3=traj_ref,timestamps=timestamps)

    from evo.core import sync

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    r_a, t_a, s = traj_est.align(traj_ref, correct_scale=True)
    return r_a, t_a, s, traj_est, traj_ref    


def traj_eval_and_plot(traj_est, traj_ref, plot_parent_dir, plot_name,printer):
    import os
    from evo.core import metrics
    from evo.tools import plot
    import matplotlib
    matplotlib.use('Agg')  # Avoid "Could not load the Qt platform" error
    import matplotlib.pyplot as plt
    if not os.path.exists(plot_parent_dir):
        os.makedirs(plot_parent_dir)
    printer.print("Calculating APE ...",FontColor.EVAL)
    data = (traj_ref, traj_est)
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data(data)
    ape_statistics = ape_metric.get_all_statistics()

    printer.print("Plotting ...",FontColor.EVAL)

    plot_collection = plot.PlotCollection("kf factor graph")
    # metric values
    fig_1 = plt.figure(figsize=(8, 8))
    plot_mode = plot.PlotMode.xy
    ax = plot.prepare_axis(fig_1, plot_mode)
    plot.traj(ax, plot_mode, traj_ref, '--', 'gray', 'reference')
    plot.traj_colormap(
    ax, traj_est, ape_metric.error, plot_mode, min_map=ape_statistics["min"],
    max_map=ape_statistics["max"], title="APE mapped onto trajectory")
    plot_collection.add_figure("2d", fig_1)
    plot_collection.export(f"{plot_parent_dir}/{plot_name}.png", False)

    return ape_statistics


def kf_traj_eval(npz_path, plot_parent_dir,plot_name, stream, logger,printer):
    r_a, t_a, s, traj_est, traj_ref = align_kf_traj(npz_path, stream,printer=printer)

    offline_video = dict(np.load(npz_path))
    
    import os
    if not os.path.exists(plot_parent_dir):
        os.makedirs(plot_parent_dir)

    ape_statistics = traj_eval_and_plot(traj_est,traj_ref,plot_parent_dir,plot_name,printer)

    output_str = "#"*10+"Keyframes traj"+"#"*10+"\n"
    output_str += f"scale: {s}\n"
    output_str += f"rotation:\n{r_a}\n"
    output_str += f"translation:{t_a}\n"
    output_str += f"statistics:\n{ape_statistics}"
    printer.print(output_str,FontColor.EVAL)
    printer.print("#"*34,FontColor.EVAL)
    out_path=f'{plot_parent_dir}/metrics_kf_traj.txt'
    with open(out_path, 'w+') as fp:
        fp.write(output_str)
    if logger is not None:
        logger.log({'kf_ate_rmse':ape_statistics['rmse'],'pose_scale':s})

    offline_video["scale"]=np.array(s)
    np.savez(npz_path,**offline_video)

    return ape_statistics, s, r_a, t_a


def full_traj_eval(traj_filler, mapper, plot_parent_dir, plot_name, stream, logger, printer, fast_mode=False):
    traj_est_inv, dino_feats = traj_filler(stream)
    traj_est_lietorch = traj_est_inv.inv()
    traj_est = traj_est_lietorch.matrix().data.cpu().numpy()

    if not fast_mode:
        # refine non-keyframe-traj from the mapping
        # this is time-consuming with minimal tracking improvement
        for i in tqdm(range(traj_est.shape[0])):
            if dino_feats is None:
                img_feat = None
            else:
                img_feat = dino_feats[i]
            w2c_refined = mapper.refine_pose_non_key_frame(i,
                                                        torch.tensor(np.linalg.inv(traj_est[i])),
                                                        features=img_feat)
            traj_est[i] = np.linalg.inv(w2c_refined.cpu().numpy())

    kf_num = traj_filler.video.counter.value
    kf_timestamps = traj_filler.video.timestamp[:kf_num].cpu().int().numpy()
    kf_poses = SE3(traj_filler.video.poses[:kf_num].clone()).inv().matrix().data.cpu().numpy()
    traj_est[kf_timestamps] = kf_poses
    traj_est_not_align = traj_est.copy()

    import os
    if not os.path.exists(plot_parent_dir):
        os.makedirs(plot_parent_dir)
        
    save_traj(traj_est_not_align,f'{plot_parent_dir}/est_poses_full.txt')


    if isinstance(stream, RGB_NoPose):
        # We don't have GT pose to evaluate
        return traj_est_not_align, None, None

    r_a, t_a, s, traj_est, traj_ref = align_full_traj(traj_est, stream, printer)    

    ape_statistics = traj_eval_and_plot(traj_est,traj_ref,plot_parent_dir,plot_name,printer)
    output_str = "#"*10+"Full traj"+"#"*10+"\n"
    output_str += f"scale: {s}\n"
    output_str += f"rotation:\n{r_a}\n"
    output_str += f"translation:{t_a}\n"
    output_str += f"statistics:\n{ape_statistics}"
    printer.print(output_str,FontColor.EVAL)
    printer.print("#"*29,FontColor.EVAL)

    
    out_path=f'{plot_parent_dir}/metrics_full_traj.txt'
    with open(out_path, 'w+') as fp:
        fp.write(output_str)
    if logger is not None:
        logger.log({'full_ate_rmse':ape_statistics['rmse']})
    return traj_est_not_align, traj_est, traj_ref

def save_traj(traj_est,output_file):
    N = traj_est.shape[0]
    tum_poses = []
    for i in range(N):
        T = traj_est[i,:,:]

        r = Rotation.from_matrix(T[:3, :3])
        quater = r.as_quat()
        formatted_string = f"{i} {T[0,3]:.6f} {T[1,3]:.6f} {T[2,3]:.6f} {quater[0]:.6f} {quater[1]:.6f} {quater[2]:.6f} {quater[3]:.6f}\n"
        tum_poses.append(formatted_string)

    with open(output_file, "w") as f:
        f.writelines(tum_poses)
