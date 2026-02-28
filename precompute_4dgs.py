import torch
import imageio
import os, os.path as osp
import numpy as np
import sys

from lib_prior.moca_processor import MoCaPrep
from lib_prior.preprocessor_utils import load_imgs, convert_from_mp4
from lib_prior.prior_loading import Saved2D, visualize_track
from lib_prior.moca_processor import mark_dynamic_region

from lib_render.render_helper import GS_BACKEND

from lib_moca.moca import moca_solve
from lib_moca.epi_helpers import analyze_track_epi, identify_tracks
from lib_moca.camera import MonocularCameras

from viz_utils import viz_list_of_colored_points_in_cam_frame
import logging
from lib_prior.moca_processor import *
from omegaconf import OmegaConf
from lib_moca.moca_misc import make_pair_list
import random
import cv2
# from ultralytics import YOLO # ttt mask
import torch.nn.functional as F

from colorama import Fore,Style
from time import gmtime, strftime

def seed_everything(seed):
    logging.info(f"seed: {seed}")
    print(f"seed: {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    logging.info(f"seed: {seed}")
    print(f"seed: {seed}")


def get_moca_processor(pre_cfg):
    moca_processor = MoCaPrep(
        dep_mode=getattr(
            pre_cfg, "dep_mode", "sensor"
        ),  # "depthcrafter", "metric3d", "uni"
        tap_mode=getattr(
            pre_cfg, "tap_mode", "bootstapir"
        ),  # "spatracker", "cotracker"
        flow_mode=getattr(pre_cfg, "flow_mode", "raft"),
        align_metric_flag=getattr(pre_cfg, "align_metric_flag", True),
    )
    return moca_processor


def load_imgs_from_dir(src):
    img_dir = osp.join(src, "images")
    img_fns = sorted(
        [it for it in os.listdir(img_dir) if it.endswith(".png") or it.endswith(".jpg")]
    )
    img_list = [imageio.imread(osp.join(img_dir, it))[..., :3] for it in img_fns]
    return img_list, img_fns


def load_imgs_from_mp4():
    raise RuntimeError("Not implemented yet")
    return


def preprocess(
    img_list: list,
    img_fns: list,
    ws: str,
    moca_processor: MoCaPrep,
    pre_cfg: OmegaConf,
    resample_for_dynamic=True,
):
    seed_everything(getattr(pre_cfg, "seed", 12345))
    start_t = time.time()
    logging.info("*" * 20 + " Preprocessing " + "*" * 20)
    logging.info(f"Working on {ws}, start phase-1 preprocessing")
    logging.info("*" * 20 + " Preprocessing " + "*" * 20)

    BOUNDARY_EHNAHCE_TH = getattr(pre_cfg, "boundary_enhance_th", -1)
    DEPTH_DIR_POSTFIX = "_depth_sharp" if BOUNDARY_EHNAHCE_TH > 0 else "_depth"

    EPI_TH = getattr(pre_cfg, "epi_th", 1e-3)
    DEPTH_BOUNDARY_TH = getattr(
        pre_cfg, "depth_boundary_th_prep", 1.0
    )  # this is in the median=1.0 space

    TAP_CHUNK_SIZE = getattr(pre_cfg, "tap_chunk_size", 5000)

    TEST_TRAINING_TIME_MODE = getattr(pre_cfg, "test_training_time_mode", False)

    if "cam" in pre_cfg:
        H_out, W_out = pre_cfg['cam']['H_out'], pre_cfg['cam']['W_out']
        H_edge, W_edge = pre_cfg['cam']['H_edge'], pre_cfg['cam']['W_edge']
        H_out_with_edge, W_out_with_edge = H_out + H_edge * 2, W_out + W_edge * 2

        H, W = pre_cfg['cam']['H'], pre_cfg['cam']['W']
        fx, fy, cx, cy = pre_cfg['cam']['fx'], pre_cfg['cam']['fy'], pre_cfg['cam']['cx'], pre_cfg['cam']['cy']
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

        known_camera_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    else:
        known_camera_K = None

    video_file_name = pre_cfg["data"]["output"] + "/" + pre_cfg["scene"] + "/video.npz"
    npz_path = video_file_name

    moca_processor.process(
        known_camera_K = known_camera_K,
        t_list=None,
        img_list=img_list,
        img_name_list=img_fns,
        save_dir=ws,
        n_track=getattr(pre_cfg, "n_track_uniform", 8192),
        # depth crafter
        depthcrafter_denoising_steps=getattr(
            pre_cfg, "depthcrafter_denoising_steps", 25
        ),
        metric_alignment_frames=getattr(pre_cfg, "metric_alignment_frames", 10),
        metric_alignment_first_quantil=getattr(
            pre_cfg, "metric_alignment_first_quantil", 0.7
        ),
        metric_alignment_bias_flag=getattr(pre_cfg, "metric_alignment_bias_flag", True),
        metric_alignment_kernel=getattr(pre_cfg, "metric_alignment_kernel", "cauchy"),
        metric_alignment_fscale=getattr(pre_cfg, "metric_alignment_fscale", 0.001),
        # TAP
        compute_tap=True,
        tap_chunk_size=TAP_CHUNK_SIZE,
        # Flow
        flow_steps=getattr(pre_cfg, "flow_steps", [1, 3]),
        epi_num_threads=getattr(pre_cfg, "epi_num_threads", 64),
        # Dep enhance for spatracker
        boundary_enhance_th=BOUNDARY_EHNAHCE_TH,  # if > 0 will create a sharp dir
        # boost
        compute_flow=getattr(pre_cfg, "compute_flow", True),
        depth_npz_path=npz_path,
        TEST_TRAINING_TIME_MODE=TEST_TRAINING_TIME_MODE,
    )

    if not resample_for_dynamic:
        duration = (time.time() - start_t) / 60.0
        logging.info(
            f"Preprocessing done, SKIP DYN RESAMPLE! time cost: {duration:.3f}min"
        )
        return

    logging.info("*" * 20 + " Preprocessing " + "*" * 20)
    logging.info(f"Working on {ws}, start phase-2 preprocessing, densify the fg TAP")


    s2d = (
        Saved2D(ws)
        .load_epi()
        .load_dep(f"{moca_processor.dep_mode}{DEPTH_DIR_POSTFIX}", DEPTH_BOUNDARY_TH, npz_path=npz_path)
        .normalize_depth(median_depth=1.0)
        .recompute_dep_mask(depth_boundary_th=DEPTH_BOUNDARY_TH)
        .load_track(f"*uniform*{moca_processor.tap_mode}", min_valid_cnt=4)
        .load_vos()
    )

    ### ttt mask exp test yolo

    # exp_mask_dir = osp.join(ws, "exp_mask_image")
    # os.makedirs(exp_mask_dir, exist_ok=True)

    # yolo_model = YOLO('pretrained/yolov9e-seg.pt')
    # yolo_mask_list = []
    # for image in img_list:
    #     image = (
    #         torch.from_numpy(image / 255.0)
    #         .clamp(0.0, 1.0)
    #         .permute(2, 0, 1)
    #         .to(device="cuda", dtype=torch.float32)
    #     )

    #     if image.shape[1] != 480 or image.shape[2] != 640:
    #         image = F.interpolate(
    #             image.unsqueeze(0),
    #             size=(480, 640), 
    #             mode='bilinear',  
    #             align_corners=False
    #         ).squeeze(0)
        
    #     combined_mask = torch.zeros((image.shape[1], image.shape[2]), device="cuda", dtype=torch.bool)

    #     results = yolo_model.predict(source=image.unsqueeze(0), classes=[0], save=False, stream=False, show=False, verbose=False, device="cuda")
    #     for result in results:
    #         masks = result.masks
    #         if masks is not None:
    #             for mask in masks.data:
    #                 mask = mask.to(torch.bool)
    #                 combined_mask |= mask 

    #     results = yolo_model.predict(source=image.unsqueeze(0), classes=[2], save=False, stream=False, show=False, verbose=False, device="cuda")
    #     for result in results:
    #         masks = result.masks
    #         if masks is not None:
    #             for mask in masks.data:
    #                 mask = mask.to(torch.bool)
    #                 combined_mask |= mask 

    #     # results = yolo_model.predict(source=image.unsqueeze(0), classes=[56], save=False, stream=False, show=False, verbose=False, device="cuda")
    #     # for result in results:
    #     #     masks = result.masks
    #     #     if masks is not None:
    #     #         for mask in masks.data:
    #     #             mask = mask.to(torch.bool)
    #     #             combined_mask |= mask 

    #     yolo_mask_list.append(combined_mask.cpu().numpy())

    # yolo_mask = np.stack(yolo_mask_list, 0)
    # if yolo_mask.shape[1] != img_list[0].shape[0] or yolo_mask.shape[2] != img_list[0].shape[1]:
    #     mask_tensor = torch.from_numpy(yolo_mask).float()

    #     # [n, 480, 640] -> [n, 1, 480, 640]
    #     mask_tensor = mask_tensor.unsqueeze(1)

    #     
    #     mask_tensor = F.interpolate(
    #         mask_tensor,
    #         size=(img_list[0].shape[0], img_list[0].shape[1]),  
    #         mode='bilinear',  
    #         align_corners=False
    #     )

    #     # [n, 1, 360, 480] -> [n, 360, 480]
    #     yolo_mask = mask_tensor.squeeze(1).numpy()
    #     # yolo_mask = np.resize(yolo_mask, (len(yolo_mask), 360, 480))
    # imageio.mimsave(
    #     osp.join(ws, "epi_resample_mask_yolo.mp4"),
    #     yolo_mask.astype(np.uint8) * 255,
    #     fps=2,
    # )

    # imageio.mimsave(
    #     osp.join(ws, "epi_resample_mask_yolo_image.mp4"),
    #     yolo_mask[:, :, :, np.newaxis] * np.stack(img_list), # .astype(np.uint8) * 255
    #     fps=2,
    # )

    # yolo_mask_images = yolo_mask[:, :, :, np.newaxis] * np.stack(img_list)
    # for i in range(yolo_mask_images.shape[0]):
    #     imageio.imwrite(osp.join(exp_mask_dir, f"{i:03d}_yolo.png"), yolo_mask_images[i])

    ### 

    if not TEST_TRAINING_TIME_MODE:
        if hasattr(s2d, "epi"):
            sample_mask = s2d.epi > EPI_TH
            
            imageio.mimsave(
                osp.join(ws, "epi_resample_mask_flow.mp4"),
                sample_mask.cpu().numpy().astype(np.uint8) * 255,
                fps=2,
            )

            imageio.mimsave(
                osp.join(ws, "epi_resample_mask_flow_image.mp4"),
                (sample_mask.cpu().numpy())[:, :, :, np.newaxis] * np.stack(img_list), # .astype(np.uint8) * 255
                fps=2,
            )

            ### ttt exp test flow

            # flow_mask_images = (sample_mask.cpu().numpy())[:, :, :, np.newaxis] * np.stack(img_list)
            # for i in range(flow_mask_images.shape[0]):
            #     imageio.imwrite(osp.join(exp_mask_dir, f"{i:03d}_flow.png"), flow_mask_images[i])
        # else:
        #     continuous_pair_list = make_pair_list(s2d.T, interval=[1, 4], dense_flag=True)
        #     F_list, epierr_list, _ = analyze_track_epi(
        #         continuous_pair_list, s2d.track, s2d.track_mask, H=s2d.H, W=s2d.W
        #     )
        #     track_static_selection, _ = identify_tracks(epierr_list, EPI_TH)
        #     sample_mask = mark_dynamic_region(
        #         s2d.track[:, ~track_static_selection],
        #         s2d.track_mask[:, ~track_static_selection],
        #         s2d.H,
        #         s2d.W,
        #         0.1,
        #     )
            ###
    motion_mask_dir = osp.join(ws, "motion_mask")
    import glob
    motion_mask_paths = glob.glob(osp.join(motion_mask_dir, "*.png"))
    motion_mask_paths.sort()

    motion_mask_list = []
    for motion_mask_path in motion_mask_paths:
        motion_mask = cv2.imread(motion_mask_path, cv2.IMREAD_GRAYSCALE)
        motion_mask = (motion_mask > 0)
        motion_mask_list.append(motion_mask)
    motion_mask_list = np.stack(motion_mask_list, 0)

    motion_mask_list = torch.Tensor(np.stack(motion_mask_list)).cuda() # .float()  # T,H,W
    
    sample_mask = motion_mask_list

    ### ttt test motion mask
    # motion_mask_images = (sample_mask.cpu().numpy())[:, :, :, np.newaxis] * np.stack(img_list)
    # for i in range(motion_mask_images.shape[0]):
    #     imageio.imwrite(osp.join(exp_mask_dir, f"{i:03d}_motion.png"), motion_mask_images[i])
    ###

    resampling_mask_dilate_ksize = getattr(pre_cfg, "resampling_mask_dilate_ksize", 7)
    sample_mask = (
        torch.nn.functional.max_pool2d(
            sample_mask[:, None].float(),
            kernel_size=resampling_mask_dilate_ksize,
            stride=1,
            padding=(resampling_mask_dilate_ksize - 1) // 2,
        )[:, 0]
        > 0.5
    )
    
    if not TEST_TRAINING_TIME_MODE:
        imageio.mimsave(
            osp.join(ws, "epi_resample_mask_motion_image.mp4"),
            (sample_mask.cpu().numpy())[:, :, :, np.newaxis] * np.stack(img_list), # .astype(np.uint8) * 255
            fps=2,
        )

        imageio.mimsave(
            osp.join(ws, "epi_resample_mask.gif"),
            sample_mask.cpu().numpy().astype(np.uint8) * 255,
        )
        imageio.mimsave(
            osp.join(ws, "epi_resample_mask_motion.mp4"),
            sample_mask.cpu().numpy().astype(np.uint8) * 255,
            fps=2,
        )

    moca_processor.compute_tap(
        ws=ws,
        save_name=f"dynamic_dep={moca_processor.dep_mode}",
        # n_track=8192 * 3,
        n_track=getattr(pre_cfg, "n_track_pdynamic", 8192 * 3),
        img_list=img_list,
        mask_list=sample_mask.detach().cpu().numpy() > 0,
        dep_list=moca_processor.load_dep_list(
            ws, f"{moca_processor.dep_mode}{DEPTH_DIR_POSTFIX}", npz_path=npz_path
        ),
        K = known_camera_K,
        # K=cams.default_K.detach().cpu().numpy(), # ! maintain the same K as the first infered static one
        max_viz_cnt=getattr(pre_cfg, "max_viz_cnt", 512),
        chunk_size=TAP_CHUNK_SIZE,
        TEST_TRAINING_TIME_MODE=TEST_TRAINING_TIME_MODE,
    )

    duration = (time.time() - start_t) / 60.0
    logging.info(f"Preprocessing done, time cost: {duration:.3f}min")
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("MoSca-V2 Preprocessing")
    parser.add_argument("--ws", type=str, help="Source folder", required=True)
    parser.add_argument("--cfg", type=str, help="profile yaml file path", required=True)
    parser.add_argument(
        "--skip_dynamic_resample", action="store_true", help="skip dynamic resample"
    )
    args, unknown = parser.parse_known_args()

    def load_config_with_inheritance(config_path) -> OmegaConf:


        cfg = OmegaConf.load(config_path)
        
        if "inherit_from" in cfg:
            inherit_path = cfg.inherit_from
            
            parent_cfg = load_config_with_inheritance(inherit_path)

            del cfg.inherit_from
            
            cfg = OmegaConf.merge(parent_cfg, cfg)
        
        return cfg

    cfg = load_config_with_inheritance(args.cfg)
    cli_cfg = OmegaConf.from_dotlist([arg.lstrip("--") for arg in unknown])
    prep_cfg = OmegaConf.merge(cfg, cli_cfg)

    output_folder = cfg['data']['output'] + '/' + cfg['scene']
    start_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    start_info = "-"*30+Fore.LIGHTRED_EX+\
                 f"\nStart Preprocess at {start_time},\n"+Style.RESET_ALL+ \
                 f"   scene: {cfg['dataset']}-{cfg['scene']},\n" \
                 f"   output: {output_folder}\n"+ \
                 "-"*30
    with open(f'{output_folder}/log.txt', 'a+') as f:
        f.write(start_info)
        
    args.ws = os.path.join(prep_cfg["data"]["output"] + "/" + prep_cfg["scene"], args.ws)
    
    img_list, img_fns = load_imgs_from_dir(args.ws)

    moca_processor = get_moca_processor(prep_cfg)

    preprocess(
        img_list=img_list,
        img_fns=img_fns,
        ws=args.ws,
        moca_processor=moca_processor,
        pre_cfg=prep_cfg,
        resample_for_dynamic=not args.skip_dynamic_resample,
    )

    end_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    print("-"*30+Fore.LIGHTRED_EX+f"\nPreprocess finishes!\n"+Style.RESET_ALL+f"{end_time}\n"+"-"*30)
    with open(f'{output_folder}/log.txt', 'a+') as f:
        f.write("-"*30+Fore.LIGHTRED_EX+f"\nPreprocess finishes!\n"+Style.RESET_ALL+f"{end_time}\n"+"-"*30)