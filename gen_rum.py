import numpy as np
import torch
import argparse
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2

from src import config
from src.utils.dyn_uncertainty.uncertainty_model import generate_uncertainty_mlp
from src.utils.mono_priors.img_feature_extractors import predict_img_features, get_feature_extractor
from src.utils.dyn_uncertainty import mapping_utils as map_utils
from src.masker import Masker

from src.utils.datasets import get_dataset
import imageio
from colorama import Fore,Style
from time import gmtime, strftime
from tqdm import tqdm

if __name__ == '__main__':

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    masker = Masker()

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()

    cfg = config.load_config(args.config)

    stream = get_dataset(cfg)

    save_dir = cfg["data"]["output"] + "/" + cfg["scene"]
    device = cfg['device']
    ht = cfg['cam']['H_out']
    wd = cfg['cam']['W_out']

    input_folder = cfg['data']['input_folder']
    if "ROOT_FOLDER_PLACEHOLDER" in input_folder:
        input_folder = input_folder.replace("ROOT_FOLDER_PLACEHOLDER", cfg['data']['root_folder'])
    
    output_folder = cfg['data']['output'] + '/' + cfg['scene']
    data = np.load(os.path.join(output_folder, "video.npz"))
    timestamps = data['timestamps']
    depths = data['depths']

    test_training_time_mode = cfg['test_training_time_mode']
    start_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    start_info = "-"*30+Fore.LIGHTRED_EX+\
                 f"\nStart RUM at {start_time},\n"+Style.RESET_ALL+ \
                 f"   scene: {cfg['dataset']}-{cfg['scene']},\n" \
                 f"   output: {output_folder}\n"+ \
                 "-"*30

    H_out, W_out = cfg['cam']['H_out'], cfg['cam']['W_out']
    H_edge, W_edge = cfg['cam']['H_edge'], cfg['cam']['W_edge']

    H_out_with_edge, W_out_with_edge = H_out + H_edge * 2, W_out + W_edge * 2
    
    down_scale = 8
    slice_h = slice(down_scale // 2 - 1, ht//down_scale*down_scale+1, down_scale)
    slice_w = slice(down_scale // 2 - 1, wd//down_scale*down_scale+1, down_scale)

    n_features = cfg["mapping"]["uncertainty_params"]["feature_dim"]
    uncer_network = generate_uncertainty_mlp(n_features)

    feat_extractor = get_feature_extractor(cfg)

    weight = torch.load(save_dir + "/uncertainty_mlp_weight.pth")
    uncer_network.load_state_dict(weight)
    
    motion_mask_dir = os.path.join(output_folder, "test/motion_mask")
    images_dir = os.path.join(output_folder, "test/images")
    instance_mask_dir = os.path.join(output_folder, "test/instance_mask")
    background_mask_dir = os.path.join(output_folder, "test/backround_mask")
    uncer_img_dir = os.path.join(output_folder, "test/uncertainty_image")
    uncer_mask_dir = os.path.join(output_folder, "test/uncer_mask")
    depth_dir = os.path.join(output_folder, "test/ru4d_depth")
    uncer_dir = os.path.join(output_folder, "test/uncertainty")
    os.makedirs(motion_mask_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(instance_mask_dir, exist_ok=True)
    os.makedirs(background_mask_dir, exist_ok=True)
    os.makedirs(uncer_img_dir, exist_ok=True)
    os.makedirs(uncer_mask_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(uncer_dir, exist_ok=True)
    
    test_dir = os.path.join(output_folder, "test", "test")
    os.makedirs(test_dir, exist_ok=True)

    viz_dir = os.path.join(output_folder, "test", "viz")
    os.makedirs(viz_dir, exist_ok=True)

    uncer_weight_list = []
    dilate_mask = False # ttt

    dp = np.ceil(3*480/ht)
    kernel = np.ones((3, 3), np.uint8)

    image_list = []
    uncer_mask_list = []
    motion_mask_list = []
    uncer_img_list = []

    for i, tstamp in tqdm(enumerate(timestamps), total=len(timestamps)):
        depth = depths[i]
        
        np.save(f"{depth_dir}/{i:03d}.npy", depth)

        tstamp = int(tstamp)

        _, color_data, depth_data, _ = stream[tstamp]
        if not test_training_time_mode:
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            im = axs[0, 0].imshow(depth_data, cmap='jet', vmin=0)
            plt.colorbar(im, ax=axs[0, 0])
            im = axs[0, 1].imshow(depth, cmap='jet', vmin=0)
            plt.colorbar(im, ax=axs[0, 1])
            im = axs[0, 2].imshow(depth - depth_data.detach().cpu().numpy(), cmap='jet')
            plt.colorbar(im, ax=axs[0, 2])

            mono_depth_path = f"{output_folder}/mono_priors/depths/{tstamp:05d}.npy"
            mono_depth = np.load(mono_depth_path)

            im = axs[1, 1].imshow(mono_depth, cmap='jet', vmin=0)
            plt.colorbar(im, ax=axs[1, 1])
            im = axs[1, 2].imshow(depth - mono_depth, cmap='jet')
            plt.colorbar(im, ax=axs[1, 2])

            plt.savefig(f"{test_dir}/{tstamp:03d}.png")
            plt.close(fig)

        color_data_np = (color_data.squeeze(0).detach().permute(1, 2, 0)*255).to(torch.uint8).cpu().numpy()
        cv2.imwrite(f"{images_dir}/{i:03d}.png", cv2.cvtColor(color_data_np, cv2.COLOR_RGB2BGR))

        if not test_training_time_mode:
            image_list.append(color_data_np)

        image = color_data
        dino_features = predict_img_features(feat_extractor,tstamp,image,cfg,device)
        
        uncer = uncer_network(dino_features, i)
        train_frac = cfg['mapping']['uncertainty_params']['train_frac_fix']

        h = ht
        w = wd
        uncer = torch.clip(uncer, min=0.1) + 1e-3
        uncer = uncer.unsqueeze(0).unsqueeze(0)
        uncer = F.interpolate(uncer, size=(h, w), mode="bilinear").squeeze(0).squeeze(0).detach()
        data_rate = 1 + 1 * map_utils.compute_bias_factor(train_frac, 0.8)

        uncer = (uncer - 0.1) * data_rate + 0.1

        uncer = uncer**2
        mask = (uncer > 3.5).detach().cpu().numpy()

        if dilate_mask:
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        if not test_training_time_mode:
            mask_img = (mask.astype(np.uint8)*255)[:, :, np.newaxis].repeat(3, axis=2)
            cv2.imwrite(f"{uncer_mask_dir}/{i:03d}.png", cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR))
            uncer_mask_list.append(cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR))

            uncertainty_map = uncer.cpu()

            plt.imshow(uncertainty_map.numpy(), cmap="jet", vmin=0, vmax=5)
            plt.axis('off')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f"{uncer_img_dir}/{i:03d}.png")
            plt.close() 

            uncer_img_list.append(cv2.cvtColor(cv2.imread(f"{uncer_img_dir}/{i:03d}.png"), cv2.COLOR_BGR2RGB))

        uncer_weight = torch.clamp(0.5/uncer, 0.0, 1.0)

        uncer_weight_list.append(uncer_weight)
        if not test_training_time_mode:
            sta_mask = torch.where(uncer_weight < 0.1, 0.0, uncer_weight)
            plt.imshow(sta_mask.detach().cpu().numpy(), cmap="jet", vmin=0, vmax=1)
            plt.axis('off')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f"{uncer_img_dir}/{i:03d}_weight.png")
            plt.close() 
        
        masks = masker.mask_generator.generate(color_data_np, mask, nms=False) # True

        anns = masks

        anns_instance = []

        motion_mask = mask

        for ann in anns:
            m = ann['segmentation']

            ratio0 = np.count_nonzero(m & mask)/ np.count_nonzero(m)

            if ratio0 < 0.2:
                continue
            
            if dilate_mask:
                m = cv2.dilate(m.astype(np.uint8), kernel, iterations=1).astype(bool)

            if not test_training_time_mode:
                anns_instance.append(ann)

            if motion_mask is None:
                motion_mask = m
            else:
                motion_mask = motion_mask | m

        if motion_mask is None:
            background_mask = mask
            motion_mask = np.zeros_like(mask)
        else:
            background_mask = mask.copy()  
            background_mask[motion_mask] = False 

            cv2.imwrite(f"{motion_mask_dir}/{i:03d}.png", motion_mask.astype(np.uint8)*255)
        if not test_training_time_mode:
            motion_mask_list.append(motion_mask.astype(np.uint8)*255)

            cv2.imwrite(f"{background_mask_dir}/{i:03d}.png", background_mask.astype(np.uint8)*255)

        anns = anns_instance

        for ann in anns:
            u0, v0, w, h = ann['bbox']
            u1, v1 = u0 + w, v0 + h
            ann["box_size"] = (u1 - u0) * (v1 - v0)

        k = 0

        while 1:
            if len(anns) == 0:
                break
            anns = sorted(anns, key=(lambda x: x['box_size']))
            if k == len(anns)-1:
                break
            ann = anns[k]
            mask0 = ann['segmentation']
            u0_0, v0_0, w, h = ann['bbox']
            u0_1, v0_1 = u0_0 + w, v0_0 + h
            for j in range(k+1, len(anns)):
                ann1 = anns[j]
                mask1 = ann1['segmentation']
                u1_0, v1_0, w, h = ann1['bbox']
                u1_1, v1_1 = u1_0 + w, v1_0 + h

                inter = max(0, min(u0_1, u1_1) - max(u0_0, u1_0)) * max(0, min(v0_1, v1_1) - max(v0_0, v1_0))
                inter_ratio = inter / (u0_1 - u0_0) / (v0_1 - v0_0)
                if inter_ratio > 0.8:
                    ann1["segmentation"] = mask1 | mask0
                    u1_0, v1_0, w, h = min(u0_0, u1_0), min(v0_0, v1_0), max(u0_1, u1_1) - min(u0_0, u1_0), max(v0_1, v1_1) - min(v0_0, v1_0)
                    u1_1, v1_1 = u1_0 + w, v1_0 + h
                    ann1["bbox"] = [u1_0, v1_0, w, h]
                    ann1["box_size"] = (u1_1 - u1_0) * (v1_1 - v1_0)
                    anns.pop(k)
                    break
                elif j == len(anns) - 1:
                    k += 1
        
        img = np.zeros((mask.shape[0], mask.shape[1], 3))
        for ann in anns:
            m = ann['segmentation']
            
            color_mask = np.random.random(3)
            img[m] = color_mask

        if not test_training_time_mode:
            cv2.imwrite(f"{instance_mask_dir}/{i:03d}.png", cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    uncer_weight_list = torch.stack(uncer_weight_list, dim=0)
    torch.save(uncer_weight_list, f"{uncer_dir}/uncer_weight.ckpt")
    if not test_training_time_mode:
        fps = 5
        imageio.mimsave(f"{viz_dir}/iamge.gif", image_list, fps=fps)
        imageio.mimsave(f"{viz_dir}/uncer_img.gif", uncer_img_list, fps=fps)
        imageio.mimsave(f"{viz_dir}/motion_mask.gif", motion_mask_list, fps=fps)
        imageio.mimsave(f"{viz_dir}/uncer_mask.gif", uncer_mask_list, fps=fps)

        imageio.mimsave(f"{viz_dir}/image.mp4", image_list, fps=fps)
        imageio.mimsave(f"{viz_dir}/uncer_img.mp4", uncer_img_list, fps=fps)
        imageio.mimsave(f"{viz_dir}/motion_mask.mp4", motion_mask_list, fps=fps)
        imageio.mimsave(f"{viz_dir}/uncer_mask.mp4", uncer_mask_list, fps=fps)
        
    end_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    print("-"*30+Fore.LIGHTRED_EX+f"\nRUM finishes!\n"+Style.RESET_ALL+f"{end_time}\n"+"-"*30)
    with open(f'{output_folder}/log.txt', 'a+') as f:
        f.write(start_info)
        f.write("-"*30+Fore.LIGHTRED_EX+f"\nRUM finishes!\n"+Style.RESET_ALL+f"{end_time}\n"+"-"*30)