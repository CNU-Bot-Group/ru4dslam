import numpy as np
import torch
import argparse
import os
import cv2
import glob

from src import config
from src.utils.dyn_uncertainty.uncertainty_model import generate_uncertainty_mlp
from src.utils.mono_priors.img_feature_extractors import  get_feature_extractor
from src.masker import Masker


from src.utils.datasets import get_dataset
from src.utils.mono_priors.metric_depth_estimators import get_metric_depth_estimator, predict_metric_depth

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

    # d0 = depths[0]

    # d_mask = data["valid_depth_masks"]
    # d_m = d_mask[0]

    # ###
    # input_folder = "/root/autodl-tmp/catkin_ws/src/ru4d-SLAM/datasets/Wild_SLAM_iPhone/shopping"
    # ###

    H_out, W_out = cfg['cam']['H_out'], cfg['cam']['W_out']
    H_edge, W_edge = cfg['cam']['H_edge'], cfg['cam']['W_edge']

    H_out_with_edge, W_out_with_edge = H_out + H_edge * 2, W_out + W_edge * 2
    
    down_scale = 8
    slice_h = slice(down_scale // 2 - 1, ht//down_scale*down_scale+1, down_scale)
    slice_w = slice(down_scale // 2 - 1, wd//down_scale*down_scale+1, down_scale)

    n_features = cfg["mapping"]["uncertainty_params"]["feature_dim"]
    uncer_network = generate_uncertainty_mlp(n_features) # , time_dim=1

    feat_extractor = get_feature_extractor(cfg)

    weight = torch.load(save_dir + "/uncertainty_mlp_weight.pth")
    uncer_network.load_state_dict(weight)
    
    test_dir = os.path.join(output_folder, "test", "test")
    os.makedirs(test_dir, exist_ok=True)

    uncer_weight_list = []
    dilate_mask = False # ttt

    dp = np.ceil(3*480/ht)
    kernel = np.ones((3, 3), np.uint8)

    image_list = []
    uncer_mask_list = []
    motion_mask_list = []
    uncer_img_list = []
    # for i, tstamp in enumerate(range(0, n_img, 10)):
    metric_depth_estimator = get_metric_depth_estimator(cfg)
    
    color_paths = sorted(
        glob.glob(f'{input_folder}/rgb/frame*.png'))
    from tqdm import tqdm

    save_path = os.path.join(input_folder, "depth_m")
    os.makedirs(save_path, exist_ok=True)
    for i, color_path in tqdm(enumerate(color_paths)):
        color_data_fullsize = cv2.imread(color_path)

        color_data = cv2.resize(color_data_fullsize, (W_out_with_edge, H_out_with_edge))
        color_data = torch.from_numpy(color_data).float().permute(2, 0, 1)[[2, 1, 0], :, :] / 255.0  # bgr -> rgb, [0, 1]
        color_data = color_data.unsqueeze(dim=0)  # [1, 3, h, w]

        # crop image edge, there are invalid value on the edge of the color image
        if W_edge > 0:
            edge = W_edge
            color_data = color_data[:, :, :, edge:-edge]

        if H_edge > 0:
            edge = H_edge
            color_data = color_data[:, :, edge:-edge, :]

        mono_depth = predict_metric_depth(metric_depth_estimator,i,color_data,cfg,device) * 1000

        mono_depth[mono_depth > 65535] = 0

        cv2.imwrite(os.path.join(save_path, f"depth_{i:05d}.png"), mono_depth.cpu().numpy().astype(np.uint16))
