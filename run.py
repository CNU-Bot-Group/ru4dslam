import numpy as np
import torch
import argparse
import os

from src import config
from src.slam import SLAM
from src.utils.datasets import get_dataset
from time import gmtime, strftime
from colorama import Fore,Style

import random
def setup_seed(seed):
    # return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--refine', action='store_true', help='To refine the mapping.')
    parser.add_argument('--nodes', action='store_true', help='To get all nodes.')
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    cfg = config.load_config(args.config)
    setup_seed(cfg['setup_seed'])
    if cfg['fast_mode']:
        # Force the final refine iterations to be 3000 if in fast mode
        cfg['mapping']['final_refine_iters'] = 3000

    output_dir = cfg['data']['output']
    output_dir = output_dir+f"/{cfg['scene']}"

    start_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    start_info = "-"*30+Fore.LIGHTRED_EX+\
                 f"\nStart ru4d-SLAM nodes:{str(args.nodes)} at {start_time},\n"+Style.RESET_ALL+ \
                 f"   scene: {cfg['dataset']}-{cfg['scene']},\n" \
                 f"   output: {output_dir}\n"+ \
                 "-"*30
    print(start_info)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config.save_config(cfg, f'{output_dir}/cfg.yaml')

    dataset = get_dataset(cfg)

    slam = SLAM(cfg,dataset)
    if args.refine:
        slam.final_refine()
    elif args.nodes:
        slam.get_all_nodes(get_all=True, iter_per_frame=30)
    else:
        slam.run()

    end_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    print("-"*30+Fore.LIGHTRED_EX+f"\nSLAM finishes!\n"+Style.RESET_ALL+f"{end_time}\n"+"-"*30)
    with open(f'{output_dir}/log.txt', 'a+') as f:
        f.write(start_info)
        f.write("-"*30+Fore.LIGHTRED_EX+f"\nSLAM nodes:{str(args.nodes)} finishes!\n"+Style.RESET_ALL+f"{end_time}\n"+"-"*30)

