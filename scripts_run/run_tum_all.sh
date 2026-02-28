#!/bin/bash

python run.py ./configs/Dynamic/TUM_RGBD/freiburg3_walking_rpy.yaml
python gen_rum.py ./configs/Dynamic/TUM_RGBD/freiburg3_walking_rpy.yaml
python precompute_4dgs.py --cfg ./configs/Dynamic/TUM_RGBD/freiburg3_walking_rpy.yaml --ws test
python reconstruct_4dgs.py --cfg ./configs/Dynamic/TUM_RGBD/freiburg3_walking_rpy.yaml --ws test

python run.py ./configs/Dynamic/TUM_RGBD/freiburg3_walking_xyz.yaml
python gen_rum.py ./configs/Dynamic/TUM_RGBD/freiburg3_walking_xyz.yaml
python precompute_4dgs.py --cfg ./configs/Dynamic/TUM_RGBD/freiburg3_walking_xyz.yaml --ws test
python reconstruct_4dgs.py --cfg ./configs/Dynamic/TUM_RGBD/freiburg3_walking_xyz.yaml --ws test

python run.py ./configs/Dynamic/TUM_RGBD/freiburg3_walking_static.yaml
python gen_rum.py ./configs/Dynamic/TUM_RGBD/freiburg3_walking_static.yaml
python precompute_4dgs.py --cfg ./configs/Dynamic/TUM_RGBD/freiburg3_walking_static.yaml --ws test
python reconstruct_4dgs.py --cfg ./configs/Dynamic/TUM_RGBD/freiburg3_walking_static.yaml --ws test

python run.py ./configs/Dynamic/TUM_RGBD/freiburg3_sitting_rpy.yaml
python gen_rum.py ./configs/Dynamic/TUM_RGBD/freiburg3_sitting_rpy.yaml
python precompute_4dgs.py --cfg ./configs/Dynamic/TUM_RGBD/freiburg3_sitting_rpy.yaml --ws test
python reconstruct_4dgs.py --cfg ./configs/Dynamic/TUM_RGBD/freiburg3_sitting_rpy.yaml --ws test

python run.py ./configs/Dynamic/TUM_RGBD/freiburg3_sitting_xyz.yaml
python gen_rum.py ./configs/Dynamic/TUM_RGBD/freiburg3_sitting_xyz.yaml
python precompute_4dgs.py --cfg ./configs/Dynamic/TUM_RGBD/freiburg3_sitting_xyz.yaml --ws test
python reconstruct_4dgs.py --cfg ./configs/Dynamic/TUM_RGBD/freiburg3_sitting_xyz.yaml --ws test

python run.py ./configs/Dynamic/TUM_RGBD/freiburg3_sitting_static.yaml
python gen_rum.py ./configs/Dynamic/TUM_RGBD/freiburg3_sitting_static.yaml
python precompute_4dgs.py --cfg ./configs/Dynamic/TUM_RGBD/freiburg3_sitting_static.yaml --ws test
python reconstruct_4dgs.py --cfg ./configs/Dynamic/TUM_RGBD/freiburg3_sitting_static.yaml --ws test