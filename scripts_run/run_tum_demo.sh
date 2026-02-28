#!/bin/bash

python run.py ./configs/Dynamic/TUM_RGBD/freiburg3_walking_rpy.yaml
python gen_rum.py ./configs/Dynamic/TUM_RGBD/freiburg3_walking_rpy.yaml
python precompute_4dgs.py --cfg ./configs/Dynamic/TUM_RGBD/freiburg3_walking_rpy.yaml --ws test
python reconstruct_4dgs.py --cfg ./configs/Dynamic/TUM_RGBD/freiburg3_walking_rpy.yaml --ws test