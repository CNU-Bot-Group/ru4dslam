#!/bin/bash

mkdir -p datasets/TUM_RGBD
cd datasets/TUM_RGBD

wget https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_rpy.tgz
tar -xvzf rgbd_dataset_freiburg3_walking_rpy.tgz
rm rgbd_dataset_freiburg3_walking_rpy.tgz

cd ../..