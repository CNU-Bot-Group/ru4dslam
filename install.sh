#!/bin/bash

ENV_NAME=ru4d
NUMPY_VERSION=1.26.4

# conda remove -n $ENV_NAME --all -y
# conda create -n $ENV_NAME gcc_linux-64=9 gxx_linux-64=9 python=3.10 mkl=2023.1.0 numpy=$NUMPY_VERSION -y

# conda activate $ENV_NAME

# which python
# which pip

# CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
# CPP=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
# CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
# $CC --version
# $CXX --version

# ################################################################################    
# pip install numpy==$NUMPY_VERSION
# conda install pytorch==2.1.2 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
# conda install fvcore iopath -c fvcore -c iopath -c conda-forge -y --verbose
# conda install nvidiacub -c bottler -y

# # pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
# # pip install fvcore==0.1.5.post20221221 --no-build-isolation
# # pip install iopath==0.1.10 --no-build-isolation
# conda install nvidiacub -c bottler -y

# pip install pyg_lib torch_scatter torch_geometric torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
# ################################################################################

# ################################################################################
# echo "Install other dependencies..."
# conda install xformers -c xformers -y
# # conda install pytorch3d -c pytorch3d -y
# # wget https://api.anaconda.org/download/pytorch3d/pytorch3d/0.7.8/linux-64/pytorch3d-0.7.8-py310_cu121_pyt212.tar.bz2
# conda install pytorch3d-0.7.8-py310_cu121_pyt212.tar.bz2
# python -m pip install -e . --no-build-isolation

pip install -r requirements.txt --no-build-isolation
################################################################################

################################################################################
echo "Install thirdparty..."
pip install -e thirdparty/lietorch/ --no-build-isolation
pip install -e thirdparty/diff-gaussian-rasterization-w-pose/ --no-build-isolation
pip install -e thirdparty/sam2/ --no-build-isolation

pip install lib_render/simple-knn --no-build-isolation
pip install lib_render/diff-gaussian-rasterization-alphadep-add3 --no-build-isolation
pip install lib_render/diff-gaussian-rasterization-alphadep --no-build-isolation
pip install lib_render/gof-diff-gaussian-rasterization --no-build-isolation
################################################################################

################################################################################
pip install -U scikit-learn 
pip install -U scipy
pip install mmcv-full==1.7.2 --no-build-isolation
################################################################################

cd pretrained
bash download_ckpts.sh
cd ..