#!/bin/bash

conda install nvidia::cuda-toolkit==12.8.1 -y
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install glfw libigl joblib numpy opencv-python plyfile PyOpenGL pyrender pyyaml scikit-image scikit-learn screeninfo setuptools tqdm trimesh
export TORCH_CUDA_ARCH_LIST="12.0"
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
