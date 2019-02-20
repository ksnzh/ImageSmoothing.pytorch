#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/

#python setup.py build_ext --inplace
#rm -rf build

CUDA_ARCH="-gencode arch=compute_30,code=sm_30 \
           -gencode arch=compute_35,code=sm_35 \
           -gencode arch=compute_50,code=sm_50 \
           -gencode arch=compute_52,code=sm_52 \
           -gencode arch=compute_60,code=sm_60 \
           -gencode arch=compute_61,code=sm_61 "

# compile edge_computation
cd EdgeComputation/src
echo "Compiling edge computation kernels by nvcc..."
nvcc -c -o edge_computation_kernel.cu.o edge_computation_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
python build.py

# compile edge_detector
cd EdgeDetector/src
echo "Compiling edge detector kernels by nvcc..."
nvcc -c -o edge_detector_kernel.cu.o edge_detector_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
python build.py

# compile smooth_edge_term_loss
cd SmoothAndEdgeTermLoss/src
echo "Compiling smooth and edge term loss kernels by nvcc..."
nvcc -c -o smooth_edge_term_loss_kernel.cu.o smooth_edge_term_loss_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
python build.py