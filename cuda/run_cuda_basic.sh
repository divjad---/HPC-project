#!/bin/sh

# sh run_cuda_basic.sh 

module load CUDA/10.1.243-GCC-8.3.0
nvcc cuda_basic.cu -O2 -lm -o cuda_basic
srun --reservation=fri --time=00:01:00 -G1 -n1 cuda_basic ../images/street.png 256 32 64
