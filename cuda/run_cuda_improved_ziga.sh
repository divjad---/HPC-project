#!/bin/sh

# sh run_cuda_basic.sh 

module load CUDA/10.1.243-GCC-8.3.0
nvcc -O2 cuda_improved_ziga.cu -lm -o cuda_improved_ziga
srun --reservation=fri --time=00:01:00 --constraint=AMD -G1 -n1 cuda_improved_ziga ../images/street.png 128 0 0 0 0 4 64
