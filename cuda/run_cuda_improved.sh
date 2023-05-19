#!/bin/sh

# sh run_cuda_basic.sh 

module load CUDA/10.1.243-GCC-8.3.0
nvcc -O2 cuda_improved.cu -lm -o cuda_improved
srun --reservation=fri --time=00:01:00 -G1 -n1 cuda_improved ../images/lili3.png 128 0 1 0 0 32 100
