#!/bin/sh
#SBATCH --job-name=cuda_basic
#SBATCH --time=00:5:00
#SBATCH --output=cuda_basic.txt
#SBATCH --reservation=fri
#SBATCH -G1
#SBATCH -n1

module load CUDA/10.1.243-GCC-8.3.0
nvcc cuda_basic.cu -O2 -lm -o cuda_basic

srun --reservation=fri --time=00:01:00 -G1 -n1 cuda_basic ../images/cvetko.png 128 32 100
srun --reservation=fri --time=00:01:00 -G1 -n1 cuda_basic ../images/cvetko.png 128 32 100
srun --reservation=fri --time=00:01:00 -G1 -n1 cuda_basic ../images/cvetko.png 128 32 100
srun --reservation=fri --time=00:01:00 -G1 -n1 cuda_basic ../images/cvetko.png 128 32 100

