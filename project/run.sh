#!/bin/sh
#SBATCH --job-name=project
#SBATCH --time=00:1:00
#SBATCH --output=project.txt
#SBATCH --reservation=fri
#SBATCH -G1
#SBATCH -n1

module load CUDA/10.1.243-GCC-8.3.0
#module load CUDA
nvcc img_comp.cu -O2 -lm -o img_compression
srun --reservation=fri -G1 -n1 img_compression penguin.jpg
#srun --reservation=fri-vr --partition=gpu -G1 -n1 img_compression penguin.jpg
