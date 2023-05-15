#!/bin/sh
#SBATCH --job-name=cuda_improved
#SBATCH --time=00:5:00
#SBATCH --output=cuda_improved.txt
#SBATCH --reservation=fri
#SBATCH -G1
#SBATCH -n1

# sh run_cuda_improved.sh
# path init_strategy early_stoppage measurePSNR BLOCK_SIZE K MAX_ITER

module load CUDA/10.1.243-GCC-8.3.0
nvcc cuda_improved.cu -O2 -lm -o cuda_improved
# srun --time=00:01:00 -G1 -n1 cuda_improved ../images/lili3.png 1 1 1 256 32 64

srun cuda_improved ../images/lili.png 0 0 1 128 16 32
srun cuda_improved ../images/lili.png 0 0 1 256 16 32
srun cuda_improved ../images/lili.png 1 0 1 128 16 32
srun cuda_improved ../images/lili.png 1 0 1 256 16 32
srun cuda_improved ../images/lili.png 0 1 1 128 16 32
srun cuda_improved ../images/lili.png 0 1 1 256 16 32
srun cuda_improved ../images/lili.png 1 1 1 128 16 32
srun cuda_improved ../images/lili.png 1 1 1 256 16 32

srun cuda_improved ../images/cvetko.png 0 0 1 128 16 32
srun cuda_improved ../images/cvetko.png 0 0 1 256 16 32
srun cuda_improved ../images/cvetko.png 1 0 1 128 16 32
srun cuda_improved ../images/cvetko.png 1 0 1 256 16 32
srun cuda_improved ../images/cvetko.png 0 1 1 128 16 32
srun cuda_improved ../images/cvetko.png 0 1 1 256 16 32
srun cuda_improved ../images/cvetko.png 1 1 1 128 16 32
srun cuda_improved ../images/cvetko.png 1 1 1 256 16 32

srun cuda_improved ../images/lili3.png 0 0 1 128 32 32
srun cuda_improved ../images/lili3.png 0 0 1 256 32 32
srun cuda_improved ../images/lili3.png 1 1 1 128 32 32
srun cuda_improved ../images/lili3.png 1 1 1 256 32 32

srun cuda_improved ../images/leopard.png 0 0 1 128 32 32
srun cuda_improved ../images/leopard.png 0 0 1 256 32 32
srun cuda_improved ../images/leopard.png 0 1 1 128 32 32
srun cuda_improved ../images/leopard.png 0 1 1 256 32 32
srun cuda_improved ../images/leopard.png 1 0 1 128 32 32
srun cuda_improved ../images/leopard.png 1 0 1 256 32 32
srun cuda_improved ../images/leopard.png 1 1 1 128 32 32
srun cuda_improved ../images/leopard.png 1 1 1 256 32 32
