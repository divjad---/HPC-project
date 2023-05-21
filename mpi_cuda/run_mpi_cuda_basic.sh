#!/bin/sh
module load OpenMPI/4.1.1-GCC-11.2.0
module load CUDA
export UCX_TLS=tcp
srun --reservation=fri-vr --partition=gpu nvcc -O2 -lm -lmpi mpi_cuda_basic.cu -o mpi_cuda_basic
srun --reservation=fri-vr --partition=gpu --mpi=pmix --gpus=2 --ntasks=2 --nodes=2 ./mpi_cuda_basic ../images/penguin.png 128 32 64
