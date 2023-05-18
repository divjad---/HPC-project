#!/bin/sh
module load OpenMPI/4.1.0-GCC-10.2.0
module load CUDA/10.1.243-GCC-8.3.0
nvcc -O2 -lm -I/usr/include/openmpi-x86_64/ -lmpi mpi_cuda_basic.cu -o mpi_cuda_basic
srun --reservation=fri --mpi=pmix --gpus=2 --ntasks=2 --nodes=2 mpi_cuda_basic ../images/street.png 128 32 64
