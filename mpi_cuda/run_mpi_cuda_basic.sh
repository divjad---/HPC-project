#!/bin/sh
module load mpi/openmpi-4.1.3
module load CUDA
export UCX_TLS=tcp
export UCX_HANDLE_ERRORS=none
srun --reservation=fri-vr --partition=gpu nvcc -O2 -I/d/hpc/software/openmpi-4.1.3/build/include -L/d/hpc/software/openmpi-4.1.3/build/lib/ -lmpi -lm mpi_cuda_basic.cu -o mpi_cuda_basic
srun --reservation=fri-vr --partition=gpu --mpi=pmix --gpus=2 --gpu-bind=single:1 --ntasks=2 --gpus-per-task=1 --nodes=2 ./mpi_cuda_basic ../images/living_room.png 128 32 100
