#!/bin/sh
module load mpi/openmpi-4.1.3
module load CUDA

export UCX_TLS=tcp
export UCX_HANDLE_ERRORS=none

srun --reservation=fri-vr --partition=gpu nvcc -O2 -I/d/hpc/software/openmpi-4.1.3/build/include -L/d/hpc/software/openmpi-4.1.3/build/lib/ -lmpi -lm mpi_cuda_improved.cu -o mpi_cuda_improved

IMAGE="../images/living_room.png"
INIT_STRATEGY=1
FUSION=1
EARLY_STOPPAGE=1
MEASURE_PSNR=0
MAX_ITERS=100

echo "[Image = $IMAGE] Running test case for different K."
for K in 8 16 32 64 128
do
    echo "K = $K"
    for i in {1..1}
    do
        echo "Iteration: $i"
        echo "Single GPU"
        srun --reservation=fri-vr --partition=gpu --mpi=pmix --gpus=1 --gpu-bind=single:1 --ntasks=1 --gpus-per-task=1 --nodes=1 ./mpi_cuda_improved ../images/living_room.png 128 $INIT_STRATEGY $FUSION $EARLY_STOPPAGE $MEASURE_PSNR $K $MAX_ITERS
        echo "Double GPU"
        srun --reservation=fri-vr --partition=gpu --mpi=pmix --gpus=2 --gpu-bind=single:1 --ntasks=2 --gpus-per-task=1 --nodes=2 ./mpi_cuda_improved ../images/living_room.png 128 $INIT_STRATEGY $FUSION $EARLY_STOPPAGE $MEASURE_PSNR $K $MAX_ITERS
    done
done