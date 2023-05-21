#!/bin/sh
#SBATCH --job-name=cuda_basic
#SBATCH --time=10:00:00
#SBATCH --output=cuda_basic.txt
#SBATCH --reservation=fri
#SBATCH --gpus=1
#SBATCH --ntasks=1

module load CUDA/10.1.243-GCC-8.3.0
nvcc cuda_basic.cu -O2 -lm -o cuda_basic

IMAGE="../images/living_room.png"
MAX_ITERS=100

echo "[Image = $IMAGE] Running test case for different K."
for K in 8 16 32 64 128
do
    echo "K = $K"
    for i in {1..3}
    do
        echo "Iteration: $i"
        srun cuda_basic $IMAGE 128 $K $MAX_ITERS
    done
done

echo "[Image = $IMAGE] Running test case for same K."
for BLOCK_SIZE in 32 64 128 256 512 1024
do
    echo "BLOCK SIZE = $BLOCK_SIZE"
    for i in {1..3}
    do
        echo "Iteration: $i"
        srun cuda_basic $IMAGE $BLOCK_SIZE 32 $MAX_ITERS
    done
done

