#!/bin/sh
#SBATCH --job-name=cuda_improved
#SBATCH --time=10:00:00
#SBATCH --output=cuda_improved.txt
#SBATCH --reservation=fri
#SBATCH --gpus=1
#SBATCH --ntasks=1

module load CUDA/10.1.243-GCC-8.3.0
nvcc -O2 cuda_improved.cu -lm -o cuda_improved

IMAGE="../images/living_room.png"

INIT_STRATEGY=0
FUSION=0
EARLY_STOPPAGE=0
MEASURE_PSNR=1
MAX_ITERS=100

echo "[Image = $IMAGE] Running test case for different K. [BASIC with Shared Atomics]"
for K in 8 16 32 64 128 256 512 
do
    echo "K = $K"
    for i in {1..1}
    do
        echo "Iteration: $i"
        srun cuda_improved $IMAGE 128 $INIT_STRATEGY $FUSION $EARLY_STOPPAGE $MEASURE_PSNR $K $MAX_ITERS
    done
done

echo "[Image = $IMAGE] Running test case for same K. [BASIC with Shared Atomics]"
for BLOCK_SIZE in 32 64 128 256 512 1024
do
    echo "BLOCK SIZE = $BLOCK_SIZE"
    for i in {1..1}
    do
        echo "Iteration: $i"
        srun cuda_improved $IMAGE $BLOCK_SIZE $INIT_STRATEGY $FUSION $EARLY_STOPPAGE $MEASURE_PSNR 32 $MAX_ITERS
    done
done


INIT_STRATEGY=0
FUSION=1
EARLY_STOPPAGE=0
MEASURE_PSNR=1
MAX_ITERS=100

echo "[Image = $IMAGE] Running test case for different K. [FUSION ONLY]"
for K in 8 16 32 64 128
do
    echo "K = $K"
    for i in {1..1}
    do
        echo "Iteration: $i"
        srun cuda_improved $IMAGE 128 $INIT_STRATEGY $FUSION $EARLY_STOPPAGE $MEASURE_PSNR $K $MAX_ITERS
    done
done

echo "[Image = $IMAGE] Running test case for same K. [FULL OPTIMIZATION]"
for BLOCK_SIZE in 32 64 128 256 512 1024
do
    echo "BLOCK SIZE = $BLOCK_SIZE"
    for i in {1..1}
    do
        echo "Iteration: $i"
        srun cuda_improved $IMAGE $BLOCK_SIZE $INIT_STRATEGY $FUSION $EARLY_STOPPAGE $MEASURE_PSNR 32 $MAX_ITERS
    done
done

INIT_STRATEGY=1
FUSION=1
EARLY_STOPPAGE=1

echo "[Image = $IMAGE] Running test case for different K. [FUSION ONLY]"
for K in 8 16 32 64 128
do
    echo "K = $K"
    for i in {1..1}
    do
        echo "Iteration: $i"
        srun cuda_improved $IMAGE 128 $INIT_STRATEGY $FUSION $EARLY_STOPPAGE $MEASURE_PSNR $K $MAX_ITERS
    done
done

echo "[Image = $IMAGE] Running test case for same K. [FULL OPTIMIZATION]"
for BLOCK_SIZE in 32 64 128 256 512 1024
do
    echo "BLOCK SIZE = $BLOCK_SIZE"
    for i in {1..1}
    do
        echo "Iteration: $i"
        srun cuda_improved $IMAGE $BLOCK_SIZE $INIT_STRATEGY $FUSION $EARLY_STOPPAGE $MEASURE_PSNR 32 $MAX_ITERS
    done
done