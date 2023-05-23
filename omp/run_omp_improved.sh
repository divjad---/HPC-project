#!/bin/sh
#SBATCH --job-name=omp_improved
#SBATCH --time=10:00:00
#SBATCH --output=omp_improved.txt
#SBATCH --reservation=fri
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

export OMP_PLACES=cores
export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=64

gcc -O2 omp_improved.c --openmp -lm -o omp_improved

IMAGE="../images/living_room.png"
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
        srun omp_improved $IMAGE $INIT_STRATEGY $FUSION $EARLY_STOPPAGE $MEASURE_PSNR $K $MAX_ITERS
    done
done

echo "[Image = $IMAGE] Running test case for same K. [FULL OPTIMIZATION]"
for CPUS_PER_TASK in 2 4 8 16 32 64
do
    echo "CPUS_PER_TASK = $CPUS_PER_TASK"
    export OMP_NUM_THREADS=$CPUS_PER_TASK
    for i in {1..1}
    do
        echo "Iteration: $i"
        srun omp_improved $IMAGE $INIT_STRATEGY $FUSION $EARLY_STOPPAGE $MEASURE_PSNR 32 $MAX_ITERS
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
        srun omp_improved $IMAGE $INIT_STRATEGY $FUSION $EARLY_STOPPAGE $MEASURE_PSNR $K $MAX_ITERS
    done
done

echo "[Image = $IMAGE] Running test case for same K. [FULL OPTIMIZATION]"
for CPUS_PER_TASK in 2 4 8 16 32 64
do
    echo "CPUS_PER_TASK = $CPUS_PER_TASK"
    export OMP_NUM_THREADS=$CPUS_PER_TASK
    for i in {1..1}
    do
        echo "Iteration: $i"
        srun omp_improved $IMAGE $INIT_STRATEGY $FUSION $EARLY_STOPPAGE $MEASURE_PSNR 32 $MAX_ITERS
    done
done