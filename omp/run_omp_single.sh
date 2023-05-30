#!/bin/sh
#SBATCH --job-name=omp_basic
#SBATCH --time=10:00:00
#SBATCH --output=omp_basic.txt
#SBATCH --reservation=fri
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

export OMP_PLACES=cores
export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=2

gcc -O2 omp_improved.c --openmp -lm -o omp_improved

IMAGE="../images/living_room.png"
INIT_STRATEGY=0
FUSION=0
EARLY_STOPPAGE=0
MEASURE_PSNR=0
MAX_ITERS=100

echo "[Image = $IMAGE] Running test case for different K."
for K in 32
do
    echo "K = $K"
    for i in {1..1}
    do
        echo "Iteration: $i"
        srun omp_improved $IMAGE $INIT_STRATEGY $FUSION $EARLY_STOPPAGE $MEASURE_PSNR $K $MAX_ITERS
    done
done