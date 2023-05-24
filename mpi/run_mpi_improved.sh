#!/bin/sh
#SBATCH --job-name=mpi_improved
#SBATCH --time=10:00:00
#SBATCH --output=mpi_improved.txt
#SBATCH --reservation=fri
#SBATCH --nodes=2

module load OpenMPI/4.1.0-GCC-10.2.0

mpicc -O3 mpi_improved.c -o mpi_improved -lm

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
    for i in {1..3}
    do
        echo "Iteration: $i"
        srun --ntasks-64 --mpi=pmix omp_improved $IMAGE $INIT_STRATEGY $FUSION $EARLY_STOPPAGE $MEASURE_PSNR $K $MAX_ITERS
    done
done

echo "[Image = $IMAGE] Running test case for same K. [FUSION ONLY]"
for N_TASKS in 2 4 8 16 32 64
do
    echo "N_TASKS = $N_TASKS"
    for i in {1..3}
    do
        echo "Iteration: $i"
        srun --ntasks=$N_TASKS --mpi=pmix omp_improved $IMAGE $INIT_STRATEGY $FUSION $EARLY_STOPPAGE $MEASURE_PSNR 32 $MAX_ITERS
    done
done

INIT_STRATEGY=1
FUSION=1
EARLY_STOPPAGE=1

echo "[Image = $IMAGE] Running test case for different K. [FULL OPTIMIZATION]"
for K in 8 16 32 64 128
do
    echo "K = $K"
    for i in {1..3}
    do
        echo "Iteration: $i"
        srun --ntasks=64 omp_improved $IMAGE $INIT_STRATEGY $FUSION $EARLY_STOPPAGE $MEASURE_PSNR $K $MAX_ITERS
    done
done

echo "[Image = $IMAGE] Running test case for same K. [FULL OPTIMIZATION]"
for N_TASKS in 2 4 8 16 32 64
do
    echo "N_TASKS = $N_TASKS"
    for i in {1..3}
    do
        echo "Iteration: $i"
        srun --ntasks=$N_TASKS omp_improved $IMAGE $INIT_STRATEGY $FUSION $EARLY_STOPPAGE $MEASURE_PSNR 32 $MAX_ITERS
    done
done