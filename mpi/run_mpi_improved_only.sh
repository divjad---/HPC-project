#!/bin/sh
module load OpenMPI/4.1.0-GCC-10.2.0

mpicc -O2 mpi_improved.c -o mpi_improved -lm

IMAGE="../images/living_room.png"
MAX_ITERS=100
MEASURE_PSNR=0
INIT_STRATEGY=1
FUSION=1
EARLY_STOPPAGE=1

echo "[Image = $IMAGE] Running test case for different K. [FULL OPTIMIZATION]"
for K in 16 32 64 128
do
    echo "K = $K"
    for i in {1..1}
    do
        echo "Iteration: $i"
        srun --job-name=mpi_improved --time=10:00:00 --reservation=fri --nodes=2 --ntasks=64 --mpi=pmix mpi_improved $IMAGE $INIT_STRATEGY $FUSION $EARLY_STOPPAGE $MEASURE_PSNR $K $MAX_ITERS
    done
done

echo "[Image = $IMAGE] Running test case for same K. [FULL OPTIMIZATION]"
for N_TASKS in 2 4 8 16 32 64
do
    echo "N_TASKS = $N_TASKS"
    for i in {1..1}
    do
        echo "Iteration: $i"
        srun --job-name=mpi_improved --time=10:00:00 --reservation=fri --nodes=2 --mpi=pmix --ntasks=$N_TASKS mpi_improved $IMAGE $INIT_STRATEGY $FUSION $EARLY_STOPPAGE $MEASURE_PSNR 32 $MAX_ITERS
    done
done