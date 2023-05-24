#!/bin/sh
module load OpenMPI/4.1.0-GCC-10.2.0

mpicc -O2 mpi_improved.c -o mpi_improved -lm

IMAGE="../images/living_room.png"
INIT_STRATEGY=0
FUSION=0
EARLY_STOPPAGE=0
MEASURE_PSNR=1
MAX_ITERS=100

echo "[Image = $IMAGE] Running test case for different K."
for K in 8 16 32 64 128
do
    echo "K = $K"
    for i in {1..3}
    do
        echo "Iteration: $i"
        srun --job-name=mpi_basic --time=10:00:00 --reservation=fri --nodes=2 --ntasks=64 --mpi=pmix mpi_improved $IMAGE $INIT_STRATEGY $FUSION $EARLY_STOPPAGE $MEASURE_PSNR $K $MAX_ITERS
    done
done

echo "[Image = $IMAGE] Running test case for same K."
for N_TASKS in 2 4 8 16 32 64
do
    echo "N_TASKS = $N_TASKS"
    for i in {1..3}
    do
        echo "Iteration: $i"
        srun --ntasks=$N_TASKS --mpi=pmix mpi_improved $IMAGE $INIT_STRATEGY $FUSION $EARLY_STOPPAGE $MEASURE_PSNR 32 $MAX_ITERS
    done
done