#!/bin/sh
module load OpenMPI/4.1.0-GCC-10.2.0

mpicc -O2 mpi_omp_improved.c -o mpi_omp_improved -lm --openmp

export OMP_PLACES=cores
export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=32

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
        srun --job-name=mpi_basic --time=10:00:00 --reservation=fri --nodes=2 --ntasks=2 --cpus-per-task=32 --mpi=pmix mpi_omp_improved $IMAGE $INIT_STRATEGY $FUSION $EARLY_STOPPAGE $MEASURE_PSNR $K $MAX_ITERS
    done
done

echo "[Image = $IMAGE] Running test case for same K."
echo "N_TASKS = 2, CPUS_PER_TASK = 32"
srun --job-name=mpi_basic --time=10:00:00 --reservation=fri --nodes=2 --mpi=pmix --ntasks=2 --cpus-per-task=32 mpi_omp_improved $IMAGE $INIT_STRATEGY $FUSION $EARLY_STOPPAGE $MEASURE_PSNR 32 $MAX_ITERS
echo "N_TASKS = 4, CPUS_PER_TASK = 16"
export OMP_NUM_THREADS=16
srun --job-name=mpi_basic --time=10:00:00 --reservation=fri --nodes=2 --mpi=pmix --ntasks=4 --cpus-per-task=16 mpi_omp_improved $IMAGE $INIT_STRATEGY $FUSION $EARLY_STOPPAGE $MEASURE_PSNR 32 $MAX_ITERS
echo "N_TASKS = 8, CPUS_PER_TASK = 8"
export OMP_NUM_THREADS=8
srun --job-name=mpi_basic --time=10:00:00 --reservation=fri --nodes=2 --mpi=pmix --ntasks=8 --cpus-per-task=8 mpi_omp_improved $IMAGE $INIT_STRATEGY $FUSION $EARLY_STOPPAGE $MEASURE_PSNR 32 $MAX_ITERS
echo "N_TASKS = 16, CPUS_PER_TASK = 4"
export OMP_NUM_THREADS=4
srun --job-name=mpi_basic --time=10:00:00 --reservation=fri --nodes=2 --mpi=pmix --ntasks=16 --cpus-per-task=4 mpi_omp_improved $IMAGE $INIT_STRATEGY $FUSION $EARLY_STOPPAGE $MEASURE_PSNR 32 $MAX_ITERS
echo "N_TASKS = 32, CPUS_PER_TASK = 2"
export OMP_NUM_THREADS=2
srun --job-name=mpi_basic --time=10:00:00 --reservation=fri --nodes=2 --mpi=pmix --ntasks=32 --cpus-per-task=2 mpi_omp_improved $IMAGE $INIT_STRATEGY $FUSION $EARLY_STOPPAGE $MEASURE_PSNR 32 $MAX_ITERS