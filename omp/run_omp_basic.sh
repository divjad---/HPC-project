#!/bin/sh
#SBATCH --job-name=omp_basic
#SBATCH --time=00:5:00
#SBATCH --output=omp_basic.txt
#SBATCH --reservation=fri
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=32

export OMP_PLACES=cores
export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=32

gcc -O2 omp_basic.c --openmp -lm -o omp_basic
srun omp_basic ../images/street.png 64 64