#!/bin/sh
#SBATCH --job-name=omp_basic
#SBATCH --time=00:5:00
#SBATCH --output=omp_basic_arnes.txt
#SBATCH --reservation=fri-vr
#SBATCH --partition=gpu
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=16

# export OMP_PLACES=cores
# export OMP_PROC_BIND=TRUE
# export OMP_NUM_THREADS=32

gcc -O2 -fopenmp -lm omp_basic.c -o omp_basic
srun --reservation=fri-vr --partition=gpu omp_basic ../images/street.png 32 64