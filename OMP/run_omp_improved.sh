#!/bin/sh
#SBATCH --job-name=omp_improved
#SBATCH --time=00:5:00
#SBATCH --output=omp_improved.txt
#SBATCH --reservation=fri
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=32

export OMP_PLACES=cores
export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=32

# path init_strategy fusion early_stoppage measurePSNR K MAX_ITER
gcc -O2 omp_improved.c --openmp -lm -o omp_improved
srun --reservation=fri omp_improved ../Images/lili.png 0 0 0 1 16 16
srun --reservation=fri omp_improved ../Images/lili.png 1 0 0 1 16 16