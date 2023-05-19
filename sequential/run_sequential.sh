#!/bin/sh
#SBATCH --job-name=sequential
#SBATCH --time=00:30:00
#SBATCH --output=sequential.txt
#SBATCH --reservation=fri

gcc -O2 sequential_basic.c --openmp -lm -o sequential_basic
srun sequential_basic ../images/lili.png 32 32
srun sequential_basic ../images/lili.png 32 32

