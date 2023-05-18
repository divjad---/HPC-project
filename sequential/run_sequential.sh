#!/bin/sh
#SBATCH --job-name=sequential
#SBATCH --time=01:15:00
#SBATCH --output=sequential.txt
#SBATCH --reservation=fri

gcc -O2 sequential_basic.c --openmp -lm -o sequential_basic
srun sequential_basic ../images/penguin.png 64 64
