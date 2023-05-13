#!/bin/sh
#SBATCH --job-name=sequential
#SBATCH --time=00:5:00
#SBATCH --output=sequential.txt
#SBATCH --reservation=fri

gcc -O2 kmeans_image_compression_sequential_BASIC.c --openmp -lm -o sequential_basic
srun sequential_basic ../Images/penguin.png