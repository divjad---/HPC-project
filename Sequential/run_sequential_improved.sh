#!/bin/sh
#SBATCH --job-name=sequential_improved
#SBATCH --time=00:5:00
#SBATCH --output=sequential_improved.txt
#SBATCH --reservation=fri

gcc -O2 kmeans_image_compression_sequential_improved.c --openmp -lm -o sequential_improved
srun sequential_improved ../Images/penguin.png 0 0
srun sequential_improved ../Images/penguin.png 0 1
srun sequential_improved ../Images/penguin.png 1 0
srun sequential_improved ../Images/penguin.png 1 1
