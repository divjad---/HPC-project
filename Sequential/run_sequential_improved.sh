#!/bin/sh
#SBATCH --job-name=sequential_improved
#SBATCH --time=00:10:00
#SBATCH --output=sequential_improved.txt
#SBATCH --reservation=fri
# path init_strategy fusion early_stoppage measurePSNR

gcc -O2 kmeans_image_compression_sequential_improved.c --openmp -lm -o sequential_improved
srun --reservation=fri sequential_improved ../Images/cvetko.png 0 0 1 1
srun --reservation=fri sequential_improved ../Images/cvetko.png 0 1 1 1
srun --reservation=fri sequential_improved ../Images/cvetko.png 1 0 1 1
srun --reservation=fri sequential_improved ../Images/cvetko.png 1 1 1 1
