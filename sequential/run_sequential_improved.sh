#!/bin/sh
#SBATCH --job-name=sequential_improved
#SBATCH --time=00:30:00
#SBATCH --output=sequential_improved.txt
#SBATCH --reservation=fri

# path init_strategy fusion early_stoppage measurePSNR K MAX_ITER

gcc -O2 sequential_improved.c --openmp -lm -o sequential_improved

srun --reservation=fri sequential_improved ../images/lili.png 0 0 0 1 16 16
srun --reservation=fri sequential_improved ../images/lili.png 1 0 0 1 16 16
srun --reservation=fri sequential_improved ../images/lili.png 0 0 1 1 16 16
srun --reservation=fri sequential_improved ../images/lili.png 1 1 1 1 16 16

srun --reservation=fri sequential_improved ../images/cvetko.png 0 0 0 1 16 16
srun --reservation=fri sequential_improved ../images/cvetko.png 0 1 0 1 16 16
srun --reservation=fri sequential_improved ../images/cvetko.png 1 0 1 1 16 16
srun --reservation=fri sequential_improved ../images/cvetko.png 1 1 1 1 16 16

srun --reservation=fri sequential_improved ../images/lili3.png 0 0 1 1 16 16
srun --reservation=fri sequential_improved ../images/lili3.png 1 1 1 1 16 16
srun --reservation=fri sequential_improved ../images/lili3.png 1 1 1 1 16 16
srun --reservation=fri sequential_improved ../images/lili3.png 1 1 1 1 16 16

