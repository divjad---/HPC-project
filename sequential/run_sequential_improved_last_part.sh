#!/bin/sh
#SBATCH --job-name=sequential_improved
#SBATCH --time=24:00:00
#SBATCH --output=sequential_improved_last_part.txt
#SBATCH --reservation=fri

# path init_strategy fusion early_stoppage measurePSNR K MAX_ITER

gcc -O2 sequential_improved.c --openmp -lm -o sequential_improved

IMAGE="../images/living_room.png"
INIT_STRATEGY=0
FUSION=1
EARLY_STOPPAGE=0
MEASURE_PSNR=0
MAX_ITERS=100

# echo "[Image = $IMAGE] Running test case for different K. [FUSION ONLY]"
# for K in 8 16 32 64 128
# do
#     echo "K = $K"
#     for i in {1..3}
#     do
#         echo "Iteration: $i"
#         srun sequential_improved $IMAGE $INIT_STRATEGY $FUSION $EARLY_STOPPAGE $MEASURE_PSNR $K $MAX_ITERS
#     done
# done

INIT_STRATEGY=1
FUSION=1
EARLY_STOPPAGE=1

echo "[Image = $IMAGE] Running test case for different K. [FULL OPTIMIZATION]"
for K in 32 64 128
do
    echo "K = $K"
    for i in 1
    do
        echo "Iteration: $i"
        srun sequential_improved $IMAGE $INIT_STRATEGY $FUSION $EARLY_STOPPAGE $MEASURE_PSNR $K $MAX_ITERS
    done
done

echo "[Image = $IMAGE] Running test case for same K."
echo "Nothing to do here. (sequential algorithm)"