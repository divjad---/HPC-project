module load CUDA
nvcc cuda_basic.cu -O2 -lm -o cuda_basic
srun --reservation=fri-vr --partition=gpu --gpus=1 cuda_basic ../images/obsolete/penguin.png 128 32 100

# module load CUDA/10.1.243-GCC-8.3.0
# nvcc cuda_improved.cu -O2 -lm -o cuda_improved
# srun --reservation=fri --time=00:01:00 -G1 -n1 cuda_improved ../images/living_room.png 128 128 100