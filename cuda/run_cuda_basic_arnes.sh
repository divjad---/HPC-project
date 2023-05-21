module load CUDA
nvcc cuda_basic.cu -O2 -lm -o cuda_basic
srun --reservation=fri-vr --partition=gpu --gpus=1 cuda_basic ../images/penguin.png 128 32 100