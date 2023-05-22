module load mpi/openmpi-4.1.3
# export UCX_TLS=tcp
srun --reservation=fri-vr --partition=gpu mpicc -O2 -lm mpi_improved.c -o mpi_improved
srun --reservation=fri-vr --partition=gpu --mpi=pmix --ntasks=2 --nodes=2 mpi_improved ../images/penguin.png 0 1 0 0 32 64