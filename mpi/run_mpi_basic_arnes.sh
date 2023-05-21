module load mpi/openmpi-4.1.3
# export UCX_TLS=tcp
srun --reservation=fri-vr --partition=gpu mpicc -O2 -lm mpi_basic.c -o mpi_basic
srun --reservation=fri-vr --partition=gpu --mpi=pmix --ntasks=2 --nodes=2 ./mpi_basic ../images/street.png 32 64