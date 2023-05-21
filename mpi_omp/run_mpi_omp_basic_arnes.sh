module load mpi/openmpi-4.1.3
srun --reservation=fri-vr --partition=gpu mpicc -O2 mpi_omp_basic.c -o mpi_omp_basic -lm --openmp
srun --reservation=fri-vr --partition=gpu --mpi=pmix --cpus-per-task=16 --ntasks=2 --nodes=2 ./mpi_omp_basic ../images/street.png 32 64