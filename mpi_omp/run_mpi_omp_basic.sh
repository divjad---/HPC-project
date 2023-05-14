module load OpenMPI/4.1.0-GCC-10.2.0
mpicc mpi_omp_basic.c -o mpi_omp_basic -lm --openmp
srun --reservation=fri --mpi=pmix --cpus-per-task=32 --ntasks=2 --nodes=2 ./mpi_omp_basic ../images/street.png 32 64