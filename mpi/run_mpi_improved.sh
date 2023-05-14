module load OpenMPI/4.1.0-GCC-10.2.0
mpicc mpi_improved.c -o mpi_improved -lm
srun --reservation=fri --mpi=pmix --ntasks=32 --nodes=2 ./mpi_improved ../images/street.png 0 1 1 0 32 64