module load OpenMPI/4.1.0-GCC-10.2.0
mpicc mpi_basic.c -o mpi_basic -lm
srun --reservation=fri --mpi=pmix --ntasks=32 --nodes=2 ./mpi_basic ../Images/street.png