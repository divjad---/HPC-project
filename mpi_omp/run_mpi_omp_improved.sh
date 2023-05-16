module load OpenMPI/4.1.0-GCC-10.2.0
mpicc -O2 mpi_omp_improved.c -o mpi_omp_improved -lm --openmp
srun --reservation=fri --mpi=pmix --cpus-per-task=32 --ntasks=2 --nodes=2 ./mpi_omp_improved ../images/mercedes_oldtimer.png 0 1 1 0 64 64