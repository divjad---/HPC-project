#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{

    
    int i = 0;

    double start_time = omp_get_wtime();

    int a = 5;
    int b = 5;
    int c = 5;
    int d = a+b+c;
    while(i < 99999999){
        a = rand() % 5;
        b = rand() % 5;
        c = rand() % 5;
        d += a+b+c+rand() % 5;
        i+=1;
    }
    
    double elapsed_time = omp_get_wtime() - start_time;
    printf("Execution time: %.8f seconds\n", elapsed_time);

    i = 0; // Reset counter i
    start_time = omp_get_wtime();
    int dd=0;
    while(i < 99999999){
        int aa = rand() % 5;
        int bb = rand() % 5;
        int cc = rand() % 5;
        dd += aa+bb+cc+rand() % 5;
        i+=1;
    }
    elapsed_time = omp_get_wtime() - start_time;
    printf("Execution time: %.8f seconds\n", elapsed_time);

    return 0;
}