#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../libs/helper_cuda.h"
#include <mpi.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../libs/stb_image.h"
#include "../libs/stb_image_write.h"

// IMPROVEMENTS
// Cluster initalization with Kmeans++
// Fusion to reduce the number of sweeps through memory and improve performance
// Early stoppage : stop in convergence

// Default values if not set with arguments
int K = 32;
int MAX_ITER = 20;
int BLOCK_SIZE = 256;
float EARLY_STOPPAGE_THRESHOLD = 1.0;


__global__ void calculateMSEKernel(unsigned char *original_image, unsigned char *compressed_image, float *mse, int width, int height, int cpp) {
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float temp[];

    temp[tid] = 0.0;

    if (global_tid < width * height * cpp) {
        int diff = (int)original_image[global_tid] - (int)compressed_image[global_tid];
        temp[tid] = (float)(diff * diff);
    }

    __syncthreads();

    // Parallel reduction - block level
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tid < i) {
            temp[tid] += temp[tid + i];
        }
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(mse, temp[0]);

}

double calculatePSNR(unsigned char *original_image, unsigned char *compressed_image, int width, int height, int cpp, size_t grid_size){
    float h_mse;
    float *d_mse;
    checkCudaErrors(cudaMalloc(&d_mse, sizeof(float)));
    checkCudaErrors(cudaMemset(d_mse, 0.0, sizeof(float)));

    int sharedArraySize = BLOCK_SIZE * sizeof(float);
    calculateMSEKernel<<<grid_size, BLOCK_SIZE, sharedArraySize>>>(original_image, compressed_image, d_mse, width, height, cpp);
    cudaDeviceSynchronize();
    getLastCudaError("Error while calculating PSNR score\n");

    cudaMemcpy(&h_mse, d_mse, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_mse);

    h_mse /= (float)(width * height * cpp);

    if (h_mse == 0)
        return INFINITY;

    return (10 * log10((255.0 * 255.0) / h_mse));
}

__device__ void atomicMaxf(float* address, int* address2, float val, int val2)
{
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(val));
        if (old == assumed)
            *address2 = val2;
    }
}

__global__ void computeDistancesAndFindMaxKernel(unsigned char *imageIn, float *centroids, float *distances, int *max_index, float *max_value, int width, int height, int cpp, int K) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / width;
    int j = tid % width;

    extern __shared__ char sharedMem[];
    float* temp_value = (float*)sharedMem;
    int* temp_index = (int*)&temp_value[blockDim.x];
    temp_value[threadIdx.x] = -1;
    temp_index[threadIdx.x] = -1;

    if (tid < width*height) {
        int pixel_index = i * width + j;
        float min_distance = 1e5;

        for (int c = 0; c < K; c++) {
            float distance = 0;
            for (int channel = 0; channel < cpp; channel++) {
                float temp = centroids[c * cpp + channel];
                float diff = temp - (float)imageIn[pixel_index * cpp + channel];
                distance += diff * diff;
            }
            if (distance < min_distance) {
                min_distance = distance;
            }
        }
        distances[pixel_index] = min_distance;
        temp_value[threadIdx.x] = min_distance;
        temp_index[threadIdx.x] = tid;
    }
    __syncthreads();

    // Parallel reduction - block level with sequential addressing
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            if (temp_value[threadIdx.x + i] > temp_value[threadIdx.x]) {
                temp_value[threadIdx.x] = temp_value[threadIdx.x + i];
                temp_index[threadIdx.x] = temp_index[threadIdx.x + i];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        if (*max_value < temp_value[0]) // This reduces the number of atomic operations
            atomicMaxf(max_value, max_index, temp_value[0], temp_index[0]);
    }
}


void init_clusters_kmeans_plus_plus(unsigned char *h_image, unsigned char * d_image, float *h_centroids, float *d_centroids, int width, int height, int cpp) {

    // Choose a random pixel for initial centroid
    int num_pixels = width * height;
    int random_pixel = rand() % num_pixels;
    for (int i = 0; i < cpp; i++) {
        h_centroids[i] = (float) h_image[random_pixel * cpp + i];
    }

    float *d_distances;
    checkCudaErrors(cudaMalloc(&d_distances, num_pixels * sizeof(float)));
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (num_pixels + threadsPerBlock - 1) / threadsPerBlock;

    int *d_max_index;
    float *d_max_value;
    checkCudaErrors(cudaMalloc(&d_max_index, sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_max_value, sizeof(float)));

    for (int k = 1; k < K; k++) {

        // Set initial values for max_index and max_value
        int initial_index = 0;
        float initial_value = -1;
        checkCudaErrors(cudaMemcpy(d_max_index, &initial_index, sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_max_value, &initial_value, sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_centroids, h_centroids, K * cpp * sizeof(float), cudaMemcpyHostToDevice));

        // Compute the distance to the nearest centroid for each data point and find the max distance and its index
        computeDistancesAndFindMaxKernel<<<blocksPerGrid, threadsPerBlock, (2 * BLOCK_SIZE * sizeof(float))>>>(d_image, d_centroids, d_distances, d_max_index, d_max_value, width, height, cpp, k);
        cudaDeviceSynchronize();
        getLastCudaError("Error while computing distances and finding max distance - kmeans++");

        // Copy max_index from device to host
        int farthest_pixel;
        cudaMemcpy(&farthest_pixel, d_max_index, sizeof(int), cudaMemcpyDeviceToHost);

        // Choose the next centroid to be the farthest data point
        for (int i = 0; i < cpp; i++) {
            h_centroids[k * cpp + i] = (float) h_image[farthest_pixel * cpp + i];
        }
    }
}

// FUSED both steps - fastest. It is again better to update pixel_to_centroid_indices every iteration, but this solution looks better:)
__global__ void assignPixelsToNearestCentroidsAndSumCentroidPositions(unsigned char *imageIn, float *centroids,  int *centroids_sums, int* elements_per_clusters, int width, int height, int cpp, int K) {

    extern __shared__ int sdata[]; // Shared memory for partial sums
    int *sdata_elements = (int*)&sdata[K * cpp]; // Shared memory for number of elements per cluster

    int tid = blockIdx.x * blockDim.x+ threadIdx.x;
    int i = tid / width;
    int j = tid % width;

    // Initialize shared memory
    for (int idx = threadIdx.x; idx < K * cpp; idx += blockDim.x) {
        sdata[idx] = 0;
    }

    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) {
        sdata_elements[idx] = 0;
    }
    __syncthreads();

    // Find nearest centroid for each pixel
    if ( tid < width*height){
        int index = i * width + j;
        int min_cluster_index = 0;
        float min_distance = 1e5;

        for (int cluster = 0; cluster < K; cluster++) {
            float curr_distance = 0;

            for (int channel = 0; channel < cpp; channel++) {
                float diff = ((float)imageIn[index * cpp + channel] - centroids[cluster * cpp + channel]);
                curr_distance += diff * diff;
            }

            if (curr_distance < min_distance) {
                min_cluster_index = cluster;
                min_distance = curr_distance;
            }
        }

        for (int channel = 0; channel < cpp; channel++) {
            atomicAdd(&sdata[min_cluster_index * cpp + channel], imageIn[index * cpp + channel]);
        }
        atomicAdd(&sdata_elements[min_cluster_index], 1);
    }
     __syncthreads();

    // Update clusters and counts in global memory
    for (int idx = threadIdx.x; idx < K * cpp; idx += blockDim.x) {
        atomicAdd(&centroids_sums[idx], sdata[idx]);
    }

    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) {
        atomicAdd(&elements_per_clusters[idx], sdata_elements[idx]);
    }
}

__global__ void assignPixelsToNearestCentroids(unsigned char *imageIn, int *pixel_cluster_indices, float *centroids, int width, int height, int cpp, int K) {

    int tid = blockIdx.x * blockDim.x+ threadIdx.x;
    int i = tid / width;
    int j = tid % width;

    // Find nearest centroid for each pixel
    if (tid < width*height){
        int index = (i * width + j) * cpp;
        int min_cluster_index = 0;
        float min_distance = 1e5;

        for (int cluster = 0; cluster < K; cluster++) {
            float curr_distance = 0;

            for (int channel = 0; channel < cpp; channel++) {
                float diff = ((float)imageIn[index + channel] - centroids[cluster * cpp + channel]);
                curr_distance += diff * diff;
            }

            if (curr_distance < min_distance) {
                min_cluster_index = cluster;
                min_distance = curr_distance;
            }
        }
        pixel_cluster_indices[i * width + j] = min_cluster_index;
    }
}

__global__ void sumCentroidPositions(unsigned char *imageIn, int *pixel_cluster_indices, int *centroids_sums, int* elements_per_clusters, int width, int height, int cpp) {

    int tid = blockIdx.x * blockDim.x+ threadIdx.x;
    int i = tid / width;
    int j = tid % width;

    // Over each pixel
    if ( i < height && j < width){
        int index = i * width + j;
        int cluster = pixel_cluster_indices[index];

        for (int channel = 0; channel < cpp; channel++) {
            atomicAdd(&centroids_sums[cluster * cpp + channel], (float)imageIn[index * cpp + channel]);
        }

        atomicAdd(&elements_per_clusters[cluster], 1);
    }
}

// SHARED ATOMICS-> this works if K*cpp is less than block size - Kernel below is more general.
__global__ void sumCentroidPositionsSharedMemory(unsigned char *imageIn, int *pixel_cluster_indices, int *centroids_sums, int* elements_per_clusters, int width, int height, int cpp, int K) {

    extern __shared__ int sdata[]; // Shared memory for partial sums
    int *sdata_elements = (int*)&sdata[K * cpp]; // Shared memory for number of elements per cluster
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / width;
    int j = tid % width;

    // Initialize shared memory
    if (threadIdx.x < K * cpp) {
        sdata[threadIdx.x] = 0;
    }

    if (threadIdx.x < K) {
        sdata_elements[threadIdx.x] = 0;
    }
    __syncthreads();

    // Iterate over each pixel
    if (i < height && j < width) {
        int index = i * width + j;
        int cluster = pixel_cluster_indices[index];

        for (int channel = 0; channel < cpp; channel++) {
            atomicAdd(&sdata[cluster * cpp + channel], imageIn[index * cpp + channel]);
        }
        atomicAdd(&sdata_elements[cluster], 1);

    }
    __syncthreads();

    // Update clusters and counts in global memory
    if (threadIdx.x < K * cpp) {
        atomicAdd(&centroids_sums[threadIdx.x], sdata[threadIdx.x]);
    }
    if (threadIdx.x < K) {
        atomicAdd(&elements_per_clusters[threadIdx.x], sdata_elements[threadIdx.x]);
    }
}

//GENERAL SOLUTION FOR SHARED ATOMICS - WORKS EVERYTIME: K * cpp can be > block size
__global__ void sumCentroidPositionsSharedMemoryWOConstraints(unsigned char *imageIn, int *pixel_cluster_indices, int *centroids_sums, int* elements_per_clusters, int width, int height, int cpp, int K) {
    extern __shared__ int sdata[]; // Shared memory for partial sums
    int *sdata_elements = (int*)&sdata[K * cpp]; // Shared memory for number of elements per cluster
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / width;
    int j = tid % width;

    // Initialize shared memory
    for (int idx = threadIdx.x; idx < K * cpp; idx += blockDim.x) {
        sdata[idx] = 0;
    }

    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) {
        sdata_elements[idx] = 0;
    }
    __syncthreads();

    // Iterate over each pixel
    if (tid < width*height) {
        int index = i * width + j;
        int cluster = pixel_cluster_indices[index];

        if (threadIdx.x < blockDim.x / 2) { // So order of atomics is not the same - just a bit faster
            // Iterate from the beginning
            for (int channel = 0; channel < cpp; channel++) {
                atomicAdd(&sdata[cluster * cpp + channel], imageIn[index * cpp + channel]);
            }
            atomicAdd(&sdata_elements[cluster], 1);
        }else {
            // Iterate from the end
            atomicAdd(&sdata_elements[cluster], 1);
            for (int channel = cpp - 1; channel >= 0; channel--) {
                atomicAdd(&sdata[cluster * cpp + channel], imageIn[index * cpp + channel]);
            }
        }
    }

    __syncthreads();

    // Update clusters and counts in global memory
    for (int idx = threadIdx.x; idx < K * cpp; idx += blockDim.x) {
        atomicAdd(&centroids_sums[idx], sdata[idx]);
    }

    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) {
        atomicAdd(&elements_per_clusters[idx], sdata_elements[idx]);
    }
}

__device__ int getRandomInteger(int lower, int upper, unsigned int seed) {
    curandState state;
    curand_init(seed, 0, 0, &state);

    float randomValue = curand_uniform(&state);
    return static_cast<int>(randomValue * (upper - lower + 1)) + lower;
}

__global__ void updateCentroidPositions(unsigned char *imageIn, float *centroids, int* centroids_sums, int* elements_per_clusters, int width, int height, int cpp, int K) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Update each centroid position by calculating the average channel value
    if (tid < K * cpp) {
        int cluster = tid / cpp;
        int channel = tid % cpp;

        if (elements_per_clusters[cluster] > 0) {
            centroids[tid] = ((float)centroids_sums[tid]) / elements_per_clusters[cluster];
        } else {
            // Assign random pixel to empty centroid
            unsigned int seed = cluster;
            int random_pixel_i = getRandomInteger(0, width * height - 1, seed);
            centroids[tid] = imageIn[random_pixel_i * cpp + channel];
        }

        centroids_sums[tid] = 0;
        if(channel == 0)
            elements_per_clusters[cluster] = 0;
    }
}

__global__ void mapPixelsToCentroidValues(unsigned char *imageIn, int *pixel_cluster_indices, float *centroids, int width, int height, int cpp, int K) {

    int tid = blockIdx.x * blockDim.x+ threadIdx.x;
    int i = tid / width;
    int j = tid % width;

    // Iterate over each pixel
    if ( i < height && j < width){
        int index = i * width + j;
        int cluster = pixel_cluster_indices[index];

        for (int channel = 0; channel < cpp; channel++) {
            imageIn[index * cpp + channel] = (unsigned char) centroids[cluster * cpp + channel];
        }
    }
}

void init_clusters_random(unsigned char *imageIn, float *centroids, int width, int height, int cpp) {
    int index;
    int num_pixels = width * height;
    for (int i = 0; i < K; i++) {
        index = rand() % num_pixels;
        for(int j = 0; j < cpp; j++){
            centroids[i * cpp + j] = (float) (imageIn[index * cpp + j]);
        }
    }
}

// Empirically tested for the image of this size (45 MB). We could implement dynamic function to set threshold
// We set the threshold so strict, so the PSNR(compression quality) with KMEANS++ and Early Stop is always greater than the PSNR of basic algorithm
float get_early_stoppage_threshold(int K){
    if (K < 16)
        return 0.1;
    if (K < 32)
        return 0.3;
    if (K < 48)
        return 0.7;
    if (K < 64)
        return 1.2;
    if (K < 128)
        return 3;
    return 4.0;
}

int main(int argc, char **argv)
{

    /* Initialize MPI */
    MPI_Init(&argc, &argv);

    /* Get rank */
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Get number of processes */
    int num_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    /* Read the arguments */
    char *image_file;
    if (argc > 1)
    {
        image_file = argv[1];
    }
    else
    {
        if (rank == 0)
        {
            fprintf(stderr, "Not enough arguments\n");
            fprintf(stderr, "Usage: %s <IMAGE_PATH>\n", argv[0]);
        }
        MPI_Finalize();
        exit(1);
    }

    int init_strategy = 0; // 0-random, 1-kmeans++
    int fusion = 0;        // 0 for standard k-means, 1 for optimized k-means with fused operations
    int measurePSNR = 0;
    int early_stopage = 0;

    if (argc > 2) BLOCK_SIZE = atoi(argv[2]);
    if (argc > 3)
        init_strategy = atoi(argv[3]);
    if (argc > 4)
        fusion = atoi(argv[4]);
    if (argc > 5)
        early_stopage = atoi(argv[5]);
    if (argc > 6)
        measurePSNR = atoi(argv[6]);
    if (argc > 7) K = atoi(argv[7]);
    if (argc > 8) MAX_ITER = atoi(argv[8]);

    EARLY_STOPPAGE_THRESHOLD = get_early_stoppage_threshold(K);

    /* Read the image */
    int width, height, cpp;
    unsigned char *input_image = stbi_load(image_file, &width, &height, &cpp, 0);

    if(!input_image) return 0;

    /* Synchronize the processes */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Set random state */
    srand(42);

    /* Begin measuring time */
    double start = MPI_Wtime();

    /* K-Means Image Compression */
    {
        /* Split the image */
        int my_image_height;
        if (rank == num_processes - 1)
        {
            // If height not divisible by number of processes then last process gets more height then others
            my_image_height = height / num_processes + height % num_processes;
        }
        else
        {
            my_image_height = height / num_processes;
        }
        unsigned char *my_image = (unsigned char *)malloc(my_image_height * width * cpp * sizeof(unsigned char));
        int *counts_send = (int *)malloc(num_processes * sizeof(int));
        int *displacements = (int *)malloc(num_processes * sizeof(int));
        if (rank == 0)
        {
            for (int i = 0; i < num_processes; i++)
            {
                int process_image_height;
                if (i == num_processes - 1)
                {
                    process_image_height = height / num_processes + height % num_processes;
                }
                else
                {
                    process_image_height = height / num_processes;
                }
                counts_send[i] = process_image_height * width * cpp;
                if (i == 0)
                {
                    displacements[i] = 0;
                }
                else
                {
                    displacements[i] = displacements[i - 1] + counts_send[i - 1];
                }
            }
        }
        MPI_Scatterv(input_image, counts_send, displacements, MPI_UNSIGNED_CHAR, my_image, my_image_height * width * cpp, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

        // Set block and grid sizes
        const size_t blockSize = BLOCK_SIZE;
        const size_t gridSize = (width * my_image_height + blockSize - 1) / blockSize;

        // Copy data to GPU
        unsigned char *d_image;
        checkCudaErrors(cudaMalloc(&d_image, width * my_image_height * cpp * sizeof(unsigned char)));
        checkCudaErrors(cudaMemcpy(d_image, my_image, width * my_image_height * cpp * sizeof(unsigned char), cudaMemcpyHostToDevice));

        float *d_centroids;
        checkCudaErrors(cudaMalloc(&d_centroids, K * cpp * sizeof(float)));

        int *d_centroids_sums;
        checkCudaErrors(cudaMalloc(&d_centroids_sums, K * cpp * sizeof(int)));
        checkCudaErrors(cudaMemset(d_centroids_sums, 0,  K * cpp * sizeof(int)));

        int *d_pixel_cluster_indices;
        checkCudaErrors(cudaMalloc(&d_pixel_cluster_indices, width * my_image_height * sizeof(int)));

        int *d_elements_per_cluster;
        checkCudaErrors(cudaMalloc(&d_elements_per_cluster, K * sizeof(int)));
        checkCudaErrors(cudaMemset(d_elements_per_cluster, 0, K  * sizeof(int)));

        // Intialize clusters
        float *h_centroids = (float *) calloc(cpp * K, sizeof(float));
        if (init_strategy == 0){
            init_clusters_random(my_image, h_centroids, width, my_image_height, cpp);
        }else{
            init_clusters_kmeans_plus_plus(my_image, d_image, h_centroids, d_centroids, width, my_image_height, cpp);
        }
        MPI_Allreduce(MPI_IN_PLACE, h_centroids, K * cpp, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        for(int i = 0; i < K * cpp; i++) {
            h_centroids[i] /= num_processes;
        }
        checkCudaErrors(cudaMemcpy(d_centroids, h_centroids, K * cpp * sizeof(float), cudaMemcpyHostToDevice));


        int shared_memory_size = (K * cpp + K) * sizeof(int);
        float *previous_centroids;
        if (early_stopage == 1)
            previous_centroids = (float *)calloc(cpp * K, sizeof(float));

        // Main loop
        if (rank == 0) {
            printf("Iteration times: [");
        }
        for (int iteration = 0; iteration < MAX_ITER; iteration++) {
            double iteration_start = MPI_Wtime();
            if (fusion == 0){
                assignPixelsToNearestCentroids<<<gridSize, blockSize>>>(d_image, d_pixel_cluster_indices, d_centroids, width, my_image_height, cpp, K);
                getLastCudaError("Error while assigning pixels to nearest centroids\n");

                sumCentroidPositionsSharedMemoryWOConstraints<<<gridSize, blockSize, shared_memory_size>>>(d_image, d_pixel_cluster_indices, d_centroids_sums, d_elements_per_cluster, width, my_image_height, cpp, K);
                getLastCudaError("Error while summation of centroid vales\n");
            }else{

                assignPixelsToNearestCentroidsAndSumCentroidPositions<<<gridSize, blockSize, shared_memory_size>>>(d_image, d_centroids, d_centroids_sums, d_elements_per_cluster, width, my_image_height, cpp, K);
                getLastCudaError("Error in fused part of the algorithm \n");
            }

            updateCentroidPositions<<<((K * cpp + BLOCK_SIZE -1)/BLOCK_SIZE), BLOCK_SIZE>>>(d_image, d_centroids, d_centroids_sums, d_elements_per_cluster, width, my_image_height, cpp, K);
            cudaDeviceSynchronize();
            getLastCudaError("Error while updating positions of centroids\n");

            checkCudaErrors(cudaMemcpy(h_centroids, d_centroids, K * cpp * sizeof(float), cudaMemcpyDeviceToHost));
            MPI_Allreduce(MPI_IN_PLACE, h_centroids, cpp * K, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            for(int i = 0; i < cpp * K; i++) {
                h_centroids[i] /= num_processes;
            }
            checkCudaErrors(cudaMemcpy(d_centroids, h_centroids, K * cpp * sizeof(float), cudaMemcpyHostToDevice));
            // Check for early stoppage
            if (early_stopage == 1)
            {
                checkCudaErrors(cudaMemcpy(h_centroids, d_centroids, K * cpp * sizeof(float), cudaMemcpyDeviceToHost));
                float max_change = 0.0;
                for (int i = 0; i < K * cpp; i++)
                {
                    float change = fabs(h_centroids[i] - previous_centroids[i]);
                    if (change > max_change)
                    {
                        max_change = change;
                    }
                }
                if (max_change <= EARLY_STOPPAGE_THRESHOLD)
                {
                    printf("EARLY STOPPAGE %f", max_change);
                    break;
                }
                memcpy(previous_centroids, h_centroids, K * cpp * sizeof(float));
            }
            if (rank == 0) {
                if (iteration > 0) {
                    printf(", ");
                }
                double iteration_end = MPI_Wtime();
                printf("%f", iteration_end - iteration_start);
            }
        }
        if (rank == 0) {
            printf("]\n");
        }
        unsigned char *d_compressed_image;
        checkCudaErrors(cudaMalloc(&d_compressed_image, width * my_image_height * cpp * sizeof(unsigned char)));

        if(fusion == 1) // Get final cluster for each pixel
            assignPixelsToNearestCentroids<<<gridSize, blockSize>>>(d_image, d_pixel_cluster_indices, d_centroids, width, my_image_height, cpp, K);

        // Map pixel values to the values of centroids
        mapPixelsToCentroidValues<<<gridSize, blockSize>>>(d_compressed_image, d_pixel_cluster_indices, d_centroids, width, my_image_height, cpp, K);
        getLastCudaError("Error while updating positions of centroids\n");

        checkCudaErrors(cudaMemcpy(my_image, d_compressed_image, width * my_image_height * cpp * sizeof(unsigned char), cudaMemcpyDeviceToHost));

        /* Gather the image from all the processes*/
        MPI_Gatherv(my_image, my_image_height * width * cpp, MPI_UNSIGNED_CHAR, input_image, counts_send, displacements, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

        double end = MPI_Wtime();
        // Save the compressed image
        if (rank == 0) {
            printf("Execution time: %.4f \n", end - start);
        }

        if (measurePSNR == 1){
            double psnr = calculatePSNR(d_image, d_compressed_image, width, my_image_height, cpp, gridSize);
            double my_psnr = 0.0;
            MPI_Reduce(&psnr, &my_psnr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            printf("PSNR: %lf\n", psnr);
        }
        if (rank == 0) {
            char output_file[256];
            strcpy(output_file, image_file);
            char *extension = strrchr(output_file, '.');
            if (extension != NULL) *extension = '\0';  // Cut off the file extension
            strcat(output_file, "_compressed_mpi_cuda_improved.png");
            stbi_write_png(output_file, width, height, cpp, input_image, width * cpp);
        }

        cudaFree(d_image);
        cudaFree(d_centroids);
        cudaFree(d_centroids_sums);
        cudaFree(d_pixel_cluster_indices);
        cudaFree(d_elements_per_cluster);
    }

    stbi_image_free(input_image);

    /* Finalize MPI */
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}