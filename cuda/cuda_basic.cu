#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../libs/helper_cuda.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../libs/stb_image.h"
#include "../libs/stb_image_write.h"

// Default values if not set with arguments
int K = 32;
int MAX_ITER = 20;
int BLOCK_SIZE = 256;

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

__global__ void assignPixelsToNearestCentroids(unsigned char *imageIn, int *pixel_cluster_indices, float *centroids, int width, int height, int cpp, int K) {

    int tid = blockIdx.x * blockDim.x+ threadIdx.x;
    int i = tid / width;
    int j = tid % width;

    // Find nearest centroid for each pixel
    if ( i < height && j < width){
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

// SLOWER than atomics in shared memory
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

// SHARED ATOMICS-> this works if K*cpp is less than block size - method below is general, that works in every scenario
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

//GENERAL SOLUTION - WORKS IN EVERY SCENARIO: K * cpp can be > block size
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

void kmeans_image_compression(unsigned char *h_image, int width, int height, int cpp, char *image_file) {

    // Create CUDA events and start recording
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaEvent_t iteration_start, iteration_stop;
    cudaEventCreate(&iteration_start);
    cudaEventCreate(&iteration_stop);

    // Set block and grid sizes
    const size_t blockSize = BLOCK_SIZE;
    const size_t gridSize = (width * height + blockSize - 1) / blockSize;

    // Intialize clusters
    float *h_centroids = (float *) calloc(cpp * K, sizeof(float));
    init_clusters_random(h_image, h_centroids, width, height, cpp);

    // Copy data to GPU
    unsigned char *d_image;
    float *d_centroids;
    int *d_centroids_sums;
    int *d_pixel_cluster_indices;
    int *d_elements_per_cluster;

    checkCudaErrors(cudaMalloc(&d_image, width * height * cpp * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&d_centroids, K * cpp * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_centroids_sums, K * cpp * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_pixel_cluster_indices, width * height * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_elements_per_cluster, K * sizeof(int)));

    checkCudaErrors(cudaMemcpy(d_image, h_image, width * height * cpp * sizeof(unsigned char), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_centroids, h_centroids, K * cpp * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_centroids_sums, 0,  K * cpp * sizeof(int)));
    checkCudaErrors(cudaMemset(d_elements_per_cluster, 0, K  * sizeof(int)));
    getLastCudaError("Error while copying data to GPU\n");

    int shared_memory_size = (K * cpp + K) * sizeof(int);

    // Main loop
    printf("Iteration times: [");
    for (int iteration = 0; iteration < MAX_ITER; iteration++) {
        cudaEventRecord(iteration_start);
        assignPixelsToNearestCentroids<<<gridSize, blockSize>>>(d_image, d_pixel_cluster_indices, d_centroids, width, height, cpp, K);
        getLastCudaError("Error while assigning pixels to nearest centroids\n");

        sumCentroidPositions<<<gridSize, blockSize, shared_memory_size>>>(d_image, d_pixel_cluster_indices, d_centroids_sums, d_elements_per_cluster, width, height, cpp);
        // sumCentroidPositionsSharedMemoryWOConstraints<<<gridSize, blockSize, shared_memory_size>>>(d_image, d_pixel_cluster_indices, d_centroids_sums, d_elements_per_cluster, width, height, cpp, K);
        getLastCudaError("Error while summation of centroid vales\n");

        updateCentroidPositions<<<((K * cpp + BLOCK_SIZE -1)/BLOCK_SIZE), BLOCK_SIZE>>>(d_image, d_centroids, d_centroids_sums, d_elements_per_cluster, width, height, cpp, K);
        getLastCudaError("Error while updating positions of centroids\n");
        cudaEventRecord(iteration_stop);
        cudaEventSynchronize(iteration_stop);
        if (iteration > 0) {
            printf(", ");
        }
        float milis = 0.0f;
        cudaEventElapsedTime(&milis, iteration_start, iteration_stop);
        printf("%f", milis);
    }
    printf("]\n");
    // Assign pixels to final clusters
    mapPixelsToCentroidValues<<<gridSize, blockSize>>>(d_image, d_pixel_cluster_indices, d_centroids, width, height, cpp, K);

    // Save the compreesed image
    checkCudaErrors(cudaMemcpy(h_image, d_image, width * height * cpp * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time: %.4f \n", milliseconds);

    char output_file[256];
    strcpy(output_file, image_file);
    char *extension = strrchr(output_file, '.');
    if (extension != NULL) *extension = '\0';  // Cut off the file extension
    strcat(output_file, "_compressedGPU.png");
    stbi_write_png(output_file, width, height, cpp, h_image, width * cpp);

    cudaFree(d_image);
    cudaFree(d_centroids);
    cudaFree(d_centroids_sums);
    cudaFree(d_pixel_cluster_indices);
    cudaFree(d_elements_per_cluster);
}

int main(int argc, char **argv)
{
    if (argc < 2){
        fprintf(stderr, "Not enough arguments\n");
        exit(1);
    }
    srand(42);
    char *image_file = argv[1];
    if (argc > 2) BLOCK_SIZE = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);
    if (argc > 4) MAX_ITER = atoi(argv[4]);

    int width, height, cpp;
    unsigned char *h_image = stbi_load(image_file, &width, &height, &cpp, 0);

    if(!h_image) return 0;

    kmeans_image_compression(h_image, width, height, cpp, image_file);

    stbi_image_free(h_image);
}