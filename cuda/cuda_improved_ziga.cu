#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <omp.h>
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

double calculatePSNR(unsigned char *original_image, unsigned char *compressed_image, int width, int height, int cpp, size_t grid_size){
    float h_mse = 0.0;
    float *d_mse;
    cudaMalloc(&d_mse, sizeof(float));
    cudaMemcpy(d_mse, &h_mse, sizeof(float), cudaMemcpyHostToDevice);

    int num_pixels = width * height * cpp;

    calculateMSEKernel<<<grid_size, BLOCK_SIZE>>>(original_image, compressed_image, d_mse, width, height, cpp);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_mse, d_mse, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_mse);

    h_mse /= (float)(num_pixels);
    if (h_mse == 0) 
        return INFINITY; 

    return (10 * log10((255.0 * 255.0) / h_mse));
}

__global__ void calculateMSEKernel(unsigned char *original_image, unsigned char *compressed_image, float *mse, int width, int height, int cpp) {
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float temp[BLOCK_SIZE];
    temp[tid] = 0.0;

    if (global_tid < width * height * cpp) {
        int diff = (int)original_image[global_tid] - (int)compressed_image[global_tid];
        temp[tid] = (float)(diff * diff);
    }

    __syncthreads();

    // Parallel reduction on a block level
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tid < i) {
            temp[tid] += temp[tid + i];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(mse, temp[0]);
    }
}

__global__ void findMaxDistanceKernel(float *distances, int *max_index, float *max_value, int num_pixels) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float temp_value[BLOCK_SIZE];
    __shared__ int temp_index[BLOCK_SIZE];
    temp_value[threadIdx.x] = (tid < num_pixels) ? distances[tid] : -1;
    temp_index[threadIdx.x] = tid;

    __syncthreads();

    // Parallel reduction on a block level - sequential addressing
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
        // TODO kaj če bi samo ena nit v for zanki preverila ? global id 0 npr
        // TODO spet sync med bloki...
        // mogoce vrnemo array dolzine st blokov kjer vsak blok shrani najbolsega in nato for na hostu?
        // for  i to st blokov...
        // https://stackoverflow.com/questions/17371275/implementing-max-reduce-in-cuda
        // https://ernie55ernie.github.io/parallel%20programming/2018/03/17/cuda-maximum-value-with-parallel-reduction.html
        atomicMax(max_value, max_index, temp_value[0], temp_index[0]);
    }
}

__global__ void computeDistancesKernel(unsigned char *imageIn, float *centroids, float *distances, int width, int height, int cpp, int K) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / width;
    int j = tid % width;

    if (i < height && j < width) {
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
    }
}

void init_clusters_kmeans_plus_plus(unsigned char *imageIn, float *centroids, int width, int height, int cpp) {
    
    // Choose a random pixel for initial centroid
    int num_pixels = width * height;
    int random_pixel = rand() % num_pixels;
    for (int i = 0; i < cpp; i++) {
        centroids[i] = (float) imageIn[random_pixel * cpp + i];
    }

    float *distances;
    cudaMallocManaged(&distances, num_pixels * sizeof(float));
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_pixels + threadsPerBlock - 1) / threadsPerBlock;

    for (int k = 1; k < K; k++) {
        int farthest_pixel = 0;
        float max_distance = -1;

        // Compute the distance to the nearest centroid for each data point
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int pixel_index = i * width + j;
                float min_distance = 1e5;
                for (int c = 1; c < k; c++) {

                    // Calculate Euclidean distance
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

                // Check if this pixel is the farthest one yet -> not selecting proportionally
                if (min_distance > max_distance) {
                    max_distance = min_distance;
                    farthest_pixel = pixel_index;
                }
            }
        }
        
        // Choose the next centroid to be the farthest data point
        for (int i = 0; i < cpp; i++) {
            centroids[k * cpp + i] = (float) imageIn[farthest_pixel * cpp + i];
        }
    }    
    free(distances);
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

__global__ void sumCentroidPositions(unsigned char *imageIn, int *pixel_cluster_indices, float *centroids, int* elements_per_clusters, int width, int height, int cpp) {
    
    int tid = blockIdx.x * blockDim.x+ threadIdx.x;
    int i = tid / width;
    int j = tid % width;

    // Iterate over each pixel
    if ( i < height && j < width){
        int index = i * width + j;
        int cluster = pixel_cluster_indices[index];

        for (int channel = 0; channel < cpp; channel++) {
            atomicAdd(&centroids[cluster * cpp + channel], (float)imageIn[index * cpp + channel]);
        }

        atomicAdd(&elements_per_clusters[cluster], 1);
    }
}

// TRYING TO REDUCE NUMBER OF ATOMIC OPERATIONS -> this works if K*cpp is less than block size -> else we should use for loops
__global__ void sumCentroidPositionsSharedMemory(unsigned char *imageIn, int *pixel_cluster_indices, float *centroids, int* elements_per_clusters, int width, int height, int cpp, int K) {

    extern __shared__ float sdata[]; // Shared memory for partial sums
    int *sdata_elements = (int*)&sdata[K * cpp]; // Shared memory for number of elements per cluster
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / width;
    int j = tid % width;

    // Initialize shared memory
    if (threadIdx.x < K * cpp) {
        sdata[threadIdx.x] = 0.0;
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
            atomicAdd(&sdata[cluster * cpp + channel], (float)imageIn[index * cpp + channel]);
        }
        atomicAdd(&sdata_elements[cluster], 1); 
    }

    __syncthreads();

    // First K threads update a different cluster in global memory
    if (threadIdx.x < K) {
        for (int channel = 0; channel < cpp; channel++) {
            atomicAdd(&centroids[threadIdx.x * cpp + channel], sdata[threadIdx.x * cpp + channel]);
        }
        atomicAdd(&elements_per_clusters[threadIdx.x], sdata_elements[threadIdx.x]);
    }
}

__global__ void updateCentroidPositions(unsigned char *imageIn, float *centroids, int* elements_per_clusters, int width, int height, int cpp, int K) {
    
    int tid = blockIdx.x * blockDim.x+ threadIdx.x;

    // Update each centroid position by calculating the average channel value
    if (tid < K * cpp) {
        int cluster = tid / cpp;
        int channel = tid % cpp;

        if (elements_per_clusters[cluster] > 0) {
            centroids[tid] = centroids[tid] / elements_per_clusters[cluster];
        } else {
            // Assign random pixel to empty centroid
            // int random_pixel_i = rand() % (width * height); // TODO
            int random_pixel_i = 0;
            centroids[tid] = imageIn[random_pixel_i * cpp + channel];
        }
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

    // Set block and grid sizes
    const size_t blockSize = BLOCK_SIZE;
    const size_t gridSize = (width * height + blockSize - 1) / blockSize;

    // Intialize clusters
    float *h_centroids = (float *) calloc(cpp * K, sizeof(float));
    init_clusters_random(h_image, h_centroids, width, height, cpp);
    
    // Copy data to GPU
    unsigned char *d_image;
    float *d_centroids;
    int *d_pixel_cluster_indices;
    int *d_elements_per_cluster;

    checkCudaErrors(cudaMalloc(&d_image, width * height * cpp * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc(&d_centroids, K * cpp * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_pixel_cluster_indices, width * height * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_elements_per_cluster, K * sizeof(int)));

    checkCudaErrors(cudaMemcpy(d_image, h_image, width * height * cpp * sizeof(unsigned char), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_centroids, h_centroids, K * cpp * sizeof(float), cudaMemcpyHostToDevice));
    getLastCudaError("Error while copying data to GPU\n");

    // Main loop
    for (int iteration = 0; iteration < MAX_ITER; iteration++) {
        assignPixelsToNearestCentroids<<<gridSize, blockSize>>>(d_image, d_pixel_cluster_indices, d_centroids, width, height, cpp, K);
        getLastCudaError("Error while assigning pixels to nearest centroids\n");

        cudaMemset(d_centroids, 0, K * cpp * sizeof(float));
        cudaMemset(d_elements_per_cluster, 0, K * sizeof(int));

        int shared_memory_size = (K * cpp + K) * sizeof(float);
        sumCentroidPositionsSharedMemory<<<gridSize, blockSize, shared_memory_size>>>(d_image, d_pixel_cluster_indices, d_centroids, d_elements_per_cluster, width, height, cpp, K);
        // sumCentroidPositions<<<gridSize, blockSize>>>(d_image, d_pixel_cluster_indices, d_centroids, d_elements_per_cluster, width, height, cpp);

        // updateCentroidPositions<<<gridSize, blockSize>>>(d_image, d_centroids, d_elements_per_cluster, width, height, cpp, K);
        updateCentroidPositions<<<((K * cpp + 512 -1)/512), 512>>>(d_image, d_centroids, d_elements_per_cluster, width, height, cpp, K);
        getLastCudaError("Error while updating positions of centroids\n");
    }

    // Assign pixels to final clusters
    mapPixelsToCentroidValues<<<gridSize, blockSize>>>(d_image, d_pixel_cluster_indices, d_centroids, width, height, cpp, K);
    checkCudaErrors(cudaMemcpy(h_image, d_image, width * height * cpp * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // Measure execution time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time: %.4f \n", milliseconds);

    // Save the compreesed image
    char output_file[256]; 
    strcpy(output_file, image_file);
    char *extension = strrchr(output_file, '.');
    if (extension != NULL) *extension = '\0';  // Cut off the file extension
    strcat(output_file, "_compressedGPU.png"); 
    stbi_write_png(output_file, width, height, cpp, h_image, width * cpp);
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