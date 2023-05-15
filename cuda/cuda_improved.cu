#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include "../libs/helper_cuda.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../libs/stb_image.h"
#include "../libs/stb_image_write.h"

// Default values if not set with arguments
int K = 32;
int MAX_ITER = 20;
int BLOCK_SIZE = 256;
#define EARLY_STOPPAGE_THRESHOLD 0.3f

void init_clusters_random(unsigned char *imageIn, float *centroids, int width, int height, int cpp) {
    int index;
    int num_pixels = width * height;
    for (int i = 0; i < K; i++) {
        index = rand() % num_pixels;
        for (int j = 0; j < cpp; j++) {
            centroids[i * cpp + j] = (float) (imageIn[index * cpp + j]);
        }
    }
}

__global__ void
computeDistances(unsigned char *imageIn, float *centroids, float *distances, int width, int height, int cpp, int k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int pixel_index = (y * width + x) * cpp;
        float min_distance = 1e5;

        for (int c = 0; c < k; c++) {
            float distance = 0;

            for (int channel = 0; channel < cpp; channel++) {
                float temp = centroids[c * cpp + channel];
                float diff = temp - static_cast<float>(imageIn[pixel_index + channel]);
                distance += diff * diff;
            }

            if (distance < min_distance) {
                min_distance = distance;
            }
        }

        distances[y * width + x] = min_distance;
    }
}

void
init_clusters_kmeans_plus_plus(unsigned char *imageIn, unsigned char *d_image, float *centroids, float *d_centroids,
                               int width, int height, int cpp) {
    int num_pixels = width * height;

    // Choose a random pixel for the initial centroid
    int random_pixel = rand() % num_pixels;
    for (int i = 0; i < cpp; i++) {
        centroids[i] = static_cast<float>(imageIn[random_pixel * cpp + i]);
    }

    float *distances;
    cudaMalloc((void **) &distances, num_pixels * sizeof(float));

    const size_t blockSize = BLOCK_SIZE;
    const size_t gridSize = (width * height + blockSize - 1) / blockSize;

    for (int k = 1; k < K; k++) {
        // Compute the distance to the nearest centroid for each data point
        computeDistances<<<gridSize, blockSize>>>(d_image, d_centroids, distances, width, height, cpp, k);

        // Copy distances from device to host for finding the farthest pixel
        float *hostDistances = new float[num_pixels];
        cudaMemcpy(hostDistances, distances, num_pixels * sizeof(float), cudaMemcpyDeviceToHost);

        // Find the farthest pixel
        int farthest_pixel = 0;
        float max_distance = -1;
        for (int i = 0; i < num_pixels; i++) {
            if (hostDistances[i] > max_distance) {
                max_distance = hostDistances[i];
                farthest_pixel = i;
            }
        }

        delete[] hostDistances;

        // Choose the next centroid to be the farthest data point
        for (int i = 0; i < cpp; i++) {
            centroids[k * cpp + i] = static_cast<float>(imageIn[farthest_pixel * cpp + i]);
        }
    }

    cudaFree(distances);
}

/*
void init_clusters_kmeans_plus_plus(unsigned char *imageIn, float *centroids, int width, int height, int cpp) {
    // Choose a random pixel for initial centroid
    int num_pixels = width * height;
    int random_pixel = rand() % num_pixels;
    for (int i = 0; i < cpp; i++) {
        centroids[i] = (float) imageIn[random_pixel * cpp + i];
    }

    float *distances;
    for (int k = 1; k < K; k++) {
        distances = (float *) malloc(num_pixels * sizeof(float));
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
                        float diff = temp - (float) imageIn[pixel_index * cpp + channel];
                        distance += diff * diff;
                    }

                    if (distance < min_distance) {
                        min_distance = distance;
                    }
                }
                //distances[pixel_index] = min_distance;

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
        free(distances);
    }
}*/

__global__ void
assignPixelsToNearestCentroids(unsigned char *imageIn, int *pixel_cluster_indices, float *centroids, int width,
                               int height, int cpp, int K) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / width;
    int j = tid % width;

    // Find nearest centroid for each pixel
    if (i < height && j < width) {
        int index = (i * width + j) * cpp;
        int min_cluster_index = 0;
        float min_distance = 1e5;

        for (int cluster = 0; cluster < K; cluster++) {
            float curr_distance = 0;

            for (int channel = 0; channel < cpp; channel++) {
                float diff = ((float) imageIn[index + channel] - centroids[cluster * cpp + channel]);
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

__global__ void
sumCentroidPositions(unsigned char *imageIn, int *pixel_cluster_indices, float *centroids, int *elements_per_clusters,
                     int width, int height, int cpp) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / width;
    int j = tid % width;

    // Iterate over each pixel
    if (i < height && j < width) {
        int index = i * width + j;
        int cluster = pixel_cluster_indices[index];

        for (int channel = 0; channel < cpp; channel++) {
            atomicAdd(&centroids[cluster * cpp + channel], (float) imageIn[index * cpp + channel]);
        }

        atomicAdd(&elements_per_clusters[cluster], 1);
    }
}

// TRYING TO REDUCE NUMBER OF ATOMIC OPERATIONS -> this works if K*cpp is less than block size -> else we should use for loops
__global__ void sumCentroidPositionsSharedMemory(unsigned char *imageIn, int *pixel_cluster_indices, float *centroids,
                                                 int *elements_per_clusters, int width, int height, int cpp, int K) {
    extern __shared__ float sdata[]; // Shared memory for partial sums
    int *sdata_elements = (int *) &sdata[K * cpp]; // Shared memory for number of elements per cluster
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
            atomicAdd(&sdata[cluster * cpp + channel], (float) imageIn[index * cpp + channel]);
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

__global__ void
updateCentroidPositions(unsigned char *imageIn, float *centroids, int *elements_per_clusters, int width, int height,
                        int cpp, int K) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Update each centroid position by calculating the average channel value
    if (tid < K * cpp) {
        int cluster = tid / cpp;
        int channel = tid % cpp;

        if (elements_per_clusters[cluster] > 0) {
            centroids[tid] = centroids[tid] / elements_per_clusters[cluster];
        } else {
            // Assign random pixel to empty centroid
            curandState_t state;
            curand_init(tid * 123, tid, 0, &state);
            int random_pixel_i = static_cast<int>(curand_uniform(&state)) % (width * height);
            centroids[tid] = imageIn[random_pixel_i * cpp + channel];
        }
    }
}

__global__ void
mapPixelsToCentroidValues(unsigned char *imageIn, int *pixel_cluster_indices, float *centroids, int width, int height,
                          int cpp, int K) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / width;
    int j = tid % width;

    // Iterate over each pixel
    if (i < height && j < width) {
        int index = i * width + j;
        int cluster = pixel_cluster_indices[index];

        for (int channel = 0; channel < cpp; channel++) {
            imageIn[index * cpp + channel] = (unsigned char) centroids[cluster * cpp + channel];
        }
    }
}

__global__ void
calculatePSNR(unsigned char *original_image, unsigned char *compressed_image, int width, int height, int cpp,
              float *mse) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / width;
    int j = tid % width;

    // Find nearest centroid for each pixel
    if (i < height && j < width) {
        int index = (i * width + j) * cpp;
        float diff = 0.0f;
        for (int c = 0; c < cpp; c++) {
            diff = (float) original_image[index + c] - (double) compressed_image[index + c];
            diff = diff * diff;
            atomicAdd(mse, diff);
        }
    }
}

void
kmeans_image_compression(unsigned char *h_image, int width, int height, int cpp, char *image_file, int init_strategy,
                         int early_stoppage, int measurePSNR) {
    int num_pixels = width * height;

    // Set block and grid sizes
    const size_t blockSize = BLOCK_SIZE;
    const size_t gridSize = (width * height + blockSize - 1) / blockSize;

    // Copy data to GPU
    unsigned char *d_image;
    unsigned char *d_image_copy;
    float *d_centroids;
    int *d_pixel_cluster_indices;
    int *d_elements_per_cluster;

    checkCudaErrors(cudaMalloc(&d_image, width * height * cpp * sizeof(unsigned char)));

    checkCudaErrors(cudaMalloc(&d_centroids, K * cpp * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_pixel_cluster_indices, width * height * sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_elements_per_cluster, K * sizeof(int)));

    checkCudaErrors(cudaMemcpy(d_image, h_image, width * height * cpp * sizeof(unsigned char), cudaMemcpyHostToDevice));

    if (measurePSNR == 1) {
        checkCudaErrors(cudaMalloc(&d_image_copy, width * height * cpp * sizeof(unsigned char)));
        checkCudaErrors(cudaMemcpy(d_image_copy, d_image, width * height * cpp * sizeof(unsigned char),
                                   cudaMemcpyDeviceToDevice));
    }

    getLastCudaError("Error while copying data to GPU\n");

    // Intialize clusters
    float *h_centroids = (float *) calloc(cpp * K, sizeof(float));
    float *prev_centroids = (float *) calloc(cpp * K, sizeof(float));
    if (init_strategy == 0) {
        init_clusters_random(h_image, h_centroids, width, height, cpp);
        checkCudaErrors(cudaMemcpy(d_centroids, h_centroids, K * cpp * sizeof(float), cudaMemcpyHostToDevice));
    } else {
        init_clusters_kmeans_plus_plus(h_image, d_image, h_centroids, d_centroids, width, height, cpp);
        checkCudaErrors(cudaMemcpy(d_centroids, h_centroids, K * cpp * sizeof(float), cudaMemcpyHostToDevice));
    }
    memcpy(prev_centroids, h_centroids, K * cpp * sizeof(float));

    // Main loop
    for (int iteration = 0; iteration < MAX_ITER; iteration++) {
        assignPixelsToNearestCentroids<<<gridSize, blockSize>>>(d_image, d_pixel_cluster_indices, d_centroids, width,
                                                                height, cpp, K);
        getLastCudaError("Error while assigning pixels to nearest centroids\n");

        cudaMemset(d_centroids, 0, K * cpp * sizeof(float));
        cudaMemset(d_elements_per_cluster, 0, K * sizeof(int));

        int shared_memory_size = (K * cpp + K) * sizeof(float);
        sumCentroidPositionsSharedMemory<<<gridSize, blockSize, shared_memory_size>>>(d_image, d_pixel_cluster_indices,
                                                                                      d_centroids,
                                                                                      d_elements_per_cluster, width,
                                                                                      height, cpp, K);

        updateCentroidPositions<<<gridSize, blockSize>>>(d_image, d_centroids, d_elements_per_cluster,
                                                         width, height, cpp, K);
        getLastCudaError("Error while updating positions of centroids\n");

        if (early_stoppage == 1) {
            checkCudaErrors(cudaMemcpy(h_centroids, d_centroids, cpp * K * sizeof(float), cudaMemcpyDeviceToHost));
            float max_change = 0.0;
            for (int i = 0; i < K * cpp; i++) {
                float change = fabs(h_centroids[i] - prev_centroids[i]);
                if (change > max_change) {
                    max_change = change;
                }
            }

            if (max_change <= EARLY_STOPPAGE_THRESHOLD) {
                printf("EARLY STOPPAGE: change %lf is below threshold %lf\n", max_change, EARLY_STOPPAGE_THRESHOLD);
                break;
            }
            memcpy(prev_centroids, h_centroids, K * cpp * sizeof(float));
        }
    }

    // Assign pixels to final clusters
    mapPixelsToCentroidValues<<<gridSize, blockSize>>>(d_image, d_pixel_cluster_indices, d_centroids, width, height,
                                                       cpp, K);

    if (measurePSNR) {
        float *d_mse;
        checkCudaErrors(cudaMalloc(&d_mse, sizeof(float)));
        checkCudaErrors(cudaMemset(d_mse, 0, sizeof(float)));
        calculatePSNR<<<gridSize, blockSize>>>(d_image_copy, d_image, width, height,
                                               cpp, d_mse);
        cudaDeviceSynchronize();
        float mse;
        checkCudaErrors(cudaMemcpy(&mse, d_mse, sizeof(float), cudaMemcpyDeviceToHost));

        mse /= (float) (num_pixels * cpp);

        if (mse == 0) {
            printf("PSNR: identical images\n");
        } else {
            double psnr = (10 * log10((255.0 * 255.0) / mse));
            printf("PSNR: %lf\n", psnr);
        }

        checkCudaErrors(cudaFree(d_mse));
        checkCudaErrors(cudaFree(d_image_copy));
    }

    // Save the compreesed image
    char output_file[256];
    strcpy(output_file, image_file);
    char *extension = strrchr(output_file, '.');
    if (extension != NULL) *extension = '\0';  // Cut off the file extension
    strcat(output_file, "_compressedGPU_improved.png");
    checkCudaErrors(cudaMemcpy(h_image, d_image, width * height * cpp * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    stbi_write_png(output_file, width, height, cpp, h_image, width * cpp);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Not enough arguments\n");
        exit(1);
    }
    srand(42);
    char *image_file = argv[1];
    int init_strategy = 1;
    int measurePSNR = 1;
    int early_stoppage = 0;
    if (argc > 2) init_strategy = atoi(argv[2]);
    if (argc > 3) early_stoppage = atoi(argv[3]);
    if (argc > 4) measurePSNR = atoi(argv[4]);
    if (argc > 5) BLOCK_SIZE = atoi(argv[5]);
    if (argc > 6) K = atoi(argv[6]);
    if (argc > 7) MAX_ITER = atoi(argv[7]);

    int width, height, cpp;
    unsigned char *h_image = stbi_load(image_file, &width, &height, &cpp, 0);

    if (!h_image) return 0;

    printf("Image path: %s\n", image_file);

    // Create CUDA events and start recording
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    kmeans_image_compression(h_image, width, height, cpp, image_file, init_strategy, early_stoppage, measurePSNR);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Exec time: %.4fms ... Init: %d ... Early stop: %d ... BLOCK SIZE: %d ... K: %d ... Max iter: %d\n",
           milliseconds, init_strategy, early_stoppage, BLOCK_SIZE, K, MAX_ITER);
    stbi_image_free(h_image);
}