#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "helper_cuda.h"
#include <iostream>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#define K 16
#define MAX_ITER 100
#define BLOCK_SIZE 256
#define cpp 3

__managed__ float sum[3] = {0, 0, 0};
__managed__ int my_count[1] = {0};

__global__ void
setCentroidsGPU(unsigned char *imageIn, int *cluster_assignments, float *centroids, int width, int height) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int i = tid / width;
    int j = tid % width;

    if (j < width && i < height) {
        int index = (i * width + j) * cpp;
        int min_index = 0;
        float min_distance = 10000000.0f;
        for (int ccc = 0; ccc < K; ccc++) {
            float distance = 0;
            for (int c = 0; c < 3; c++) {
                float diff = imageIn[index + c] - centroids[ccc * cpp + c];

                distance += diff * diff;
            }

            if (distance < min_distance) {
                min_index = ccc;
                min_distance = distance;
            }
        }

        cluster_assignments[(i * width + j)] = min_index;
    }
}

__global__ void
meansGPU(unsigned char *imageIn, int *cluster_assignments, int width, int height, int cluster) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int i = tid / width;
    int j = tid % width;

    if (j < width && i < height) {
        int index = (i * width + j) * cpp;
        if (cluster_assignments[(i * width + j)] == cluster) {
            int pixel[3] = {imageIn[index], imageIn[index + 1], imageIn[index + 2]};
            atomicAdd(&my_count[0], 1);
            atomicAdd(&sum[0], pixel[0]);
            atomicAdd(&sum[1], pixel[1]);
            atomicAdd(&sum[2], pixel[2]);
        }
    }
}

__global__ void
compressImageGPU(unsigned char *imageIn, int *cluster_assignments, float *centroids, int width, int height) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int i = tid / width;
    int j = tid % width;

    if (j < width && i < height) {
        int index = (i * width + j) * cpp;
        int cluster = cluster_assignments[i * width + j];

        imageIn[index] = static_cast<unsigned char>(centroids[cluster * cpp]);
        imageIn[index + 1] = static_cast<unsigned char>(centroids[cluster * cpp + 1]);
        imageIn[index + 2] = static_cast<unsigned char>(centroids[cluster * cpp + 2]);
    }
}

void setCentroids(unsigned char *imageIn, int *cluster_assignments, float *centroids, int width, int height) {
    for (int iii = 0; iii < height; iii++) {
        for (int jjj = 0; jjj < width; jjj++) {
            int index = (iii * width + jjj) * cpp;
            int min_index = 0;
            float min_distance = 10000000.0f;
            for (int ccc = 0; ccc < K; ccc++) {
                float distance = 0;
                for (int j = 0; j < 3; j++) {
                    float diff = (float) imageIn[index + j] - centroids[ccc * cpp + j];

                    distance += diff * diff;
                }

                if (distance < min_distance) {
                    min_index = ccc;
                    min_distance = distance;
                }
            }

            cluster_assignments[iii * width + jjj] = min_index;
        }
    }
}

int compare_int(const void *a, const void *b) {
    return (*(int *) a - *(int *) b);
}

void calculateMedians(unsigned char *imageIn, int *cluster_assignments, float *centroids, int width, int height) {
    for (int iii = 0; iii < K; iii++) {
        int *reds = NULL;
        int *greens = NULL;
        int *blues = NULL;
        int count = 0;

        for (int ccc = 0; ccc < height; ccc++) {
            for (int jjj = 0; jjj < width; jjj++) {
                if (cluster_assignments[ccc * width + jjj] == iii) {
                    int index = (ccc * width + jjj) * cpp;
                    count++;
                    reds = (int *) realloc(reds, count * sizeof(int));
                    greens = (int *) realloc(greens, count * sizeof(int));
                    blues = (int *) realloc(blues, count * sizeof(int));
                    reds[count - 1] = imageIn[index];
                    greens[count - 1] = imageIn[index + 1];
                    blues[count - 1] = imageIn[index + 2];
                }
            }
        }

        if (count > 0) {
            qsort(reds, count, sizeof(int), compare_int);
            qsort(greens, count, sizeof(int), compare_int);
            qsort(blues, count, sizeof(int), compare_int);

            int mid = count / 2;
            if (count % 2 == 0) {
                centroids[iii * cpp] = (float) (reds[mid - 1] + reds[mid]) / 2.0f;
                centroids[iii * cpp + 1] = (float) (greens[mid - 1] + greens[mid]) / 2.0f;
                centroids[iii * cpp + 2] = (float) (blues[mid - 1] + blues[mid]) / 2.0f;
            } else {
                centroids[iii * cpp] = (float) reds[mid];
                centroids[iii * cpp + 1] = (float) greens[mid];
                centroids[iii * cpp + 2] = (float) blues[mid];
            }
        }

        free(reds);
        free(greens);
        free(blues);
    }
}

void calculateMeans(unsigned char *imageIn, int *cluster_assignments, float *centroids, int width, int height) {
    for (int iii = 0; iii < K; iii++) {
        sum[0] = 0;
        sum[1] = 0;
        sum[2] = 0;
        int count = 0;

        for (int ccc = 0; ccc < height; ccc++) {
            for (int jjj = 0; jjj < width; jjj++) {
                if (cluster_assignments[ccc * width + jjj] == iii) {
                    int index = (ccc * width + jjj) * cpp;
                    sum[0] += imageIn[index];
                    sum[1] += imageIn[index + 1];
                    sum[2] += imageIn[index + 2];
                    count++;
                }
            }
        }

        if (count > 0) {
            centroids[iii * cpp] = (float) sum[0] / (float) count;
            centroids[iii * cpp + 1] = (float) sum[1] / (float) count;
            centroids[iii * cpp + 2] = (float) sum[2] / (float) count;
        }
    }
}

double calculatePSNR(unsigned char *original_image, unsigned char *compressed_image, int width, int height) {
    double mse = 0.0;
    int num_pixels = width * height;
    for (int i = 0; i < num_pixels * cpp; i++) {
        int diff = (int) original_image[i] - (int) compressed_image[i];
        mse += (double) (diff * diff);
    }

    mse /= (double) (num_pixels * cpp);

    if (mse == 0) {
        return INFINITY; // Both images are identical, so the PSNR is infinite
    }

    double max_pixel_value = 255.0;
    double psnr = 10 * log10((max_pixel_value * max_pixel_value) / mse);
    return psnr;
}

float euclideanDistanceSquared(float *point1, unsigned char *point2) {
    float distance = 0.0;
    for (int i = 0; i < 3; i++) {
        float diff = point1[i] - point2[i];
        distance += diff * diff;
    }
    return distance;
}

void initializeCentroidsWithKMeansPlusPlus(unsigned char *imageIn, float *centroids, int width, int height) {
    int num_pixels = width * height;

    // Choose one centroid uniformly at random from the data points
    int random_index = rand() % num_pixels;
    for (int i = 0; i < cpp; i++) {
        centroids[i] = static_cast<float>(imageIn[random_index * cpp + i]);
    }

    for (int k = 1; k < K; k++) {
        float distances[num_pixels];
        float total_distance = 0;

        // Compute the distance to the nearest centroid for each data point
        for (int i = 0; i < num_pixels; i++) {
            float min_distance = euclideanDistanceSquared(centroids, &imageIn[i * cpp]);
            for (int j = 1; j < k; j++) {
                float distance = euclideanDistanceSquared(&centroids[j * cpp], &imageIn[i * cpp]);
                if (distance < min_distance) {
                    min_distance = distance;
                }
            }
            distances[i] = min_distance;
            total_distance += min_distance;
        }

        // Choose the next centroid with probability proportional to the distance squared
        float random_value = static_cast<float>(rand()) / RAND_MAX * total_distance;
        int chosen_index = 0;
        float cumulative_distance = distances[0];
        while (random_value > cumulative_distance) {
            chosen_index++;
            cumulative_distance += distances[chosen_index];
        }

        for (int i = 0; i < cpp; i++) {
            centroids[k * cpp + i] = static_cast<float>(imageIn[chosen_index * cpp + i]);
        }
    }
}

void calculateCpu(unsigned char *imageIn, int width, int height) {
    int num_pixels = width * height;
    float *centroids = (float *) calloc(3 * K, sizeof(float));

    /*
    for (int i = 0; i < K; i++) {
        int random_index = rand() % num_pixels;
        centroids[i * cpp] = static_cast<float>(imageIn[random_index * cpp]);
        centroids[i * cpp + 1] = static_cast<float>(imageIn[random_index * cpp + 1]);
        centroids[i * cpp + 2] = static_cast<float>(imageIn[random_index * cpp + 2]);
    }*/

    initializeCentroidsWithKMeansPlusPlus(imageIn, centroids, width, height);

    int *cluster_assignments = (int *) calloc(num_pixels, sizeof(int));

    for (int iii = 0; iii < MAX_ITER; iii++) {
        setCentroids(imageIn, cluster_assignments, centroids, width, height);

        //calculateMedians(imageIn, cluster_assignments, centroids, width, height);
        calculateMeans(imageIn, cluster_assignments, centroids, width, height);
    }

    unsigned char *compressed_image = new unsigned char[num_pixels * cpp];

    for (int i = 0; i < num_pixels; i++) {
        int cluster = cluster_assignments[i];
        compressed_image[i * cpp] = static_cast<unsigned char>(centroids[cluster * cpp]);
        compressed_image[i * cpp + 1] = static_cast<unsigned char>(centroids[cluster * cpp + 1]);
        compressed_image[i * cpp + 2] = static_cast<unsigned char>(centroids[cluster * cpp + 2]);
    }

    double psnr = calculatePSNR(imageIn, compressed_image, width, height);
    printf("PSNR CPU: %lf\n", psnr);

    // Save the compressed image
    stbi_write_png("image_compressed_CPU.png", width, height, cpp, compressed_image, width * cpp);

    free(centroids);
    free(compressed_image);
    free(cluster_assignments);
}

void calculateGpu(unsigned char *imageIn, int width, int height) {
    int num_pixels = width * height;
    const size_t blockSize = BLOCK_SIZE;
    const size_t gridSize = (num_pixels - 1) / BLOCK_SIZE + 1;
    float *h_centroids = (float *) calloc(cpp * K, sizeof(float));

    /*
    for (int i = 0; i < K; i++) {
        int random_index = rand() % num_pixels;
        h_centroids[i * cpp] = static_cast<float>(imageIn[random_index * cpp]);
        h_centroids[i * cpp + 1] = static_cast<float>(imageIn[random_index * cpp + 1]);
        h_centroids[i * cpp + 2] = static_cast<float>(imageIn[random_index * cpp + 2]);
    }*/

    initializeCentroidsWithKMeansPlusPlus(imageIn, h_centroids, width, height);

    unsigned char *d_image;
    checkCudaErrors(cudaMalloc(&d_image, width * height * cpp * sizeof(unsigned char)));
    checkCudaErrors(
            cudaMemcpy(d_image, imageIn, width * height * cpp * sizeof(unsigned char), cudaMemcpyHostToDevice));

    float *d_centroids;
    checkCudaErrors(cudaMalloc(&d_centroids, cpp * K * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_centroids, h_centroids, cpp * K * sizeof(float), cudaMemcpyHostToDevice));

    printf("Transfered centroids to GPU\n");

    int *h_cluster_assignments = (int *) calloc(num_pixels, sizeof(int));
    int *d_cluster_assignments;
    checkCudaErrors(cudaMalloc(&d_cluster_assignments, num_pixels * sizeof(int)));
    checkCudaErrors(cudaMemset(d_cluster_assignments, 0, num_pixels * sizeof(int)));

    for (int iii = 0; iii < MAX_ITER; iii++) {
        setCentroidsGPU<<<gridSize, blockSize>>>(d_image, d_cluster_assignments, d_centroids, width, height);
        getLastCudaError("setCentroidsGPU execution failed\n");
        cudaDeviceSynchronize();

        for (int jjj = 0; jjj < K; jjj++) {
            sum[0] = 0;
            sum[1] = 0;
            sum[2] = 0;
            my_count[0] = 0;

            meansGPU<<<gridSize, blockSize>>>(d_image, d_cluster_assignments, width, height, jjj);
            getLastCudaError("meansGPU execution failed\n");
            cudaDeviceSynchronize();

            if (my_count[0] > 0) {
                h_centroids[jjj * cpp] = (float) sum[0] / (float) my_count[0];
                h_centroids[jjj * cpp + 1] = (float) sum[1] / (float) my_count[0];
                h_centroids[jjj * cpp + 2] = (float) sum[2] / (float) my_count[0];

                checkCudaErrors(cudaMemcpy(d_centroids, h_centroids, cpp * K * sizeof(float), cudaMemcpyHostToDevice));
            }
        }
    }

    unsigned char *compressed_image = new unsigned char[num_pixels * cpp];

    checkCudaErrors(
            cudaMemcpy(h_cluster_assignments, d_cluster_assignments, num_pixels * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_centroids, d_centroids, cpp * K * sizeof(float), cudaMemcpyDeviceToHost));

    compressImageGPU<<<gridSize, blockSize>>>(d_image, d_cluster_assignments, d_centroids, width, height);
    cudaDeviceSynchronize();
    checkCudaErrors(
            cudaMemcpy(compressed_image, d_image, width * height * cpp * sizeof(unsigned char),
                       cudaMemcpyDeviceToHost));
    stbi_write_png("image_compressed_GPU.png", width, height, cpp, compressed_image, width * cpp);

    double psnr = calculatePSNR(imageIn, compressed_image, width, height);
    printf("PSNR GPU: %lf\n", psnr);

    printf("Finished\n");

    checkCudaErrors(cudaFree(d_centroids));
    checkCudaErrors(cudaFree(d_image));
    checkCudaErrors(cudaFree(d_cluster_assignments));
    free(h_centroids);
    free(compressed_image);
    free(h_cluster_assignments);
}

int main(int argc, char **argv) {
    printf("Before image load\n");

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image>" << std::endl;
        return 1;
    }

    printf("Before image load\n");

    int width, height, channels;
    unsigned char *input_image = stbi_load(argv[1], &width, &height, &channels, STBI_rgb);

    if (!input_image) {
        std::cerr << "Error: Unable to load image." << std::endl;
        return 1;
    }

    printf("Image loaded\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start
    cudaEventRecord(start);

    calculateCpu(input_image, width, height);

    cudaEventRecord(stop);

    // Wait for all events to finish
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Time CPU: %0.3f milliseconds \n", milliseconds);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    calculateGpu(input_image, width, height);

    cudaEventRecord(stop);

    // Wait for all events to finish
    cudaEventSynchronize(stop);

    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Time GPU: %0.3f milliseconds \n", milliseconds);
}