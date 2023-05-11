#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include "helper_cuda.h"
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#define K 16
#define MAX_ITER 100
#define EPSILON 1e-5

__global__ void assign_clusters(const unsigned char *image_data, int *cluster_assignment, const float *centroids, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels) {
        int pixel_index = idx * 3;
        int pixel[3] = {image_data[pixel_index], image_data[pixel_index + 1], image_data[pixel_index + 2]};

        float min_distance = 1e20;
        int min_index = -1;
        for (int i = 0; i < K; i++) {
            int centroid_index = i * 3;
            float diff[3] = {pixel[0] - centroids[centroid_index], pixel[1] - centroids[centroid_index + 1],
                           pixel[2] - centroids[centroid_index + 2]};
            float distance = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
            if (distance < min_distance) {
                min_distance = distance;
                min_index = i;
            }
        }
        cluster_assignment[idx] = min_index;
    }
}

__global__ void update_centroids(const unsigned char *image_data, const int *cluster_assignment, float *centroids, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < K) {
        float sum[3] = {0, 0, 0};
        int count = 0;

        for (int i = 0; i < num_pixels; i++) {
            if (cluster_assignment[i] == idx) {
                int pixel_index = i * 3;
                int pixel[3] = {image_data[pixel_index], image_data[pixel_index + 1], image_data[pixel_index + 2]};
                sum[0] += pixel[0];
                sum[1] += pixel[1];
                sum[2] += pixel[2];
                count++;
            }
        }

        if (count > 0) {
            int centroid_index = idx * 3;
            centroids[centroid_index] = sum[0] / count;
            centroids[centroid_index + 1] = sum[1] / count;
            centroids[centroid_index + 2] = sum[2] / count;
        }
    }
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

    int num_pixels = width * height;
    int grid_size = (num_pixels + 1023) / 1024;

    unsigned char *d_image_data;
    cudaMalloc(&d_image_data, num_pixels * 3 * sizeof(unsigned char));
    cudaMemcpy(d_image_data, input_image, num_pixels * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    printf("Image copied to device\n");

    int *d_cluster_assignment;
    cudaMalloc(&d_cluster_assignment, num_pixels * sizeof(int));

    float *d_centroids;
    cudaMalloc(&d_centroids, K * 3 * sizeof(float));

    // Initialize centroids
    srand(time(NULL));
    for (int i = 0; i < K; i++) {
        printf("Cluster: %d\n", i);
        int random_index = rand() % num_pixels;
        float centroid[3] = {static_cast<float>(input_image[random_index * 3]),
                             static_cast<float>(input_image[random_index * 3 + 1]),
                             static_cast<float>(input_image[random_index * 3 + 2])};
        cudaMemcpy(d_centroids + i * 3, centroid, 3 * sizeof(float), cudaMemcpyHostToDevice);
    }

    // K-means clustering
    for (int iter = 0; iter < MAX_ITER; iter++) {
        printf("Iteration: %d\n", iter);
        assign_clusters<<<grid_size, 1024>>>(d_image_data, d_cluster_assignment, d_centroids, num_pixels);
        cudaDeviceSynchronize();

        float old_centroids[K * 3];
        cudaMemcpy(old_centroids, d_centroids, K * 3 * sizeof(float), cudaMemcpyDeviceToHost);

        update_centroids<<<(K + 1023) / 1024, 1024>>>(d_image_data, d_cluster_assignment, d_centroids, num_pixels);
        cudaDeviceSynchronize();

        float new_centroids[K * 3];
        cudaMemcpy(new_centroids, d_centroids, K * 3 * sizeof(float), cudaMemcpyDeviceToHost);

        float max_diff = 0;
        for (int i = 0; i < K; i++) {
            float diff[3] = {old_centroids[i * 3] - new_centroids[i * 3],
                             old_centroids[i * 3 + 1] - new_centroids[i * 3 + 1],
                             old_centroids[i * 3 + 2] - new_centroids[i * 3 + 2]};
            float dist = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
            max_diff = fmaxf(max_diff, dist);
        }

        if (max_diff < EPSILON) break;
    }

    // Replace pixels with their corresponding centroid values
    unsigned char *compressed_image = new unsigned char[num_pixels * 3];

    for (int i = 0; i < num_pixels; i++) {
        int cluster;
        cudaMemcpy(&cluster, d_cluster_assignment + i, sizeof(int), cudaMemcpyDeviceToHost);
        float centroid[3];
        cudaMemcpy(centroid, d_centroids + cluster * 3, 3 * sizeof(float), cudaMemcpyDeviceToHost);
        compressed_image[i * 3] = static_cast<unsigned char>(centroid[0]);
        compressed_image[i * 3 + 1] = static_cast<unsigned char>(centroid[1]);
        compressed_image[i * 3 + 2] = static_cast<unsigned char>(centroid[2]);
    }

    // Save the compressed image
    stbi_write_png("image_compressed.png", width, height, 3, compressed_image, width * 3);

    // Clean up
    delete[] compressed_image;
    stbi_image_free(input_image);
    cudaFree(d_image_data);
    cudaFree(d_cluster_assignment);
    cudaFree(d_centroids);

    return 0;
}