#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
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

void updateCentroidPositions(unsigned char *imageIn, int *pixel_cluster_indices, float *centroids, int width, int height, int cpp) {
    
    float *cluster_values_per_channel = (float *)calloc(cpp * K, sizeof(float));
    int *elements_per_cluster = (int *)calloc(K, sizeof(int));
    
    // Iterate over each pixel
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = i * width + j;
            int cluster = pixel_cluster_indices[index];

            for (int channel = 0; channel < cpp; channel++) {
                cluster_values_per_channel[cluster * cpp + channel] += imageIn[index * cpp + channel];
            }

            elements_per_cluster[cluster]++;
        }
    }

    // Update each centroid position by calculating the average channel value
    for (int cluster = 0; cluster < K; cluster++) {
        int random_pixel_i = rand() % (width * height);
        for (int channel = 0; channel < cpp; channel++) {
            if (elements_per_cluster[cluster] > 0) {
                centroids[cluster * cpp + channel] = cluster_values_per_channel[cluster * cpp + channel] / elements_per_cluster[cluster];
            }else{
                // Assign random pixel to empty centroid
                centroids[cluster * cpp + channel] = imageIn[random_pixel_i * cpp + channel];
            }
        }
    }
    free(elements_per_cluster);
}

void assignPixelsToNearestCentroids(unsigned char *imageIn, int *pixel_cluster_indices, float *centroids, int width, int height, int cpp) {
    // Iterate through each pixel
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int index = (i * width + j) * cpp;
            
            // Find nearest centroid
            int min_cluster_index = 0;
            float min_distance = FLT_MAX;

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
}

void kmeans_image_compression(unsigned char *imageIn, int width, int height, int cpp) {
    int num_pixels = width * height;
    float *centroids = (float *) calloc(cpp * K, sizeof(float));

    // Intialize clusters
    init_clusters_random(imageIn, centroids, width, height, cpp);
    
    int *pixel_cluster_indices = (int *)calloc(num_pixels, sizeof(int));

    // Main loop
    for (int iteration = 0; iteration < MAX_ITER; iteration++) {
        assignPixelsToNearestCentroids(imageIn, pixel_cluster_indices, centroids, width, height, cpp);
        updateCentroidPositions(imageIn, pixel_cluster_indices, centroids, width, height, cpp);
    }

    // Assign pixels to final clusters
    for (int i = 0; i < num_pixels; i++) {
        int cluster = pixel_cluster_indices[i];
        for (int channel = 0; channel < cpp; channel++) {
            imageIn[i * cpp + channel] = (unsigned char) centroids[cluster * cpp + channel];
        }
    }
    free(pixel_cluster_indices);
    free(centroids);
}

int main(int argc, char **argv)
{
    if (argc < 2){
        fprintf(stderr, "Not enough arguments\n");
        exit(1);
    }
    srand(42);
    char *image_file = argv[1];
    if (argc > 2) K = atoi(argv[2]);
    if (argc > 3) MAX_ITER = atoi(argv[3]);

    int width, height, cpp;
    unsigned char *h_image = stbi_load(image_file, &width, &height, &cpp, 0);
    
    if(!h_image) return 0;

    // Set block and grid sizes
    const size_t blockSize = BLOCK_SIZE;
    const size_t gridSize = (width * height + blockSize - 1) / blockSize;

    // Create CUDA events and start recording
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Copy data to GPU
    unsigned char *d_image;
    unsigned int *d_hist;

    checkCudaErrors(cudaMalloc(&d_image, width * height * cpp * sizeof(unsigned char)));



    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // TODO save d_image
    // Save the compreesed image
    char output_file[256]; 
    strcpy(output_file, image_file);
    char *extension = strrchr(output_file, '.');
    if (extension != NULL) *extension = '\0';  // Cut off the file extension
    strcat(output_file, "_compressedGPU.png"); 
    stbi_write_png(output_file, width, height, cpp, h_image, width * cpp);

    printf("Execution time: %.4f \n", milliseconds);
    stbi_image_free(h_image);
}