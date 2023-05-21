#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../libs/stb_image.h"
#include "../libs/stb_image_write.h"

// Performance is very similar if we use seperate int array for clusters sums or just one float centroids

// Default values
int K = 32;
int MAX_ITER = 20;

void init_clusters_random(unsigned char *imageIn, float *centroids, int width, int height, int cpp)
{
    int index;
    int num_pixels = width * height;
    for (int i = 0; i < K; i++)
    {
        index = rand() % num_pixels;
        for (int j = 0; j < cpp; j++)
        {
            centroids[i * cpp + j] = (float)(imageIn[index * cpp + j]);
        }
    }
}

void updateCentroidPositions(unsigned char *imageIn, int *pixel_cluster_indices, float *centroids, int *centroids_sums, int *elements_per_cluster, int width, int height, int cpp)
{

    // Iterate over each pixel
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int index = i * width + j;
            int cluster = pixel_cluster_indices[index];

            for (int channel = 0; channel < cpp; channel++)
            {
                centroids_sums[cluster * cpp + channel] += imageIn[index * cpp + channel];
            }

            elements_per_cluster[cluster]++;
        }
    }

    // Update each centroid position by calculating the average channel value
    for (int cluster = 0; cluster < K; cluster++)
    {
        int random_pixel_i = rand() % (width * height);
        for (int channel = 0; channel < cpp; channel++)
        {
            if (elements_per_cluster[cluster] > 0)
            {
                centroids[cluster * cpp + channel] = ((float)centroids_sums[cluster * cpp + channel]) / elements_per_cluster[cluster];
                // Reset centroid sums
                centroids_sums[cluster * cpp + channel] = 0;
            }
            else
            {
                // Assign random pixel to empty centroid
                centroids[cluster * cpp + channel] = imageIn[random_pixel_i * cpp + channel];
            }
        }
        // Reset centroid counts
        elements_per_cluster[cluster] = 0;
    }
}

void assignPixelsToNearestCentroids(unsigned char *imageIn, int *pixel_cluster_indices, float *centroids, int width, int height, int cpp)
{
    // Iterate through each pixel
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int index = (i * width + j) * cpp;

            // Find nearest centroid
            int min_cluster_index = 0;
            float min_distance = FLT_MAX;

            for (int cluster = 0; cluster < K; cluster++)
            {
                float curr_distance = 0;

                for (int channel = 0; channel < cpp; channel++)
                {
                    float diff = ((float)imageIn[index + channel] - centroids[cluster * cpp + channel]);
                    curr_distance += diff * diff;
                }

                if (curr_distance < min_distance)
                {
                    min_cluster_index = cluster;
                    min_distance = curr_distance;
                }
            }
            pixel_cluster_indices[i * width + j] = min_cluster_index;
        }
    }
}

void kmeans_image_compression(unsigned char *imageIn, int width, int height, int cpp)
{
    int num_pixels = width * height;
    float *centroids = (float *)calloc(cpp * K, sizeof(float));
    int *centroids_sums = (int *)calloc(cpp * K, sizeof(int));
    int *elements_per_cluster = (int *)calloc(K, sizeof(int));

    // Intialize clusters
    init_clusters_random(imageIn, centroids, width, height, cpp);

    int *pixel_cluster_indices = (int *)calloc(num_pixels, sizeof(int));

    // Main loop
    printf("Iteration times: [");
    for (int iteration = 0; iteration < MAX_ITER; iteration++)
    {
        double iteration_start = omp_get_wtime();
        assignPixelsToNearestCentroids(imageIn, pixel_cluster_indices, centroids, width, height, cpp);
        updateCentroidPositions(imageIn, pixel_cluster_indices, centroids, centroids_sums, elements_per_cluster, width, height, cpp);
        if (iteration > 0)
        {
            printf(", ");
        }
        printf("%lf", omp_get_wtime() - iteration_start);
    }
    printf("]\n");

    // Assign pixels to final clusters
    for (int i = 0; i < num_pixels; i++)
    {
        int cluster = pixel_cluster_indices[i];
        for (int channel = 0; channel < cpp; channel++)
        {
            imageIn[i * cpp + channel] = (unsigned char)centroids[cluster * cpp + channel];
        }
    }
    free(pixel_cluster_indices);
    free(centroids);
    free(centroids_sums);
    free(elements_per_cluster);
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Not enough arguments\n");
        exit(1);
    }
    srand(42);
    char *image_file = argv[1];
    if (argc > 2)
        K = atoi(argv[2]);
    if (argc > 3)
        MAX_ITER = atoi(argv[3]);

    int width, height, cpp;
    unsigned char *input_image = stbi_load(image_file, &width, &height, &cpp, 0);

    double start_time = omp_get_wtime();
    kmeans_image_compression(input_image, width, height, cpp);
    double elapsed_time = omp_get_wtime() - start_time;

    // Save the compreesed image
    char output_file[256];
    strcpy(output_file, image_file);
    char *extension = strrchr(output_file, '.');
    if (extension != NULL)
        *extension = '\0'; // Cut off the file extension
    strcat(output_file, "_compressed.png");
    stbi_write_png(output_file, width, height, cpp, input_image, width * cpp);

    printf("Execution time: %.4f seconds\n", elapsed_time);
    stbi_image_free(input_image);
}