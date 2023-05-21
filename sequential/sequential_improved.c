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

// IMPROVEMENTS
// Cluster initalization with Kmeans++
// Fusion to reduce the number of sweeps through memory and improve performance
// Early stoppage : TODO, what criteria is best

// Default values
int K = 32;
int MAX_ITER = 16;
#define EARLY_STOPPAGE_THRESHOLD 0.3 // TESTED

void init_clusters_random(unsigned char *imageIn, float *centroids, int width, int height, int cpp)
{
    int index;
    int num_pixels = width * height;
    for (int i = 0; i < K; i++)
    {
        index = rand() % num_pixels;
        for (int j = 0; j < cpp; j++)
        {
            centroids[i * cpp + j] = (float)imageIn[index * cpp + j];
        }
    }
}

double calculatePSNR(unsigned char *original_image, unsigned char *compressed_image, int width, int height, int cpp)
{
    double mse = 0.0;
    int num_pixels = width * height;
    for (int i = 0; i < num_pixels * cpp; i++)
    {
        int diff = (int)original_image[i] - (int)compressed_image[i];
        mse += (double)(diff * diff);
    }

    mse /= (double)(num_pixels * cpp);

    if (mse == 0)
    {
        return INFINITY; // Both images are identical
    }

    return (10 * log10((255.0 * 255.0) / mse));
}

void init_clusters_kmeans_plus_plus(unsigned char *imageIn, float *centroids, int width, int height, int cpp)
{
    // Choose a random pixel for initial centroid
    int num_pixels = width * height;
    int random_pixel = rand() % num_pixels;
    for (int i = 0; i < cpp; i++)
    {
        centroids[i] = (float)imageIn[random_pixel * cpp + i];
    }

    float *distances;
    for (int k = 1; k < K; k++)
    {
        distances = malloc(num_pixels * sizeof(float));
        int farthest_pixel = 0;
        float max_distance = -1;

        // Compute the distance to the nearest centroid for each data point
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                int pixel_index = i * width + j;
                float min_distance = 1e5;
                for (int c = 1; c < k; c++)
                {

                    // Calculate Euclidean distance
                    float distance = 0;
                    for (int channel = 0; channel < cpp; channel++)
                    {
                        float temp = centroids[c * cpp + channel];
                        float diff = temp - (float)imageIn[pixel_index * cpp + channel];
                        distance += diff * diff;
                    }

                    if (distance < min_distance)
                    {
                        min_distance = distance;
                    }
                }
                distances[pixel_index] = min_distance;

                // Check if this pixel is the farthest one yet -> not selecting proportionally
                if (min_distance > max_distance)
                {
                    max_distance = min_distance;
                    farthest_pixel = pixel_index;
                }
            }
        }

        // Choose the next centroid to be the farthest data point
        for (int i = 0; i < cpp; i++)
        {
            centroids[k * cpp + i] = (float)imageIn[farthest_pixel * cpp + i];
        }
    }
    free(distances);
}

// FUSED Kmeans steps
void assignPixelsAndUpdateCentroids(unsigned char *imageIn, int *pixel_cluster_indices, float *centroids, int *centroids_sums, int *elements_per_cluster, int width, int height, int cpp)
{
    int num_pixels = width * height;

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
            // We do this here, because exec times are constantely a bit better than going through the whole image again at the end...
            // We could pass in an argument that indicates last iteration, but we have early stoppage option. No need to further complicate the code for such a tiny differences
            pixel_cluster_indices[i * width + j] = min_cluster_index;

            // Update cluster values and count
            for (int channel = 0; channel < cpp; channel++)
            {
                centroids_sums[min_cluster_index * cpp + channel] += imageIn[index + channel];
            }
            elements_per_cluster[min_cluster_index]++;
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

double kmeans_image_compression(unsigned char *imageIn, int width, int height, int cpp, int init_strategy, int fusion, int early_stopage, int measurePSNR)
{
    int num_pixels = width * height;
    float *centroids = (float *)calloc(cpp * K, sizeof(float));
    int *centroids_sums = (int *)calloc(cpp * K, sizeof(int));
    int *elements_per_cluster = (int *)calloc(K, sizeof(int));

    // Intialize clusters
    if (init_strategy == 0)
    {
        init_clusters_random(imageIn, centroids, width, height, cpp);
    }
    else
    {
        init_clusters_kmeans_plus_plus(imageIn, centroids, width, height, cpp);
    }

    int *pixel_cluster_indices = (int *)calloc(num_pixels, sizeof(int));
    float *previous_centroids;
    if (early_stopage == 1)
        previous_centroids = (float *)calloc(cpp * K, sizeof(float));

    // Main loop
    printf("Iteration times: [");
    for (int iteration = 0; iteration < MAX_ITER; iteration++)
    {
        double iteration_start = omp_get_wtime();
        if (fusion == 0)
        {
            assignPixelsToNearestCentroids(imageIn, pixel_cluster_indices, centroids, width, height, cpp);
            updateCentroidPositions(imageIn, pixel_cluster_indices, centroids, centroids_sums, elements_per_cluster, width, height, cpp);
        }
        else
        {
            assignPixelsAndUpdateCentroids(imageIn, pixel_cluster_indices, centroids, centroids_sums, elements_per_cluster, width, height, cpp);
        }

        // Check for early stoppage
        if (early_stopage == 1)
        {
            float max_change = 0.0;
            for (int i = 0; i < K * cpp; i++)
            {
                float change = fabs(centroids[i] - previous_centroids[i]);
                if (change > max_change)
                {
                    max_change = change;
                }
            }
            if (max_change <= EARLY_STOPPAGE_THRESHOLD)
            {
                printf("EARLY STOPPAGE ");
                break;
            }
            memcpy(previous_centroids, centroids, K * cpp * sizeof(float));
        }
        if (iteration > 0)
        {
            printf(", ");
        }
        printf("%lf", omp_get_wtime() - iteration_start);
    }
    printf("]\n");
    // if (fusion == 1){
    //     assignPixelsToNearestCentroids(imageIn, pixel_cluster_indices, centroids, width, height, cpp);
    // }

    // Assign pixels to final clusters
    double end_time;
    if (measurePSNR == 0)
    {
        for (int i = 0; i < num_pixels; i++)
        {
            int cluster = pixel_cluster_indices[i];
            for (int channel = 0; channel < cpp; channel++)
            {
                imageIn[i * cpp + channel] = (unsigned char)centroids[cluster * cpp + channel];
            }
        }
        end_time = omp_get_wtime();
    }
    else
    { // Measure PSNR
        unsigned char *original_image = (unsigned char *)malloc(num_pixels * cpp * sizeof(unsigned char));
        for (int i = 0; i < num_pixels; i++)
        {
            int cluster = pixel_cluster_indices[i];
            for (int channel = 0; channel < cpp; channel++)
            {
                original_image[i * cpp + channel] = imageIn[i * cpp + channel];
                imageIn[i * cpp + channel] = (unsigned char)centroids[cluster * cpp + channel];
            }
        }
        end_time = omp_get_wtime();
        double psnr = calculatePSNR(original_image, imageIn, width, height, cpp);
        printf("PSNR: %lf\n", psnr);
    }
    free(pixel_cluster_indices);
    free(centroids);
    free(centroids_sums);
    free(elements_per_cluster);
    if (early_stopage == 1)
        free(previous_centroids);

    // Return execution time
    return end_time;
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
    int init_strategy = 0; // 0-random, 1-kmeans++
    int fusion = 0;        // 0 for standard k-means, 1 for optimized k-means with fused operations
    int measurePSNR = 0;
    int early_stopage = 0;
    if (argc > 2)
        init_strategy = atoi(argv[2]);
    if (argc > 3)
        fusion = atoi(argv[3]);
    if (argc > 4)
        early_stopage = atoi(argv[4]);
    if (argc > 5)
        measurePSNR = atoi(argv[5]);
    if (argc > 6)
        K = atoi(argv[6]);
    if (argc > 7)
        MAX_ITER = atoi(argv[7]);

    int width, height, cpp;
    unsigned char *input_image = stbi_load(image_file, &width, &height, &cpp, 0);

    double start_time = omp_get_wtime();
    double end_time = kmeans_image_compression(input_image, width, height, cpp, init_strategy, fusion, early_stopage, measurePSNR);
    double elapsed_time = end_time - start_time;

    // Save the compreesed image
    char output_file[256];
    strcpy(output_file, image_file);
    char *extension = strrchr(output_file, '.');
    if (extension != NULL)
        *extension = '\0'; // Cut off the file extension
    strcat(output_file, "_compressed.png");
    stbi_write_png(output_file, width, height, cpp, input_image, width * cpp);

    printf("Execution time: %.4f\nInit: %d ... Fusion: %d ... Early stop: %d ... K: %d ... Max iter: %d\n",
           elapsed_time, init_strategy, fusion, early_stopage, K, MAX_ITER);

    stbi_image_free(input_image);
}