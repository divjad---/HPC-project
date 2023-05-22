/* Standard libraries */
#include <mpi.h>
#include <float.h>
#include <math.h>

/* Libraries for handling images */
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../libs/stb_image.h"
#include "../libs/stb_image_write.h"

/* K-MEANS PARAMETERS */
int K = 32;
int MAX_ITER = 20;
float EARLY_STOPPAGE_THRESHOLD = 0.3f;

void init_clusters_random(unsigned char *image_in, float *centroids, int width, int height, int cpp)
{
    int index;
    int num_pixels = width * height;
    for (int i = 0; i < K; i++)
    {
        index = rand() % num_pixels;
        for (int j = 0; j < cpp; j++)
        {
            centroids[i * cpp + j] = (float)(image_in[index * cpp + j]);
        }
    }
}

void init_clusters_kmeans_plus_plus(unsigned char *image_in, float *centroids, int width, int height, int cpp)
{
    // Choose a random pixel for the initial centroid
    int num_pixels = width * height;
    int random_pixel = rand() % num_pixels;
    for (int i = 0; i < cpp; i++)
    {
        centroids[i] = (float)image_in[random_pixel * cpp + i];
    }

    float *distances;
    for (int k = 1; k < K; k++)
    {
        distances = malloc(num_pixels * sizeof(float));
        int farthest_pixel = 0;
        float max_distance = -1.0f;

        // Compute the distance to the nearest centroid for each data point
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                int pixel_index = i * width + j;
                float min_distance = 1e5;
                for (int c = 1; c < k; c++)
                {
                    // Euclidean distance
                    float distance = 0;
                    for (int channel = 0; channel < cpp; channel++)
                    {
                        float temp = centroids[c * cpp + channel];
                        float diff = temp - (float)image_in[pixel_index * cpp + channel];
                        distance += diff * diff;
                    }

                    if (distance < min_distance)
                    {
                        min_distance = distance;
                    }
                }
                distances[pixel_index] = min_distance;

                if (min_distance > max_distance)
                {
                    max_distance = min_distance;
                    farthest_pixel = pixel_index;
                }
            }
        }
        // Choose next centroid to be the farthest
        for (int i = 0; i < cpp; i++)
        {
            centroids[k * cpp + i] = (float)image_in[farthest_pixel * cpp + i];
        }
    }
    free(distances);
}

double calculatePSNR(unsigned char *original_image, unsigned char *compressed_image, int width, int height, int cpp)
{
    double psnr = 0.0;
    int num_pixels = width * height;
    for (int i = 0; i < num_pixels * cpp; i++)
    {
        int diff = (int)original_image[i] - (int)compressed_image[i];
        psnr += (double)(diff * diff);
    }
    psnr /= (double)(num_pixels * cpp);
    if (psnr == 0.0)
    {
        return INFINITY;
    }
    return (10 * log10((255.0 * 255.0) / psnr));
}

void assignPixelsAndUpdateCentroids(unsigned char *image_in, int *pixel_cluster_indices, int *elements_per_cluster, float *centroids, int width, int height, int cpp)
{
    int num_pixels = width * height;

    float *cluster_values_per_channel = (float *)calloc(cpp * K, sizeof(float));

    // Iterate through each pixel
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int index = (i * width + j) * cpp;

            // Find the nearest centroid
            int min_cluster_index = 0;
            float min_distance = FLT_MAX;

            for (int cluster = 0; cluster < K; cluster++)
            {
                float curr_distance = 0;

                for (int channel = 0; channel < cpp; channel++)
                {
                    float diff = ((float)image_in[index + channel] - centroids[cluster * cpp + channel]);
                    curr_distance += diff * diff;
                }

                if (curr_distance < min_distance)
                {
                    min_cluster_index = cluster;
                    min_distance = curr_distance;
                }
            }
            pixel_cluster_indices[i * width + j] = min_cluster_index;

            // Update cluster values and count
            for (int channel = 0; channel < cpp; channel++)
            {
                cluster_values_per_channel[min_cluster_index * cpp + channel] += image_in[index + channel];
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
                centroids[cluster * cpp + channel] = cluster_values_per_channel[cluster * cpp + channel];
            }
            else
            {
                centroids[cluster * cpp + channel] = image_in[random_pixel_i * cpp + channel] * elements_per_cluster[cluster];
            }
        }
    }

    // Free the memory
    free(cluster_values_per_channel);
}

void updateCentroidPositions(unsigned char *image_in, int *pixel_cluster_indices, int* elements_per_cluster, float *centroids, int width, int height, int cpp)
{
    float *cluster_values_per_channel = (float *)calloc(cpp * K, sizeof(float));

    // Iterate over each pixel
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            int index = i * width + j;
            int cluster = pixel_cluster_indices[index];

            for (int channel = 0; channel < cpp; channel++)
            {
                cluster_values_per_channel[cluster * cpp + channel] += image_in[index * cpp + channel];
            }

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
                centroids[cluster * cpp + channel] = cluster_values_per_channel[cluster * cpp + channel];
            }
            else
            {
                centroids[cluster * cpp + channel] = image_in[random_pixel_i * cpp + channel] * elements_per_cluster[cluster];
            }
        }
    }
    free(cluster_values_per_channel);
}

void assignPixelsToNearestCentroids(unsigned char *image_in, int *pixel_cluster_indices, int* elements_per_cluster, float *centroids, int width, int height, int cpp)
{
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
                float curr_distance = 0.0f;
                for (int channel = 0; channel < cpp; channel++)
                {
                    float diff = ((float)image_in[index + channel] - centroids[cluster * cpp + channel]);
                    curr_distance += diff * diff;
                }

                if (curr_distance < min_distance)
                {
                    min_cluster_index = cluster;
                    min_distance = curr_distance;
                }
            }
            pixel_cluster_indices[i * width + j] = min_cluster_index;
            elements_per_cluster[min_cluster_index] += 1;
        }
    }
}

int main(int argc, char **argv)
{
    /* Initialize MPI */
    MPI_Init(&argc, &argv);

    /* Get rank */
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Get number of processes*/
    int num_processes;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    /* Read the arguments */

    // #1 - Name of image
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

    // #2 - init strategy (0 = random, 1 = k-means++)
    int init_strategy = 0;
    if (argc > 2)
        init_strategy = atoi(argv[2]);

    // #3 - perform fusion (0 = off , 1 = on)
    int fusion = 0;
    if (argc > 3)
        fusion = atoi(argv[3]);

    // #4 - early stoppage (0 = off, 1 = on)
    int early_stoppage = 0;
    if (argc > 4)
        early_stoppage = atoi(argv[4]);

    // #5 - measure PSNR (0 = off, 1 = on)
    int measure_psnr = 0;
    if (argc > 5)
        measure_psnr = atoi(argv[5]);

    // #6 - number of clusters
    if (argc > 6)
        K = atoi(argv[6]);

    // #7 - number of iterations
    if (argc > 7)
        MAX_ITER = atoi(argv[7]);

    /* Read the image */
    int width, height, cpp;
    unsigned char *input_image = stbi_load(image_file, &width, &height, &cpp, 0);

    /* Begin measuring time */
    double start = MPI_Wtime();
    double end;

    /* K-Means Image Compression*/
    {
        /* Split the image among the processes (takes into account image not necessarily divisible by number of processes) */
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
        unsigned char *my_image = (unsigned char *)malloc(my_image_height * width * cpp * sizeof(int));
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

        /* Initialize clusters (same seed -> equal initialization -> no need for broadcast) */
        srand(42);
        float *centroids = (float *)calloc(cpp * K, sizeof(float));
        if (rank == 0)
        {
            if (init_strategy == 0)
            {
                init_clusters_random(input_image, centroids, width, height, cpp);
            }
            else
            {
                init_clusters_kmeans_plus_plus(input_image, centroids, width, height, cpp);
            }
        }
        MPI_Bcast(centroids, cpp * K, MPI_FLOAT, 0, MPI_COMM_WORLD);

        float *previous_centroids;
        if (early_stoppage == 1)
            previous_centroids = (float *)calloc(cpp * K, sizeof(float));

        /* Main loop */
        if (rank == 0)
        {
            printf("Iteration times: [");
        }
        int num_my_pixels = my_image_height * width;
        int *pixel_cluster_indices = (int *)calloc(num_my_pixels, sizeof(int));
        int *elements_per_cluster = (int *)calloc(cpp * K, sizeof(int));
        for (int i = 0; i < MAX_ITER; i++)
        {
            double iteration_start = MPI_Wtime();
            // Check fusion
            if (fusion == 0)
            {
                assignPixelsToNearestCentroids(my_image, pixel_cluster_indices, elements_per_cluster, centroids, width, my_image_height, cpp);
                updateCentroidPositions(my_image, pixel_cluster_indices, elements_per_cluster, centroids, width, my_image_height, cpp);
            }
            else
            {
                assignPixelsAndUpdateCentroids(my_image, pixel_cluster_indices, elements_per_cluster, centroids, width, my_image_height, cpp);
            }

            MPI_Allreduce(MPI_IN_PLACE, centroids, cpp * K, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, elements_per_cluster, cpp * K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            for (int j = 0; j < K; j++)
            {
                for (int k = 0; k < cpp; k++)
                {
                    centroids[j * cpp + k] /= elements_per_cluster[j];
                }
            }
            memset(elements_per_cluster, 0, cpp * K * sizeof(float));

            // Check for early stoppage
            if (early_stoppage == 1)
            {
                float max_change = 0.0f;
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
                    break;
                }
                memcpy(previous_centroids, centroids, K * cpp * sizeof(float));
            }

            if (i > 0 && rank == 0)
            {
                printf(", ");
            }
            if (rank == 0)
            {
                printf("%lf", MPI_Wtime() - iteration_start);
            }
        }
        if (rank == 0)
        {
            printf("]\n");
        }
        /* Assign pixels to final clusters */
        if (!measure_psnr)
        {
            for (int i = 0; i < num_my_pixels; i++)
            {
                int cluster = pixel_cluster_indices[i];
                for (int channel = 0; channel < cpp; channel++)
                {
                    my_image[i * cpp + channel] = (unsigned char)centroids[cluster * cpp + channel];
                }
            }
            end = MPI_Wtime();
        }
        else
        {
            unsigned char *my_original_image = (unsigned char *)malloc(num_my_pixels * cpp * sizeof(unsigned char));
            for (int i = 0; i < num_my_pixels; i++)
            {
                int cluster = pixel_cluster_indices[i];
                for (int channel = 0; channel < cpp; channel++)
                {
                    my_original_image[i * cpp + channel] = my_image[i * cpp + channel];
                    my_image[i * cpp + channel] = (unsigned char)centroids[cluster * cpp + channel];
                }
            }
            end = MPI_Wtime();
            double psnr = calculatePSNR(my_original_image, my_image, width, my_image_height, cpp);
            double psnr_joint = 0.0;
            /* Reduce psnr */
            MPI_Reduce(&psnr, &psnr_joint, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0)
            {
                printf("PSNR: %lf\n", psnr_joint);
            }
            free(my_original_image);
        }

        /* Gather the image from all the processes*/
        MPI_Gatherv(my_image, my_image_height * width * cpp, MPI_UNSIGNED_CHAR, input_image, counts_send, displacements, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

        /* Free the resources */
        free(my_image);
        free(counts_send);
        free(displacements);

        free(pixel_cluster_indices);
        free(elements_per_cluster);
        free(centroids);
        if (early_stoppage == 1)
        {
            free(previous_centroids);
        }
    }

    /* Save the compressed image */
    if (rank == 0)
    {
        printf("Execution time: %.4f seconds\n", end - start);

        // Name the compressed image
        char output_file[256];
        strcpy(output_file, image_file);
        char *extension = strrchr(output_file, '.');
        if (extension != NULL)
        {
            *extension = '\0';
        }
        strcat(output_file, "_compressed_mpi_improved.png");

        stbi_write_png(output_file, width, height, cpp, input_image, width * cpp);
    }

    /* Free the image */
    MPI_Barrier(MPI_COMM_WORLD);
    stbi_image_free(input_image);

    /* Finalize MPI */
    MPI_Finalize();

    return 0;
}