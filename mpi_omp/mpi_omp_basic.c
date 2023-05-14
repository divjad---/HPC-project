/* Standard libraries */
#include <mpi.h>
#include <float.h>

/* Libraries for handling images */
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../libs/stb_image.h"
#include "../libs/stb_image_write.h"

/* K-MEANS PARAMETERS */
int K = 32;
int MAX_ITER = 20;

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

void updateCentroidPositions(unsigned char *image_in, int *pixel_cluster_indices, float *centroids, int width, int height, int cpp)
{
    float *cluster_values_per_channel = (float *)calloc(cpp * K, sizeof(float));
    int *elements_per_cluster = (int *)calloc(K, sizeof(int));

// Iterate over each pixel
#pragma omp parallel for schedule(dynamic, 16) reduction(+ : cluster_values_per_channel[ : K * cpp], elements_per_cluster[ : K]) // TODO select group size
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
                centroids[cluster * cpp + channel] = cluster_values_per_channel[cluster * cpp + channel] / elements_per_cluster[cluster];
            }
            else
            {
                centroids[cluster * cpp + channel] = image_in[random_pixel_i * cpp + channel];
            }
        }
    }
    free(elements_per_cluster);
}

void assignPixelsToNearestCentroids(unsigned char *image_in, int *pixel_cluster_indices, float *centroids, int width, int height, int cpp)
{
// Iterate through each pixel
#pragma omp parallel for schedule(dynamic, 16)
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

    if (argc > 2)
        K = atoi(argv[2]);
    if (argc > 3)
        MAX_ITER = atoi(argv[3]);

    /* Read the image */
    int width, height, cpp;
    unsigned char *input_image = stbi_load(image_file, &width, &height, &cpp, 0);

    /* Synchronize processes */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Begin measuring time */
    double start = MPI_Wtime();

    /* K-Means Image Compression*/
    {
        /* Split the image among the processes (takes into account image not necessarily divisible by number of processes) */
        int my_image_height;
        if (rank == 0)
        {
            // If height not divisible by number of processes then process 0 gets more height then others
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
                if (i == 0)
                {
                    process_image_height = height / num_processes + height % num_processes;
                }
                else
                {
                    process_image_height = height / num_processes;
                }
                counts_send[i] = process_image_height * width * cpp;
                displacements[i] = i * (process_image_height * width * cpp);
            }
        }
        MPI_Scatterv(input_image, counts_send, displacements, MPI_UNSIGNED_CHAR, my_image, my_image_height * width * cpp, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

        /* Initialize clusters (same seed -> equal initialization -> no need for broadcast) */
        srand(42);
        float *centroids = (float *)calloc(cpp * K, sizeof(float));
        init_clusters_random(input_image, centroids, width, height, cpp);

        /* Main loop */
        int num_my_pixels = my_image_height * width;
        int *pixel_cluster_indices = (int *)calloc(num_my_pixels, sizeof(int));
        for (int i = 0; i < MAX_ITER; i++)
        {
            assignPixelsToNearestCentroids(my_image, pixel_cluster_indices, centroids, width, my_image_height, cpp);
            updateCentroidPositions(my_image, pixel_cluster_indices, centroids, width, my_image_height, cpp);
        }

        /* Assign pixels to final clusters */
        for (int i = 0; i < num_my_pixels; i++)
        {
            int cluster = pixel_cluster_indices[i];
            for (int channel = 0; channel < cpp; channel++)
            {
                my_image[i * cpp + channel] = (unsigned char)centroids[cluster * cpp + channel];
            }
        }

        /* Gather the image from all the processes*/
        MPI_Gatherv(my_image, my_image_height * width * cpp, MPI_UNSIGNED_CHAR, input_image, counts_send, displacements, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

        /* Free the resources */
        free(pixel_cluster_indices);
        free(centroids);
    }

    /* End measuring time */
    double end = MPI_Wtime();

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
        strcat(output_file, "_compressed_mpi_basic.png");

        stbi_write_png(output_file, width, height, cpp, input_image, width * cpp);
    }

    /* Free the image */
    MPI_Barrier(MPI_COMM_WORLD);
    stbi_image_free(input_image);

    /* Finalize MPI */
    MPI_Finalize();

    return 0;
}