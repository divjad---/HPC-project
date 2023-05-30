/* Standard libraries */
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../libs/helper_cuda.h"
#include <float.h>

/* Libraries for handling images */
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../libs/stb_image.h"
#include "../libs/stb_image_write.h"

/* K-MEANS PARAMETERS */
int K = 32;
int MAX_ITER = 20;
int BLOCK_SIZE = 256;

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

__global__ void assignPixelsToNearestCentroids(unsigned char *imageIn, int *pixel_cluster_indices, float *centroids, int width, int height, int cpp, int K)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / width;
    int j = tid % width;

    // Find nearest centroid for each pixel
    if (i < height && j < width)
    {
        int index = (i * width + j) * cpp;
        int min_cluster_index = 0;
        float min_distance = 1e5;

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

__global__ void sumCentroidPositions(unsigned char *imageIn, int *pixel_cluster_indices, float *centroids, int *elements_per_clusters, int width, int height, int cpp)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / width;
    int j = tid % width;

    // Iterate over each pixel
    if (i < height && j < width)
    {
        int index = i * width + j;
        int cluster = pixel_cluster_indices[index];

        for (int channel = 0; channel < cpp; channel++)
        {
            atomicAdd(&centroids[cluster * cpp + channel], (float)imageIn[index * cpp + channel]);
        }

        atomicAdd(&elements_per_clusters[cluster], 1);
    }
}

// SHARED ATOMICS-> this works if K*cpp is less than block size -> else we should use for loops !!!
__global__ void sumCentroidPositionsSharedMemory(unsigned char *imageIn, int *pixel_cluster_indices, float *centroids, int *elements_per_clusters, int width, int height, int cpp, int K)
{

    extern __shared__ float sdata[];              // Shared memory for partial sums
    int *sdata_elements = (int *)&sdata[K * cpp]; // Shared memory for number of elements per cluster
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / width;
    int j = tid % width;

    // Initialize shared memory
    if (threadIdx.x < K * cpp)
    {
        sdata[threadIdx.x] = 0.0;
    }

    if (threadIdx.x < K)
    {
        sdata_elements[threadIdx.x] = 0;
    }
    __syncthreads();

    // Iterate over each pixel
    if (i < height && j < width)
    {
        int index = i * width + j;
        int cluster = pixel_cluster_indices[index];

        for (int channel = 0; channel < cpp; channel++)
        {
            atomicAdd(&sdata[cluster * cpp + channel], (float)imageIn[index * cpp + channel]);
        }
        atomicAdd(&sdata_elements[cluster], 1);

        __syncthreads();

        // Update clusters and counts in global memory
        if (threadIdx.x < K * cpp)
        {
            atomicAdd(&centroids[threadIdx.x], sdata[threadIdx.x]);
        }
        if (threadIdx.x < K)
        {
            atomicAdd(&elements_per_clusters[threadIdx.x], sdata_elements[threadIdx.x]);
        }
    }
}

// K * cpp can be > block size
__global__ void sumCentroidPositionsSharedMemoryWOConstraints(unsigned char *imageIn, int *pixel_cluster_indices, float *centroids, int *elements_per_clusters, int width, int height, int cpp, int K)
{

    extern __shared__ float sdata[];              // Shared memory for partial sums
    int *sdata_elements = (int *)&sdata[K * cpp]; // Shared memory for number of elements per cluster
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / width;
    int j = tid % width;

    // Initialize shared memory
    for (int idx = threadIdx.x; idx < K * cpp; idx += blockDim.x)
    {
        sdata[idx] = 0.0f;
    }

    for (int idx = threadIdx.x; idx < K; idx += blockDim.x)
    {
        sdata_elements[idx] = 0;
    }
    __syncthreads();

    // Iterate over each pixel
    if (i < height && j < width)
    {
        int index = i * width + j;
        int cluster = pixel_cluster_indices[index];

        for (int channel = 0; channel < cpp; channel++)
        {
            atomicAdd(&sdata[cluster * cpp + channel], (float)imageIn[index * cpp + channel]);
        }
        atomicAdd(&sdata_elements[cluster], 1);
    }

    __syncthreads();

    // Update clusters and counts in global memory
    for (int idx = threadIdx.x; idx < K * cpp; idx += blockDim.x)
    {
        atomicAdd(&centroids[idx], sdata[idx]);
    }

    for (int idx = threadIdx.x; idx < K; idx += blockDim.x)
    {
        atomicAdd(&elements_per_clusters[idx], sdata_elements[idx]);
    }
}

__global__ void updateCentroidPositions(unsigned char *imageIn, float *centroids, int *elements_per_clusters, int width, int height, int cpp, int K)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Update each centroid position by calculating the average channel value
    if (tid < K * cpp)
    {
        int cluster = tid / cpp;
        int channel = tid % cpp;

        if (elements_per_clusters[cluster] > 0)
        {
            centroids[tid] = centroids[tid] / elements_per_clusters[cluster];
        }
        else
        {
            // Assign random pixel to empty centroid
            // int random_pixel_i = rand() % (width * height); // TODO
            int random_pixel_i = 0;
            centroids[tid] = imageIn[random_pixel_i * cpp + channel];
        }
    }
}

__global__ void mapPixelsToCentroidValues(unsigned char *imageIn, int *pixel_cluster_indices, float *centroids, int width, int height, int cpp, int K)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = tid / width;
    int j = tid % width;

    // Iterate over each pixel
    if (i < height && j < width)
    {
        int index = i * width + j;
        int cluster = pixel_cluster_indices[index];

        for (int channel = 0; channel < cpp; channel++)
        {
            imageIn[index * cpp + channel] = (unsigned char)centroids[cluster * cpp + channel];
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

    /* Get number of processes */
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
        BLOCK_SIZE = atoi(argv[2]);
    if (argc > 3)
        K = atoi(argv[3]);
    if (argc > 4)
        MAX_ITER = atoi(argv[4]);

    /* Read the image */
    int width, height, cpp;
    unsigned char *input_image = stbi_load(image_file, &width, &height, &cpp, 0);

    /* Synchronize the processes */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Begin measuring time */
    double start = MPI_Wtime();

    /* K-Means Image Compression */

    /* Set random seed*/
    srand(42);

    /* Split the image */
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
    unsigned char *my_image = (unsigned char *)malloc(my_image_height * width * cpp * sizeof(unsigned char));
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

    /* Start the algorithm on GPU */

    // Set block size and grid sizes
    const size_t block_size = BLOCK_SIZE;
    const size_t grid_size = (my_image_height * width + block_size - 1) / block_size;

    // Initialize clusters
    float *h_centroids = (float *)calloc(cpp * K, sizeof(float));
    if (rank == 0) {
        init_clusters_random(input_image, h_centroids, width, height, cpp);
    }
    MPI_Bcast(h_centroids, cpp * K, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Copy data to the GPU
    unsigned char *d_my_image;
    checkCudaErrors(cudaMalloc(&d_my_image, my_image_height * width * cpp * sizeof(unsigned char)));
    checkCudaErrors(cudaMemcpy(d_my_image, my_image, my_image_height * width * cpp * sizeof(unsigned char), cudaMemcpyHostToDevice));

    float *d_centroids;
    checkCudaErrors(cudaMalloc(&d_centroids, K * cpp * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_centroids, h_centroids, K * cpp * sizeof(float), cudaMemcpyHostToDevice));

    int *d_pixel_cluster_indices;
    checkCudaErrors(cudaMalloc(&d_pixel_cluster_indices, my_image_height * width * cpp * sizeof(int)));

    int *d_elements_per_cluster;
    checkCudaErrors(cudaMalloc(&d_elements_per_cluster, K * sizeof(int)));

    // Main loop
    for (int i = 0; i < MAX_ITER; i++)
    {
        assignPixelsToNearestCentroids<<<grid_size, block_size>>>(d_my_image, d_pixel_cluster_indices, d_centroids, width, my_image_height, cpp, K);
        getLastCudaError("Error assigning pixels to centroids!\n");
        cudaDeviceSynchronize();

        cudaMemset(d_centroids, 0, K * cpp * sizeof(float));
        cudaMemset(d_elements_per_cluster, 0, K * sizeof(float));
        int shared_memory_size = (K * cpp + K) * sizeof(float);
        sumCentroidPositionsSharedMemory<<<grid_size, block_size, shared_memory_size>>>(d_my_image, d_pixel_cluster_indices, d_centroids, d_elements_per_cluster, width, my_image_height, cpp, K);
        getLastCudaError("Error summing centroids positions in shared memory!\n");
        cudaDeviceSynchronize();

        updateCentroidPositions<<<((K * cpp + block_size - 1) / block_size), block_size>>>(d_my_image, d_centroids, d_elements_per_cluster, width, my_image_height, cpp, K);
        getLastCudaError("Error updating centroid positions!\n");

        // Copy centroids from device -> host
        checkCudaErrors(cudaMemcpy(h_centroids, d_centroids, K * cpp * sizeof(float), cudaMemcpyDeviceToHost));
        // Reduce centroids
        MPI_Allreduce(MPI_IN_PLACE, h_centroids, cpp * K, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        for (int j = 0; j < cpp * K; j++)
        {
            h_centroids[j] /= num_processes;
        }
        // Copy centroids host -> device
        checkCudaErrors(cudaMemcpy(d_centroids, h_centroids, K * cpp * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Asign pixels to final clusters
    mapPixelsToCentroidValues<<<grid_size, block_size>>>(d_my_image, d_pixel_cluster_indices, d_centroids, width, my_image_height, cpp, K);

    // Copy image to host
    checkCudaErrors(cudaMemcpy(my_image, d_my_image, width * my_image_height * cpp * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    /* Gather the image from all the processes */
    MPI_Gatherv(my_image, my_image_height * width * cpp, MPI_UNSIGNED_CHAR, input_image, counts_send, displacements, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

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
        strcat(output_file, "_compressed_mpi_cuda_basic.png");

        stbi_write_png(output_file, width, height, cpp, input_image, width * cpp);
    }

    stbi_image_free(input_image);

    /* Finalize MPI */
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}