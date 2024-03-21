
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <cstdint>      // Data types
#include <iostream>     // File operations
#include <fstream>

// #define M 512       // Lenna width
// #define N 512       // Lenna height
#define M 941       // VR width
#define N 704       // VR height
#define C 3         // Colors
#define OFFSET 15   // Header length

//writeToCSV
//help function to write stuff to csv
void writeRecordToFile(std::string filename, int fieldOne, int fieldTwo, float fieldThree)
{
    std::ofstream file;
    file.open(filename, std::ios_base::app);
    file << fieldOne << "," << fieldTwo << "," << fieldThree << std::endl;
    file.close();
}

uint8_t* get_image_array(void)
{
    // Try opening the file
    FILE* imageFile;
    imageFile = fopen("./input_image.ppm", "rb");
    if (imageFile == NULL) {
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }

    // Initialize empty image array
    uint8_t* image_array = (uint8_t*)malloc(M * N * C * sizeof(uint8_t) + OFFSET);

    // Read the image
    fread(image_array, sizeof(uint8_t), M * N * C * sizeof(uint8_t) + OFFSET, imageFile);

    // Close the file
    fclose(imageFile);

    // Move the starting pointer and return the flattened image array
    return image_array + OFFSET;
}

void save_image_array(uint8_t* image_array, char* filename)
{
    // Try opening the file
    FILE* imageFile;
    imageFile = fopen(filename, "wb");
    if (imageFile == NULL) {
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }

    // Configure the file
    fprintf(imageFile, "P6\n");               // P6 filetype
    fprintf(imageFile, "%d %d\n", M, N);      // dimensions
    fprintf(imageFile, "255\n");              // Max pixel

    // Write the image
    fwrite(image_array, 1, M * N * C, imageFile);

    // Close the file
    fclose(imageFile);
}

void save_image_grey(uint8_t* image_array, char* filename)
{
    // Try opening the file
    FILE* imageFile;
    imageFile = fopen(filename, "wb");
    if (imageFile == NULL) {
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }

    // Configure the file
    fprintf(imageFile, "P5\n");               // P5 filetype
    fprintf(imageFile, "%d %d\n", M, N);      // dimensions
    fprintf(imageFile, "255\n");              // Max pixel

    // Write the image
    fwrite(image_array, 1, M * N, imageFile);

    // Close the file
    fclose(imageFile);
}

//function to convert RGBRGB to RR..RGG..GBB..B array
void convertToRRGGBB(uint8_t* input, uint8_t* output, int numPixels)
{
    for (int i = 0; i < numPixels; i++)
    {
        output[i] = input[(3*i)];
        output[numPixels + i] = input[(3 * i) + 1];
        output[(2*numPixels) + i] = input[(3 * i) + 2];
    }
}

//function to convert RR..RGG..GBB..B to RGBRGB array
void convertToRGBRGB(uint8_t* input, uint8_t* output, int numPixels)
{
    for (int i = 0; i < numPixels; i++)
    {
        output[(3 * i)] = input[i];
        output[(3 * i) + 1] = input[numPixels + i];
        output[(3 * i) + 2] = input[(2 * numPixels) + i];
    }
}

__global__ void greyScaleCoalescedKernel(uint8_t* input, uint8_t* output, int numElements, int stride)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x; // + 3072 every iteration
    while (i < numElements)
    {
        output[i] = (input[3 * i] + input[(3 * i) + 1] + input[(3 * i) + 2]) / 3;
        i += stride;
    }
}


void greyScaleCoalescedFunction(uint8_t* image_array, uint8_t* new_image_array)
{
    // Allocate the device input nd output vector
    uint8_t* gpuA = NULL;
    cudaMalloc((uint8_t**)&gpuA, N * M * C * sizeof(uint8_t));

    uint8_t* gpuB = NULL;
    cudaMalloc((uint8_t**)&gpuB, N * M * sizeof(uint8_t));

    // Copy the host input vector and output int in host memory to the device input
    cudaMemcpy(gpuA, image_array, (N * M * C * sizeof(uint8_t)), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuB, new_image_array, (N * M * sizeof(uint8_t)), cudaMemcpyHostToDevice);

    int threadsPerBlock = 64;
    int blocksPerGrid = 48;
    greyScaleCoalescedKernel << <blocksPerGrid, threadsPerBlock >> > (gpuA, gpuB, N * M, threadsPerBlock * blocksPerGrid);

    cudaMemcpy(new_image_array, gpuB, (N * M * sizeof(uint8_t)), cudaMemcpyDeviceToHost);

    // Free up memory from GPU
    cudaFree(gpuA);
    cudaFree(gpuB);
}

//non-CoalescedCoalesced greyscale kernel
__global__ void greyScaleBasicKernel(uint8_t* input, uint8_t* output, int numElements, int stride)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x; // + 3072 every iteration
    while (i < numElements)
    {
        output[i] = ((input[i] + input[i + numElements] + input[(2*numElements) + i]) / 3);
        i += stride;
    }
}

//function that uses basic greyscale kernel
void greyScaleBasicFunction(uint8_t* image_array, uint8_t* new_image_array)
{
    // Allocate the device input nd output vector
    uint8_t* gpuA = NULL;
    cudaMalloc((uint8_t**)&gpuA, N * M * C * sizeof(uint8_t));

    uint8_t* gpuB = NULL;
    cudaMalloc((uint8_t**)&gpuB, N * M * sizeof(uint8_t));

    // Copy the host input vector and output int in host memory to the device input
    cudaMemcpy(gpuA, image_array, (N * M * C * sizeof(uint8_t)), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuB, new_image_array, (N * M * sizeof(uint8_t)), cudaMemcpyHostToDevice);

    int threadsPerBlock = 64;
    int blocksPerGrid = 48;
    greyScaleBasicKernel << <blocksPerGrid, threadsPerBlock >> > (gpuA, gpuB, N * M, threadsPerBlock * blocksPerGrid);

    cudaMemcpy(new_image_array, gpuB, (N * M * sizeof(uint8_t)), cudaMemcpyDeviceToHost);

    // Free up memory from GPU
    cudaFree(gpuA);
    cudaFree(gpuB);
}


int main()
{
    // Read the image
    uint8_t* image_array = get_image_array();

    //create RRGGBB format of the original image
    uint8_t* image_RRGGBB = (uint8_t*)malloc(M * N * C);
    convertToRRGGBB(image_array, image_RRGGBB, N * M);

    // Allocate memory for output average greyscale
    uint8_t* outBasic = (uint8_t*)malloc(M * N);
    uint8_t* outCoalesced = (uint8_t*)malloc(M * N);

    //basic gpu kernel for greyscale
    greyScaleBasicFunction(image_RRGGBB, outBasic);

    //test gpu kernel for Coalesced
    greyScaleCoalescedFunction(image_array, outCoalesced);

    //save image
    save_image_grey(outBasic, "./rrggbbgrey.ppm");
    save_image_grey(outCoalesced, "./coalescedGrey.ppm");

    free(image_array);
    free(image_RRGGBB);
    free(outCoalesced);
    free(outBasic);

    return 0;
}
