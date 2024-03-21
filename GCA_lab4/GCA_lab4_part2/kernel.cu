
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdint>      // Data types
#include <iostream>     // File operations

//there are two matrices: A & B
// dimensions of A are NxM
// dimensions of B are MxO
// to use the rolling window kernels, the values of N, M and O should be multiples of WINDOW
#define N 2
#define M 6
#define O 2
#define WINDOW 2

//shared memory kernel
__global__ void multiplyMatrixKernelSharedRollingWindow(int* C, const int* A, const int* B)
{
    __shared__ int sharedA[WINDOW * WINDOW], sharedB[WINDOW * WINDOW], sharedC[WINDOW*WINDOW];

    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    const int windowRow = index / WINDOW;
    const int windowCol = index % WINDOW;

    int row = windowRow, col = windowCol;

    for (int r = 0; r < N / WINDOW; r++)
    {
        for (int c = 0; c < O / WINDOW; c++)
        {
            sharedC[index] = 0;
            for (int i = 0; i < M / WINDOW; i++)
            {
                sharedA[index] = A[row * M + windowCol + (WINDOW * i)];
                sharedB[index] = B[windowRow * O + col + 2 * (WINDOW * WINDOW * i)];
                __syncthreads();

                int a = windowRow * WINDOW;
                int b = windowCol;
                for (int j = 0; j < WINDOW; j++)
                {
                    sharedC[index] += sharedA[a] * sharedB[b];
                    a++;
                    b += WINDOW;
                }
                __syncthreads();
            }
            C[row*M + col] = sharedC[index];
            col += WINDOW;
        }
        row += WINDOW;
        col = windowCol;
    }
}

//function that call kernel with shared memory and rolling window
void multiplyMatrixFunctionSharedRollingWindow(int* A, int* B, int* C)
{
    // Allocate the device input and output matrices
    int* gpuA = NULL;
    cudaMalloc((int**)&gpuA, N * M * sizeof(int));
    int* gpuB = NULL;
    cudaMalloc((int**)&gpuB, M * O * sizeof(int));
    int* gpuC = NULL;
    cudaMalloc((int**)&gpuC, N * O * sizeof(int));

    // Copy the host input and output matrices to the alocated memory on the gpu
    cudaMemcpy(gpuA, A, (N * M * sizeof(int)), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuB, B, (M * O * sizeof(int)), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuC, C, (N * O * sizeof(int)), cudaMemcpyHostToDevice);

    int threadsPerBlock = WINDOW * WINDOW;
    int blocksPerGrid = 1;
    multiplyMatrixKernelSharedRollingWindow << <blocksPerGrid, threadsPerBlock >> > (gpuC, gpuA, gpuB);

    cudaMemcpy(C, gpuC, (N * O * sizeof(int)), cudaMemcpyDeviceToHost);

    // Free up memory from GPU
    cudaFree(gpuA);
    cudaFree(gpuB);
    cudaFree(gpuC);
}

//shared memory kernel
__global__ void multiplyMatrixKernelSharedWindow(int* C, const int* A, const int* B)
{
    __shared__ int sharedA[WINDOW * WINDOW], sharedB[WINDOW * WINDOW], sharedC[WINDOW * WINDOW];

    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    const int row = index / O;// *M;
    const int col = index % O;
    sharedC[index] = 0;

    for (int i = 0; i < (M / WINDOW); i++)
    {
        // Load data into shared memory using a rolling window
        sharedA[index] = A[row * M + col + (WINDOW*i)];
        sharedB[index] = B[index + (WINDOW*WINDOW*i)];
        __syncthreads();

        int a = row * WINDOW;
        int b = col;

        // Perform matrix multiplication using shared memory data
        for (int j = 0; j < WINDOW; j++)
        {
            sharedC[index] += sharedA[a] * sharedB[b];
            a++;
            b += WINDOW;
        }
        __syncthreads();  // Make sure all threads finish using shared memory before loading new data
    }
    C[index] = sharedC[index];
}

//function that call kernel with shared memory and rolling window
void multiplyMatrixFunctionSharedWindow(int* A, int* B, int* C)
{
    // Allocate the device input and output matrices
    int* gpuA = NULL;
    cudaMalloc((int**)&gpuA, N * M * sizeof(int));
    int* gpuB = NULL;
    cudaMalloc((int**)&gpuB, M * O * sizeof(int));
    int* gpuC = NULL;
    cudaMalloc((int**)&gpuC, N * O * sizeof(int));

    // Copy the host input and output matrices to the alocated memory on the gpu
    cudaMemcpy(gpuA, A, (N * M * sizeof(int)), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuB, B, (M * O * sizeof(int)), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuC, C, (N * O * sizeof(int)), cudaMemcpyHostToDevice);

    int threadsPerBlock = WINDOW*WINDOW;
    int blocksPerGrid = 1;
    multiplyMatrixKernelSharedWindow << <blocksPerGrid, threadsPerBlock >> > (gpuC, gpuA, gpuB);

    cudaMemcpy(C, gpuC, (N * O * sizeof(int)), cudaMemcpyDeviceToHost);

    // Free up memory from GPU
    cudaFree(gpuA);
    cudaFree(gpuB);
    cudaFree(gpuC);
}

//shared memory kernel
__global__ void multiplyMatrixKernelShared(int* C, const int* A, const int* B)
{
    __shared__ int sharedA[N*M], sharedB[M*O];

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int row = (index / O) * M;
    int col = index % O;
    int val = 0;

    sharedA[index] = A[index];
    sharedB[index] = B[index];
    __syncthreads();

    for (int i = 0; i < M; i++)
    {
        val += sharedA[row] * sharedB[col];
        row++;
        col += O;
    }
    C[index] = val;
}

//function that call kernel with shared memory
void multiplyMatrixFunctionShared(int* A, int* B, int* C)
{
    // Allocate the device input and output matrices
    int* gpuA = NULL;
    cudaMalloc((int**)&gpuA, N * M * sizeof(int));
    int* gpuB = NULL;
    cudaMalloc((int**)&gpuB, M * O * sizeof(int));
    int* gpuC = NULL;
    cudaMalloc((int**)&gpuC, N * O * sizeof(int));

    // Copy the host input and output matrices to the alocated memory on the gpu
    cudaMemcpy(gpuA, A, (N * M * sizeof(int)), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuB, B, (M * O * sizeof(int)), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuC, C, (N * O * sizeof(int)), cudaMemcpyHostToDevice);

    int threadsPerBlock = N * O;
    int blocksPerGrid = 1;
    multiplyMatrixKernelShared << <blocksPerGrid, threadsPerBlock >> > (gpuC, gpuA, gpuB);

    cudaMemcpy(C, gpuC, (N * O * sizeof(int)), cudaMemcpyDeviceToHost);

    // Free up memory from GPU
    cudaFree(gpuA);
    cudaFree(gpuB);
    cudaFree(gpuC);
}

//basic kernel for multiplication
__global__ void multiplyMatrixKernelBasic(int *C, const int *A, const int *B)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int row = (index / O) * M;
    int col = index % O;
    int val = 0;
    for (int i = 0; i < M; i++)
    {
        val += A[row] * B[col];
        row++;
        col += O;
    }
    C[index] = val;
}

//function that call basic multiplication kernel
void multiplyMatrixFunctionBasic(int* A, int* B, int* C)
{
    // Allocate the device input and output matrices
    int* gpuA = NULL;
    cudaMalloc((int**)&gpuA, N * M * sizeof(int));
    int* gpuB = NULL;
    cudaMalloc((int**)&gpuB, M * O * sizeof(int));
    int* gpuC = NULL;
    cudaMalloc((int**)&gpuC, N * O * sizeof(int));

    // Copy the host input and output matrices to the alocated memory on the gpu
    cudaMemcpy(gpuA, A, (N * M * sizeof(int)), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuB, B, (M * O * sizeof(int)), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuC, C, (N * O * sizeof(int)), cudaMemcpyHostToDevice);

    int threadsPerBlock = N*O;
    int blocksPerGrid = 1;
    multiplyMatrixKernelBasic << <blocksPerGrid, threadsPerBlock >> > (gpuC, gpuA, gpuB);

    cudaMemcpy(C, gpuC, (N * O * sizeof(int)), cudaMemcpyDeviceToHost);

    // Free up memory from GPU
    cudaFree(gpuA);
    cudaFree(gpuB);
    cudaFree(gpuC);
}

void fillArrayWithRandomValues(int* arr, int size) 
{
    for (int i = 0; i < size; i++)
    {
        arr[i] = i;
    }
}


int main()
{
    //allocate memory for matrices (input and output)
    size_t sizeA = N * M * sizeof(int); //amount of bytes
    int* A = (int*)malloc(sizeA); //memory alocation
    size_t sizeB = O * M * sizeof(int); //amount of bytes
    int* B = (int*)malloc(sizeB); //memory alocation
    size_t sizeC = N * O * sizeof(int); //amount of bytes
    int* C = (int*)malloc(sizeC); //memory alocation
    int* D = (int*)malloc(sizeC); //memory alocation
    int* E = (int*)malloc(sizeC);

    //fill up input matrices with random values
    fillArrayWithRandomValues(A, N * M);
    fillArrayWithRandomValues(B, M * O);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            printf("A - element %d,%d: %d\n", i, j, A[i * M + j]);
        }
    }

    printf("\n");

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < O; j++)
        {
            printf("B - element %d,%d: %d\n", i, j, B[i * O + j]);
        }
    }

    printf("\n");


    //multiply with basic kernel
    multiplyMatrixFunctionBasic(A, B, C);

    multiplyMatrixFunctionSharedRollingWindow(A, B, E);
    
        for (int i = 0; i < N * O; i++)
        {
            printf("C[%d] = %d\tE[%d] = %d\n", i, C[i], i, E[i]);
        }
    

    //free up memory
    free(A);
    free(B);
    free(C);
    free(D);
    free(E);
}
