
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define TILE 32
#define N 32
#define M 32
#define O 32

__constant__ int cA[TILE*TILE], cB[TILE*TILE];

//HELPER FUNCTION TO CREATE ARRAY OF LENGTH N WHERE VALUE i = i
int* getRandomArray(int n)
{
    size_t size = n * sizeof(int);
    int* A = (int*)malloc(size);
    for (int i = 0; i < n; i++)
    {
        A[i] = i;
    }
    return A;
}
//END HELPER FUNCTION



//GENERIC PART
//CODE FOR TILING MULTIPLICATION ON GPU, WORKS WITH GLOBAL, SHARED AND CONSTANT MEMORY

//multiply matrix A & B and put write the result in C
//this function can be used to multiply whole matrices or parts/tiles
//this function will also work for shared/constant memory
//HOW TO USE:
//mak sure A & B are filled with the correct data
//call the function
//read the resulting values in C
//row & col are coordinates of th value in the resulting matrix
//row = thread id / width of C (integer division)
//col = thread id % width of C
__device__ void multiplyMatrix(int* C, int* A, int* B)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x; //thread index
    const int r = index / TILE; //row in tile
    const int c = index % TILE; //colum in tile

    int a = r * TILE;
    int b = c;
    int val = 0;

    for (int i = 0; i < TILE; i++)
    {
        val += A[a] * B[b];
        a++;
        b += TILE;
    }
    C[r * TILE + c] += val;
}

//copy a tile to shared memory
//parameters: 
//sA = target
//A = source
//row = row index of the desired tile
//col = colum index of the desired tile
//width = width of the source matrix A
__device__ void copyToShared(int* sA, int* A, int row, int col, int width)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x; //thread index
    const int r = index / TILE; //row in TILE
    const int c = index % TILE; //colum in TILE
    int i = r * width + c; //index in global memory

    sA[index] = A[i + row*TILE*width + col*TILE];
}

//multiply a complete tile in the output matrix C (using tiling)
//parameters:
//C = target matrix (A*B = C)
//A = source matrix A for the A*B operation
//B = source matrix B for the A*B operation
//sC = space in shared memory for C
//sA = space in shared memory for A
//sB = space in shared memory for B
//row = row index of the desired tile
//col = colum index of the desired tile
__device__ void multiplyTile(int* C, int* A, int* B, int* sC, int* sA, int* sB, int row, int col)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x; //thread index
    const int r = index / TILE; //row within TILE
    const int c = index % TILE; //colum within TILE
    int i = r * O + c; //index in C without offset of row and col

    __syncthreads();
    sC[index] = 0;
    for (int i = 0; i < M / TILE; i++)
    {
        copyToShared(sA, A, row, i, M);
        copyToShared(sB, B, i, col, O);
        __syncthreads();
        multiplyMatrix(sC, sA, sB);
        __syncthreads();
    }
    
    C[i + row * TILE * O + col * TILE] = sC[index];
}
//END OF GENERIC PART



//CONSTANT MEMORY
//execute function and GPU kernel for constant memory multiplication

//multiply two matrices using tiling in constant memory (since c is not constant shared memory is used)
//function will iterate over all the tiles in the target matric C
//and call the multiplyTile function with the corresponding row and colum indexes
//parameters:
//C = target matrix (A*B = C)
//A = source matrix A for the A*B operation
//B = source matrix B for the A*B operation
__global__ void multiplyKernelConstant(int* C, int* A, int* B)
{
    __shared__ int sharedC[TILE * TILE];
    for (int i = 0; i < N / TILE; i++)
    {
        for (int j = 0; j < O / TILE; j++)
        {
            multiplyTile(C, A, B, sharedC, cA, cB, i, j);
        }
    }
}

//this code executes on CPU
//this function executes the GPU kernel to multiply matrix A and B into C
//using constant memory
void multiplyConstant(int* C, int* A, int* B)
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

    int threadsPerBlock = TILE * TILE;
    int blocksPerGrid = 1;
    multiplyKernelConstant << <blocksPerGrid, threadsPerBlock >> > (gpuC, gpuA, gpuB);

    cudaMemcpy(C, gpuC, (N * O * sizeof(int)), cudaMemcpyDeviceToHost);

    // Free up memory from GPU
    cudaFree(gpuA);
    cudaFree(gpuB);
    cudaFree(gpuC);
}

void executeConstant()
{
    int* A = getRandomArray(N * M);
    int* B = getRandomArray(M * O);
    int* C = (int*)malloc(N * O * sizeof(int));
    multiplyConstant(C, A, B);
    //free up memory
    free(A);
    free(B);
    free(C);
}
//END OF CONSTANT MEMORY



//SHARED MEMORY
//execute function and GPU kernel for shared memory multiplication

//multiply two matrices using tiling in shared memory
//function will iterate over all the tiles in the target matric C
//and call the multiplyTile function with the corresponding row and colum indexes
//parameters:
//C = target matrix (A*B = C)
//A = source matrix A for the A*B operation
//B = source matrix B for the A*B operation
__global__ void multiplyKernelShared(int* C, int* A, int* B)
{
    __shared__ int sharedA[TILE * TILE], sharedB[TILE * TILE], sharedC[TILE * TILE];
    for (int i = 0; i < N / TILE; i++)
    {
        for (int j = 0; j < O / TILE; j++)
        {
            multiplyTile(C, A, B, sharedC, sharedA, sharedB, i, j);
        }
    }
}

//this code executes on CPU
//this function executes the GPU kernel to multiply matrix A and B into C
//using shared memory
void multiplyShared(int* C, int* A, int* B)
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

    int threadsPerBlock = TILE * TILE;
    int blocksPerGrid = 1;
    multiplyKernelShared << <blocksPerGrid, threadsPerBlock >> > (gpuC, gpuA, gpuB);

    cudaMemcpy(C, gpuC, (N * O * sizeof(int)), cudaMemcpyDeviceToHost);

    // Free up memory from GPU
    cudaFree(gpuA);
    cudaFree(gpuB);
    cudaFree(gpuC);
}

void executeShared()
{
    int* A = getRandomArray(N * M);
    int* B = getRandomArray(M * O);
    int* C = (int*)malloc(N * O * sizeof(int));
    multiplyShared(C, A, B);
    free(A);
    free(B);
    free(C);
}
//END OF SHARED MEMORY



//GLOBAL MEMORY
//execute function and GPU kernel for shared memory multiplication

__global__ void multiplyKernelGlobal(int* C, const int* A, const int* B)
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

void multiplyGlobal(int* C, int* A, int* B)
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

    int threadsPerBlock = 1024;
    int blocksPerGrid = (N*O + threadsPerBlock - 1) / threadsPerBlock;
    multiplyKernelGlobal << <blocksPerGrid, threadsPerBlock >> > (gpuC, gpuA, gpuB);

    cudaMemcpy(C, gpuC, (N * O * sizeof(int)), cudaMemcpyDeviceToHost);

    // Free up memory from GPU
    cudaFree(gpuA);
    cudaFree(gpuB);
    cudaFree(gpuC);
}

void executeGlobal()
{
    int* A = getRandomArray(N * M);
    int* B = getRandomArray(M * O);
    int* C = (int*)malloc(N * O * sizeof(int));
    multiplyGlobal(C, A, B);
    free(A);
    free(B);
    free(C);
}
//END OF GLOBAL MEMORY



//MAIN
int main()
{
    executeGlobal();
    executeShared();
    executeConstant();
    return 0;
}


//code to print array:
/*for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            printf("A - element %d,%d: %d\n", i, j, A[i * M + j]);
        }
    }
    printf("\n");*/