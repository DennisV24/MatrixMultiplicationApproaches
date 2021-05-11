#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "cuda_device_runtime_api.h"
#include "device_functions.h"
#include "cuda_runtime_api.h"

#include <stdio.h>
constexpr auto blockSize = 8;
constexpr auto size = 64;

cudaError_t matrixMultWithCuda(int*, int*, int*, int size);

__global__ void kMatrixMult(int *dA, int *dB, int *dC, int size)
{
    __shared__ int shA[blockSize][blockSize];
    __shared__ int shB[blockSize][blockSize];
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    int cArrIdx = y * size + x;
    int cVal = 0;
    int numSubBlocks = size / blockSize;
    for (unsigned int i = 0; i < numSubBlocks; i++) {
        auto aIdx = y * size + (i * blockSize + threadIdx.x);
        auto bIdx = x + size * (i * blockSize + threadIdx.y);
        shA[threadIdx.y][threadIdx.x] = dA[aIdx];
        shB[threadIdx.y][threadIdx.x] = dB[bIdx];
        __syncthreads();
        for (unsigned int k = 0; k < blockSize; k++) {
            auto sharedAVal = shA[threadIdx.y][k];
            auto sharedBVal = shB[k][threadIdx.x];
            cVal += sharedAVal * sharedBVal;
            __syncthreads();
        }
    }
    dC[cArrIdx] = cVal;
}

int main()
{
    int* A;
    int* B;
    int* C;
    A = new int[size*size];
    B = new int[size*size];
    C = new int[size*size];
    for (unsigned int i = 0; i < size; i++) {
        for (unsigned int j = 0; j < size; j++) {
            A[i * size + j] = (i * 2 + j / 2) % 128;
            B[i * size + j] = (i / 2 + j * 2) % 128;
        }
    }

    cudaError_t cudaStatus = matrixMultWithCuda(A, B, C, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "matrixMultWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    cudaDeviceSynchronize();
    printf("Matrix Multiplication Success\n");
    // for external checking
    printf("\n\n MATRIX A: \n\n");
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            printf("%d ", A[i*size + j]);
        }
        printf("\n");
    }
    printf("\n\n MATRIX B: \n\n");
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            printf("%d ", B[i*size + j]);
        }
        printf("\n");
    }
    printf("\n\n MATRIX C: \n\n");
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            printf("%d ", C[i*size + j]);
        }
        printf("\n");
    }
	return 0;
}

cudaError_t matrixMultWithCuda(int *A, int *B, int *C,int size)
{
    int* dA;
    int* dB;
    int* dC;
    cudaEvent_t startEvent;
    cudaEvent_t finishEvent;
    cudaError_t cudaStatus;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&finishEvent);
    cudaEventRecord(startEvent);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dA, size * size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dB, size * size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dC, size * size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dA, A, size *  size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dB, B, size *  size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    dim3 dimGrid(size/blockSize, size/blockSize);
    dim3 dimBlock(blockSize, blockSize);
    kMatrixMult<<<dimGrid, dimBlock>>>(dA, dB, dC, size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(C, dC, size * size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaEventRecord(finishEvent);
    cudaEventSynchronize(finishEvent);
    float timePassed;
    cudaEventElapsedTime(&timePassed, startEvent, finishEvent);
    printf("Time diff: %f ms\n", timePassed);

Error:
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(finishEvent);
    
    return cudaStatus;
}
