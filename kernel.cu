#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include <stdlib.h>
#include "kernel.cuh"

using namespace std;

/*
// Taken from: Graph Algorithms in the Language of Linear Algebra page 19 by (Charles M. Rader)
 The implementation is in this form:
 D = I + A + A^2 + A^3 + A^4 +···
    I = Identity matrix
    A = Input Matrix

 SCC = D ∧ transpose(D)
    SCC = Is a matrix with the strongly connected Components identified with 1.
    ∧ = Bit-wise and operation
*/

const int TILE_DIM = 32;

__global__ void mainKernelNaiveLoop(int* I, int* A, int* B, int* C, int* OUT, const int N) {
    const int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    const int COL = blockIdx.x * blockDim.x + threadIdx.x;

    // Avoid accesing data beyond the end of the arrays.
    // See: Cuda by Example page 44ff
    if (ROW < N && COL < N)
    {
        // Calculate A^n + A^(n+1)
        matrixAddNaive(OUT, A, N, ROW, COL);
        /*
        * Calculate A^n * A^(n+1): Matrix A represents A^n
        * Matrix B represents the input matrix
        * Matrix C represents the output of this matrix multiplication
        */
        matrixMultiplicationNaive(A, B, C, N, ROW, COL);
        /*
        * Copy A^n (which is matrix C) to A
        * That means, after this copy operation we have two identical matrices: A and C which contain both A^n+1
        * This is needed in the next operation "matrixAddNaive" to sum up the results of A^n with A^n+1
        */
        matrixCopyNaive(A, C, N, ROW, COL);
        //matrixCopyNaive(OUT, C, N, ROW, COL);
    }
}

__global__ void mainKernelNaiveEnd(int* I, int* A, int* B, int* C, int* OUT, const int N) {

    const int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    const int COL = blockIdx.x * blockDim.x + threadIdx.x;

    // Avoid accesing data beyond the end of the arrays.
    // See: Cuda by Example page 44ff
    if (ROW < N && COL < N)
    {
        /*
        * Because we terminate the first kernel without doing the last sum, we have to do it here.
        * The last sum would be A^n + A^n+1
        */
        matrixAddNaive(OUT, A, N, ROW, COL);

        // Add identity matrix to result
        matrixAddNaive(OUT, I, N, ROW, COL);

        // Add transpose the result and write the transposed matrix back to A --> A will be overriden here
        matrixTransposeNaive(A, OUT, N, ROW, COL);
        // Do elementwise AND operation with matrix OUT and A. Two integers > 0 will return TRUE which is encoded in the result matrix as 1 (FALSE = 0)
        matrixBitAndNaive(OUT, A, N, ROW, COL);
    }
}

__device__ void matrixAddNaive(int* OUT, int* A, const int N, const int ROW, const int COL) {
    OUT[ROW * N + COL] = OUT[ROW * N + COL] + A[ROW * N + COL];
}

__device__ void matrixMultiplicationNaive(int* A, int* B, int* C, const int N, const int ROW, const int COL) {

    unsigned int tmpSum = 0;
    // each thread computes one element of the block sub-matrix
    for (int i = 0; i < N; i++) {
        tmpSum += A[ROW * N + i] * B[i * N + COL];
    }

    // Take care of overflows 
    if (tmpSum != 0)
        tmpSum = 1;

    C[ROW * N + COL] = tmpSum;
}

__device__ void matrixCopyNaive(int* A, int* C, const int N, const int ROW, const int COL) {
    A[ROW * N + COL] = C[ROW * N + COL];

    // Example to get the threadIDs. Needed for debugging purpose only
    //A[ROW * N + COL] = ROW * 10 + COL;
}

__device__ void matrixTransposeNaive(int* A, int* C, const int N, const int ROW, const int COL) {
    A[ROW * N + COL] = C[COL * N + ROW];
}

__device__ void matrixBitAndNaive(int* A, int* C, const int N, const int ROW, const int COL)
{
    if (A[ROW * N + COL] && C[ROW * N + COL])
        A[ROW * N + COL] = 1;
    else
        A[ROW * N + COL] = 0;
}

__host__ void sccNaive(int* I, int* A, int* B, int* C, int* OUT, const int N, const int POW, const int MaxThreadsPerBlock) {

    // declare the number of blocks per grid and the number of threads per block
    // use 1 to MaxThreadsPerBlock threads per block
    dim3 blocksPerGrid(1, 1);
    dim3 threadsPerBlock(32, 32);

    if (N * N > MaxThreadsPerBlock)
    {
        if (MaxThreadsPerBlock == 1024)
        {
            threadsPerBlock.x = 32;
            threadsPerBlock.y = 32;

            // See the meaning of (N + (MaxThreadsPerBlock - 1)) in "Cuda by Example" page 64-65
            blocksPerGrid.x = (N + (32 - 1)) / threadsPerBlock.x;
            blocksPerGrid.y = (N + (32 - 1)) / threadsPerBlock.y;
        }
    }

    for (int i = 0; i < POW; i++) {
        mainKernelNaiveLoop << <blocksPerGrid, threadsPerBlock >> > (I, A, B, C, OUT, N);
        //cudaDeviceSynchronize();   // Is cudaDeviceSynchronize necessary?
    }
    mainKernelNaiveEnd << <blocksPerGrid, threadsPerBlock >> > (I, A, B, C, OUT, N);
}

__global__ void mainKernelSharedLoop(int* I, int* A, int* B, int* C, int* OUT, const int N) {

    // Row and column for tiled matrix multiplication with shared memory.
    // blockIdx.x and blockIdx.y must be multiplyied with the dimension of the tile (TILE_DIM)
    int ROW_Shared = blockIdx.y * TILE_DIM + threadIdx.y;
    int COL_Shared = blockIdx.x * TILE_DIM + threadIdx.x;

    // Row and column for rest of the calculations
    const int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    const int COL = blockIdx.x * blockDim.x + threadIdx.x;

    // Avoid accesing data beyond the end of the arrays.
    // See: Cuda by Example page 44ff

    if (ROW < N && COL < N)
        matrixAddNaive(OUT, A, N, ROW, COL);

    if (ROW_Shared < N && COL_Shared < N)
        matrixMultiplicationTiled(A, B, C, N, ROW_Shared, COL_Shared);

    if (ROW < N && COL < N)
        matrixCopyNaive(A, C, N, ROW, COL);
}

// Copy coalesced
// Taken from: https://devblogs.nvidia.com/efficient-matrix-transpose-cuda-cc/
// No advantages could be observed so far compared to the naive copy we implemented
__device__ void copyCoalesced(int* odata, const int* idata)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int width = gridDim.x * blockDim.x;

    for (int j = 0; j < blockDim.x; j += blockDim.x)
        odata[(y + j) * width + x] = idata[(y + j) * width + x];
}

__global__ void mainKernelSharedEnd(int* I, int* A, int* B, int* C, int* OUT, const int N) {

    const int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    const int COL = blockIdx.x * blockDim.x + threadIdx.x;

    // Avoid accesing data beyond the end of the arrays.
    // See: Cuda by Example page 44ff
    if (ROW < N && COL < N)
    {
        /*
        * Because we terminate the first kernel without doing the last sum, we have to do it here.
        * The last sum would be A^n + A^n+1
        */
        matrixAddNaive(OUT, A, N, ROW, COL);

        // Add identity matrix to result
        matrixAddNaive(OUT, I, N, ROW, COL);

        // Add transpose the result and write the transposed matrix back to A --> A will be overriden here
        matrixTransposeNaive(A, OUT, N, ROW, COL);
        // Do elementwise AND operation with matrix OUT and A. Two integers > 0 will return TRUE which is encoded in the result matrix as 1 (FALSE = 0)
        matrixBitAndNaive(OUT, A, N, ROW, COL);
    }
}

__device__ void matrixMultiplicationTiled(int* A, int* B, int* C, const int N, const int ROW, const int COL)
{
    float interimResult = 0;

    // Reserve shared memory with fixed tile dimension
    __shared__ float A_shared[TILE_DIM][TILE_DIM];
    __shared__ float B_shared[TILE_DIM][TILE_DIM];

    // Outer loop of of the muliplication
    // Tile sweeps over the whole matrix
    for (int k = 0; k < (TILE_DIM + N - 1) / TILE_DIM; k++) {

        if (k * TILE_DIM + threadIdx.x < N && ROW < N)
            A_shared[threadIdx.y][threadIdx.x] = A[ROW * N + k * TILE_DIM + threadIdx.x];
        else
            A_shared[threadIdx.y][threadIdx.x] = 0.0;

        if (k * TILE_DIM + threadIdx.y < N && COL < N)
            B_shared[threadIdx.y][threadIdx.x] = B[(k * TILE_DIM + threadIdx.y) * N + COL];
        else
            B_shared[threadIdx.y][threadIdx.x] = 0.0;

        // Sync barrier
        __syncthreads();

        // Inner loop for doing the actual multiplication over single stripes in tile
        for (int n = 0; n < TILE_DIM; ++n)
            interimResult += A_shared[threadIdx.y][n] * B_shared[n][threadIdx.x];

        // Sync barrier
        __syncthreads();
    }

    // Take care of overflows 
    if (interimResult != 0)
        interimResult = 1;

    // Assign the interim results to the output matrix C with respect to ROW and COL
    if (ROW < N && COL < N)
        C[((blockIdx.y * blockDim.y + threadIdx.y) * N) + (blockIdx.x * blockDim.x) + threadIdx.x] = interimResult;
}

__host__ void sccShared(int* I, int* A, int* B, int* C, int* OUT, const int N, const int POW, const int MaxThreadsPerBlock) {

    // declare the number of blocks per grid and the number of threads per block
    // use 1 to MaxThreadsPerBlock threads per block
    dim3 blocksPerGrid(1, 1);
    dim3 threadsPerBlock(32, 32);

    if (N * N > MaxThreadsPerBlock) {
        if (MaxThreadsPerBlock == 1024)
        {
            threadsPerBlock.x = 32;
            threadsPerBlock.y = 32;
            // See the meaning of (N + (MaxThreadsPerBlock - 1)) in "Cuda by Example" page 64-65
            blocksPerGrid.x = (N + (32 - 1)) / threadsPerBlock.x;
            blocksPerGrid.y = (N + (32 - 1)) / threadsPerBlock.y;
        }
    }

    //size_t SHMEM = N * sizeof(int);
    for (int i = 0; i < POW; i++) {
        //mainKernelSharedLoop<<<blocksPerGrid, threadsPerBlock, 2 * (30*30*sizeof(int))>>>(I, A, B, C, OUT, N);
        mainKernelSharedLoop << <blocksPerGrid, threadsPerBlock >> > (I, A, B, C, OUT, N);
        cudaDeviceSynchronize(); // Is cudaDeviceSynchronize necessary?
    }
    mainKernelSharedEnd << <blocksPerGrid, threadsPerBlock >> > (I, A, B, C, OUT, N);
}