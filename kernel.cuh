#ifndef KERNEL_CUH_
#define KERNEL_CUH_

inline __global__ void mainKernelNaive(int *I, int *A, int *B, int *C, int *OUT, const int N, const int POW);
inline __device__ void matrixMultiplicationNaive(int* A, int* B, int* C, const int N, const int ROW, const int COL);
inline __device__ void matrixCopyNaive(int *A, int *C, const int N, const int ROW, const int COL);
inline __device__ void matrixAddNaive(int *OUT, int *A, const int N, const int ROW, const int COL);
inline __device__ void matrixTransposeNaive(int *A, int *C, const int N, const int ROW, const int COL);
inline __device__ void matrixBitAndNaive(int *A, int *C, const int N, const int ROW, const int COL);

inline __global__ void mainKernelNaiveLoop(int *I, int *A, int *B, int *C, int *OUT, const int N);
inline __global__ void mainKernelNaiveEnd(int *I, int *A, int *B, int *C, int *OUT, const int N);

inline __device__ void matrixMultiplicationTiled(int* A, int* B, int* C, const int N, const int ROW, const int COL);
inline __device__ void matrixMulShared(const int *a, const int *b, int *c, const int N);
inline __device__ void copyCoalesced(int *odata, const int *idata);

inline void sccNaive(int* d_I, int* d_A, int* d_B, int* d_C, int* d_OUT, int N, int POW, int MaxThreadsPerBlock);
inline void sccShared(int* d_I, int* d_A, int* d_B, int* d_C, int* d_OUT, int N, int POW, int MaxThreadsPerBlock);

inline __global__ void mainKernelSharedLoop(int* I, int* A, int* B, int* C, int* OUT, const int N);
inline __global__ void mainKernelSharedEnd(int* I, int* A, int* B, int* C, int* OUT, const int N);
void sccShared(int *d_I, int *d_A, int *d_B, int *d_C, int *d_OUT, int N, int POW, int MaxThreadsPerBlock);

#endif // KERNEL_CUH
