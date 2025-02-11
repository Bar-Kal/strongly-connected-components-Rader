#ifndef RADER_CUH_
#define RADER_CUH_

// Specify which memory strategy should be run
#define CPU 1
#define NAIVE 1
#define PINNED 0 
#define ZC 0
#define SHARED 0

//Prototypes
inline void cpu_printSCC(std::vector<int>& gpuResults, const int N);
inline void cpu_writeSCC(std::vector<int>& gpuResults, const int N, const string path);
inline std::vector<int> cpu_read_matrix(std::string path, int *N);
inline void cpu_clean_initialization_vectors(std::vector<int>& A, std::vector<int>& B, std::vector<int>& C);
inline int cpu_calculate_error(std::vector<int>& cpuResults, std::vector<int>& gpuResults, const int N);
inline void cpu_padding_matrix(std::vector<int>& inputMatrix, int *N);
inline void cpu_printMatrix(std::vector<int>& inputMatrix, const int N);
inline void cpu_matMul(std::vector<int>& A, std::vector<int>& B, std::vector<int>& C, const int N);
inline void cpu_sumMat(std::vector<int>& OUT, std::vector<int>& A, const int N);
inline void cpu_matTranspose(std::vector<int>& A, std::vector<int>& B, const int N);
inline void cpu_matBitAnd(std::vector<int>& A, std::vector<int>& B, const int N);

#endif