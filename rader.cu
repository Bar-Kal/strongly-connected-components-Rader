#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "kernel.cu"
#include "rader.cuh"
#include "gpuHelper.h"
#include <math.h>
#include "asprintf.h"
//#include "testSharedKernel.cu"

// --- Timing includes
// Timer implementations taken from: https://stackoverflow.com/questions/7876624/timing-cuda-operations
// Implemented from SO user: https://stackoverflow.com/users/1886641/jackolantern
#include "timing/TimingCPU.h"
#include "timing/TimingCPU.cpp"
#include "timing/TimingGPU.cu"
#include "timing/TimingGPU.cuh"

using namespace std;

int main(int argc, char* argv[])
{
    string argument1 = "";
    //string argument2 = "./output_1_732.txt";
    string argument2 = "";
    long int argument3 = 0;

    // Check the number of parameters
    //if (argc < 2) {
    //    // Tell the user how to run the program
    //    std::cerr << "Too few arguments.\nUsage: " << argv[0] << " path_input_file  [-w PATH_OUTPUT_FILE] [-s STEPS]" << std::endl;       
    //    std::cerr << "If the option [-w] is specified, the output will be written to this path." << std::endl; 
    //    std::cerr << "With the option [-s], the number of steps are specified. If not specified, number of steps = N (worst case)." << std::endl; 
    //    cout << argv[0] << "  " << argv[1] << std::endl;
    //    return 1;
    //}

    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "-w") 
        {
            if (i + 1 < argc)
            { // Make sure we aren't at the end of argv!
                argument2 = argv[++i]; // Increment 'i' so we don't get the argument as the next argv[i].
            }
            else //Argument -w specified but no path given
            {
                std::cerr << "-w option requires one argument." << std::endl;
                return 1;
            }  
        }
        if (std::string(argv[i]) == "-s") 
        {
            if (i + 1 < argc)
            { 
                char *endptr;
                argument3 = strtol(argv[++i], &endptr, 0);
            }
            else //Argument -s specified but no path given
            {
                std::cerr << "-s option requires one argument." << std::endl;
                return 1;
            }  
        }
        else
        {
            argument1 = argv[1];
        }
    }

    TimingCPU timer_CPU;
    TimingGPU timer_naive_GPU;
    TimingGPU timer_pinned;
    TimingGPU timer_zerocopy;
    TimingGPU timer_shared_GPU;

    // Get properties of GPU
    // Note! this will only get the properties of first GPU    
    cudaDeviceProp  deviceProp;
    cudaGetDeviceProperties( &deviceProp, 0 );

    std::cout << "Reading data file and creating identity matrix on CPU..." << std::endl;
    timer_CPU.StartCounter();

    //std::string file_path = argument1;
    std::string file_path = "./data/example_144";

    int N;    // Dimension of matrix (width or height, must be same) will be set in the function "cpu_read_matrix"
    int POW;
    int err_naive;
    int err_pinned;
    int err_zerocopy;
    int err_shared;

    // Read in the input matrix
    std::vector<int> h_A = cpu_read_matrix(file_path, &N);
    std::cout << "Print matrix CPU" << std::endl;
    cpu_printMatrix(h_A, N);

    // If N is not a multiple of 2 than do a padding and increment N by 1
    //cpu_padding_matrix(h_A, &N);

    if(argument3 != 0)
        POW = argument3;  // Will define how often loop over the input matrix. (D = I + A + A^2 + A^3 + A^4 +···)
    else
        POW = N;
    
    const int SIZE = N*N;
    const int MaxThreadsPerBlock = deviceProp.maxThreadsPerBlock;

    // If matrix is empty than either an error occured or the file was empty
    if(h_A.empty())
        return 1;

    // Create vectors on host
    // Vectors are sequence containers representing arrays that can change in size.
    std::vector<int> h_I(SIZE);
    std::vector<int> h_B(SIZE);
	std::vector<int> h_C(SIZE);
    std::vector<int> h_OUT(SIZE);
    std::vector<int> h_returnNaiveGPUOUT(SIZE);
    std::vector<int> h_returnSharedGPUOUT(SIZE);

    // Create idendity matrix on CPU  
    std::fill(h_I.begin(), h_I.end(), 0);
    for (int i=0; i < N*N; i += N + 1){
        h_I[i] = 1;
    }

/*****************************************************************************************/
/******************************Start CPU calculations*************************************/
/*****************************************************************************************/

#if CPU == 1
    std::cout << "CPU calculation started" << std::endl;

    // Initialize output matrix with 0's
    cpu_clean_initialization_vectors(h_B, h_C, h_OUT);

    // Copy h_A -> h_B. We will need it for the CPU calculations.
    h_B = h_A;
    // Different way to copy vectors
    //std::copy(std::begin(h_A), std::end(h_A), std::begin(h_B));
    
    // Now do the matrix multiplication on the CPU

    timer_CPU.StartCounter();
    for (int i = 0; i < POW; i++)
    {
        cpu_sumMat(h_OUT, h_A, N);
        cpu_matMul(h_A, h_B, h_C, N);
        h_A = h_C;
    }
    cpu_sumMat(h_OUT, h_A, N);
    cpu_sumMat(h_OUT, h_I, N);
    
    cpu_matTranspose(h_A, h_OUT, N);
    
    cpu_matBitAnd(h_OUT, h_A, N);

    //std::cout << "Print matrix CPU" << std::endl;
    cpu_printMatrix(h_OUT, N);

    if(argument2 != "")
        cpu_writeSCC(h_OUT, N, argument2);
    // Free Cuda
    cudaFree(0);

    std::cout << "CPU Timing = " << timer_CPU.GetCounter() << " ms" << std::endl;

#endif	

/*****************************************************************************************/
/******************************End CPU calculations***************************************/
/*****************************************************************************************/

/*****************************************************************************************/
/***************************Start Naive GPU calculations**********************************/
/*****************************************************************************************/
#if NAIVE == 1     

    // Overwrite matrices with 0's  
    cpu_clean_initialization_vectors(h_A, h_B, h_C);

    // Read in the input matrix
    h_A = cpu_read_matrix(file_path, &N);
    // If N is not a multiple of 2 than do a padding and increment N by 1
    //cpu_padding_matrix(h_A, &N);

    // If matrix is empty than either an error occured or the file was empty
    if(h_A.empty())
        return 1;

    std::cout << "CPU time = " << timer_CPU.GetCounter() << " ms" << std::endl;

    std::cout << "\nNaive GPU calculation started" << std::endl;

    // Allocate memory on the device
    gpuHelper<int> d_I(SIZE, 'M');
    gpuHelper<int> d_A(SIZE, 'M');
    gpuHelper<int> d_B(SIZE, 'M');
    gpuHelper<int> d_C(SIZE, 'M');
    gpuHelper<int> d_OUT(SIZE, 'M');

    // Copy from host to device on allocated memory
    d_I.set(&h_I[0], SIZE);
    d_A.set(&h_A[0], SIZE);
    d_B.set(&h_A[0], SIZE);
    d_C.set(&h_A[0], SIZE);
    d_OUT.set(&h_I[0], SIZE);

    // Start Naive-GPU timer
    timer_naive_GPU.StartCounter();

    // Call scc naive
    sccNaive(d_I.getData(), d_A.getData(), d_B.getData(), d_C.getData(), d_OUT.getData(), N, POW, MaxThreadsPerBlock);
    // Sync all threads before copying results to host
    cudaDeviceSynchronize();
    // Copy result to host back
    d_OUT.get(&h_returnNaiveGPUOUT[0], SIZE);
    // Stop Naive-GPU-Timer
    float gpuTimeNaive = timer_naive_GPU.GetCounter();

    err_naive = cpu_calculate_error(h_OUT, h_returnNaiveGPUOUT, N);

    //std::cout << "Print result matrix with SCCs" << std::endl;
    cpu_printMatrix(h_returnNaiveGPUOUT, N);

    // Free Cuda
    cudaFree(0);

    std::cout << "Naive GPU time  = " << gpuTimeNaive << " ms" << std::endl;
    if(CPU==1)
    cout << "Naive GPU error: " << err_naive << std::endl;

#endif

/*****************************************************************************************/
/******************************End Naive GPU calculations*********************************/
/*****************************************************************************************/

/*****************************************************************************************/
/**********************Start Naive Pinned memory calculations*****************************/
/*****************************************************************************************/

// It seems that cudaMallocHost and cudaHostAlloc have the same functionality.
// For CUDA < 3.0 use cudaMallocHost
// See: https://forums.developer.nvidia.com/t/cudamallochost-and-cudahostalloc-differences-and-usage/21056/2
#if PINNED == 1  
    std::cout << "\nGPU Naive calculation with pinned memory started" << std::endl;

    // canMapHostMemory see: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_115414c4b1fedd1a22030522d54caa653
    if (deviceProp.canMapHostMemory == 1)
    {
        // Overwrite matrices with 0's
        cpu_clean_initialization_vectors(h_A, h_B, h_C);

        // Read in the input matrix
        h_A = cpu_read_matrix(file_path, &N);
        // If N is not a multiple of 2 than do a padding and increment N by 1
        cpu_padding_matrix(h_A, &N);

        // If matrix is empty than either an error occured or the file was empty
        if(h_A.empty())
            return 1;

        // Allocate pinned memory on the device
        gpuHelper<int> d_I_pinned(SIZE, 'P');
        gpuHelper<int> d_A_pinned(SIZE, 'P');
        gpuHelper<int> d_B_pinned(SIZE, 'P');
        gpuHelper<int> d_C_pinned(SIZE, 'P');
        gpuHelper<int> d_OUT_pinned(SIZE, 'P');

        // Copy from host to device on allocated memory
        d_I_pinned.set(&h_I[0], SIZE);
        d_A_pinned.set(&h_A[0], SIZE);
        d_B_pinned.set(&h_A[0], SIZE);
        d_C_pinned.set(&h_A[0], SIZE);
        d_OUT_pinned.set(&h_OUT[0], SIZE);

        // Start Naive-GPU timer
        timer_pinned.StartCounter();

        // Call scc naive
        sccNaive(d_I_pinned.getData(), d_A_pinned.getData(), d_B_pinned.getData(), d_C_pinned.getData(), 
                 d_OUT_pinned.getData(), N, POW, MaxThreadsPerBlock);
        // Sync all threads before copying results to host
        cudaDeviceSynchronize();
        // Copy result to host back
        d_OUT_pinned.get(&h_returnNaiveGPUOUT[0], SIZE);
        // Stop Naive-GPU-Timer
        float pinnedTimeNaive = timer_pinned.GetCounter();

        err_pinned = cpu_calculate_error(h_OUT, h_returnNaiveGPUOUT, N);

        // Free Cuda
        cudaFree(0);

        std::cout << "Pinned naive time  = " << pinnedTimeNaive << " ms" << std::endl;
        if(CPU==1)
        cout << "Pinned naive error: " << err_pinned << std::endl;
    }
    else 
    { 
        printf("Pinned memory allocation can't be done. Device does not support mapping CPU host memory!\n");
    }
#endif
/*****************************************************************************************/
/**********************End Naive Pinned memory calculations*****************************/
/*****************************************************************************************/

/*****************************************************************************************/
/************************Start Naive Zero-Copy calculations*******************************/
/*****************************************************************************************/
#if ZC == 1
    std::cout << "\nNaive calculation with zero-copy memory started" << std::endl;
    // canMapHostMemory see: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_115414c4b1fedd1a22030522d54caa653
    if (deviceProp.canMapHostMemory == 1)
    {
        int *h_I_zc, *h_A_zc, *h_B_zc, *h_C_zc, *h_OUT_zc;
        int *d_I_zc, *d_A_zc, *d_B_zc, *d_C_zc, *d_OUT_zc;

        cudaHostAlloc( (void**)&h_I_zc, SIZE * sizeof(int), cudaHostAllocWriteCombined | cudaHostAllocMapped  );
        cudaHostAlloc( (void**)&h_A_zc, SIZE * sizeof(int), cudaHostAllocMapped  );
        cudaHostAlloc( (void**)&h_B_zc, SIZE * sizeof(int), cudaHostAllocWriteCombined | cudaHostAllocMapped  );
        cudaHostAlloc( (void**)&h_C_zc, SIZE * sizeof(int), cudaHostAllocMapped  );
        cudaHostAlloc( (void**)&h_OUT_zc, SIZE * sizeof(int), cudaHostAllocMapped  );

        for (int i=0; i < SIZE; i ++){
            h_I_zc[i] = 0;
        }
        for (int i=0; i < SIZE; i += N + 1){
            h_I_zc[i] = 1;
        }

        // Read in the input matrix
        std::vector<int> dummy = cpu_read_matrix(file_path, &N);
        // If N is not a multiple of 2 than do a padding and increment N by 1
        cpu_padding_matrix(dummy, &N);

        for (int i=0; i < SIZE; i++){
            h_A_zc[i] = dummy.at(i);
            h_B_zc[i] = dummy.at(i);
            h_C_zc[i] = dummy.at(i);
        }

        cudaHostGetDevicePointer( &d_I_zc, h_I_zc, 0 );
        cudaHostGetDevicePointer( &d_A_zc, h_A_zc, 0 );
        cudaHostGetDevicePointer( &d_B_zc, h_B_zc, 0 );
        cudaHostGetDevicePointer( &d_C_zc, h_C_zc, 0 );
        cudaHostGetDevicePointer( &d_OUT_zc, h_OUT_zc, 0 );

        // If matrix is empty than either an error occured or the file was empty
        if(h_A.empty())
            return 1;

        // Start Naive-GPU timer
        timer_zerocopy.StartCounter();

        // Call scc naive
        sccNaive(d_I_zc, d_A_zc, d_B_zc, d_C_zc, d_OUT_zc, N, POW, MaxThreadsPerBlock);
        
        // Sync all threads with cudaThreadSynchronize !!!
        cudaThreadSynchronize();

        // Stop Naive-GPU-Timer
        float zerocopyTimeNaive = timer_zerocopy.GetCounter();

        std::vector<int> dummy_OUT(d_OUT_zc, d_OUT_zc + SIZE);

        err_zerocopy = cpu_calculate_error(h_OUT, dummy_OUT, N);

        cudaFreeHost( h_I_zc );
        cudaFreeHost( h_A_zc );
        cudaFreeHost( h_B_zc );
        cudaFreeHost( h_C_zc );
        cudaFreeHost( h_OUT_zc );

        // Free Cuda
        cudaFree(0);

        std::cout << "Zero-Copy naive time  = " << zerocopyTimeNaive << " ms" << std::endl;
        if(CPU==1)
        cout << "Zero-Copy naive error: " << err_zerocopy << std::endl;
    }
    else 
    { 
        printf("Pinned memory allocation can't be done. Device does not support mapping CPU host memory!\n");
    }
#endif
/*****************************************************************************************/
/************************End Naive Zero-Copy calculations*******************************/
/*****************************************************************************************/

/*****************************************************************************************/
/***************************Start Shared GPU calculations*********************************/
/*****************************************************************************************/
#if SHARED == 1    
    std::cout << "\nGPU Shared calculation started" << std::endl;

    // Overwrite matrices with 0's
    cpu_clean_initialization_vectors(h_A, h_B, h_C);
    
    // Read in the input matrix
    h_A = cpu_read_matrix(file_path, &N);
    // If N is not a multiple of 2 than do a padding and increment N by 1
    // cpu_padding_matrix(h_A, &N);

    // If matrix is empty than either an error occured or the file was empty
    if(h_A.empty())
        return 1;

    // Allocate memory on the device
    gpuHelper<int> d_I_shared(SIZE, 'M');
    gpuHelper<int> d_A_shared(SIZE, 'M');
    gpuHelper<int> d_B_shared(SIZE, 'M');
    gpuHelper<int> d_C_shared(SIZE, 'M');
    ////gpuHelper<int> d_OUT(SIZE, 'M');
    gpuHelper<int> d_Shared_OUT(SIZE, 'M');
  
    // Copy from host to device on allocated memory
    d_I_shared.set(&h_I[0], SIZE);
    d_A_shared.set(&h_A[0], SIZE);
    d_B_shared.set(&h_A[0], SIZE);
    d_C_shared.set(&h_A[0], SIZE);
    d_Shared_OUT.set(&h_A[0], SIZE);

    // Start Shared-GPU timer
    timer_shared_GPU.StartCounter();
    // Call scc shared
    sccShared(d_I_shared.getData(), d_A_shared.getData(), d_B_shared.getData(), d_C_shared.getData(), d_Shared_OUT.getData(), N, POW, MaxThreadsPerBlock);
    // Sync all threads before copying results to host
    cudaDeviceSynchronize();
    // Copy result to host
    d_Shared_OUT.get(&h_returnSharedGPUOUT[0], SIZE);
    // Stop Shared-GPU-Timer
    float gpuTimeShared = timer_shared_GPU.GetCounter();

    //cpu_printMatrix(h_returnSharedGPUOUT, N);

    err_shared = cpu_calculate_error(h_OUT, h_returnSharedGPUOUT, N);
    std::cout << "Shared GPU time  = " << gpuTimeShared << " ms" << std::endl;
    cout << "Shared GPU error: " << err_shared << std::endl;
    // Free Cuda
    cudaFree(0);

#endif
/*****************************************************************************************/
/******************************End Shared GPU calculations********************************/
/*****************************************************************************************/

    std::cout << "\nApplication finished" << std::endl;
    cudaDeviceReset();

    cpu_printSCC(h_returnNaiveGPUOUT, N);

    if(argument2 != "")
        cpu_writeSCC(h_returnNaiveGPUOUT, N, argument2);

    return 0;
}

/******************************Methods********************************/

std::vector<int> cpu_read_matrix(std::string path, int *N)
{
	std::ifstream infile(path.c_str());

	char tag;
	std::size_t num_nodes, num_edges;
	bool directed;

	infile >> tag >> num_nodes >> num_edges >> directed;
	if (!infile) {
		std::cerr << "Loading dense graph failed, first line is invalid\n";
		return vector<int>();
	}
	if (tag != 'H') {
		std::cerr << "Loading dense graph failed, wrong header tag in the first line\n";
		return vector<int>();
	}
	if (!directed) {
		std::cerr << "Loading dense graph failed, only directed graphs are supported\n";
		return vector<int>();
	}
    
	std::size_t size = num_nodes * num_nodes;
	std::vector<int> matrix(size);

	for (std::size_t i = 0; i < num_edges; ++i) {
		std::size_t src, dest;
		int weight;
		infile >> tag >> src >> dest >> weight;
		if (!infile) {
			std::cerr << "Loading dense graph failed, edge #" << i << " is invalid\n";
			return vector<int>();
		}
		if (tag != 'E') {
			std::cerr << "Loading dense graph failed, edge #" << i << " has wrong tag\n";
			return vector<int>();
		}

		std::size_t index = src * num_nodes + dest;

		// Write 1 to adjacency matrix
		matrix[index] = 1;
    }
    *N = num_nodes;
	return matrix;
}

/*
* If the input matrix is not a multiple of two than do a padding so the calculations on the GPU will be easier to follow.
*/
void cpu_padding_matrix(std::vector<int>& inputMatrix, int *N)
{
    // Do padding if N is not a multiple of 2
    if(*N % 2 != 0)
    {
        // Add 0 to the end of every line
        for (int i = 1; i <= *N; i++) {
            inputMatrix.insert(inputMatrix.begin() + i + *N, 0);
        }
        // Add a last row with 0 to the input matrix
        for (int i = 0; i <= *N; i++) {
            inputMatrix.push_back(0);
        }

        *N = *N + 1; // Padding was needed, so we have to increment N
    }
}

/*
* Calculates the difference between two vectors.
*/
int cpu_calculate_error(std::vector<int>& cpuResults, std::vector<int>& gpuResults, const int N)
{
    int err = 0;
    // Check the result and make sure it is correct
    for (int i=0; i < N*N; i++){
        err += cpuResults[i] - gpuResults[i];
    }
    return err;
}

/*
* Print SCC's to std::out.
*/
void cpu_printSCC(std::vector<int>& gpuResults, const int N)
{
    //int writtenSCCs[N] = { };
    int* writtenSCCs = new int[N];
    int ind = 0;
    int sccCounter = 0;

    // Output of SCC's
    for (int ROW = 0; ROW < N; ROW++) {
        if(writtenSCCs[ROW] != 1)
        {
            std::cout << "\nSCC" << sccCounter << ": ";
		    for (int COL = 0; COL < N; COL++) {
                if(gpuResults[ROW * N + COL] == 1)
                {
                    std::cout << COL << ", ";
                    writtenSCCs[COL] = 1;
                }
                ind++;
            }
            sccCounter++;
        }
	}
}

/*
* Write SCC's to file.
*/
void cpu_writeSCC(std::vector<int>& gpuResults, const int N, const string path)
{
    //int writtenSCCs[N] = { }; 
    int* writtenSCCs = new int[N];
    int ind = 0;
    int sccCounter = 0;

    ofstream file;
    file.open(path.c_str());
    string lines = "";
    string s;
    char* str;

    // Output of SCC's
    for (int ROW = 0; ROW < N; ROW++) {
        if(writtenSCCs[ROW] != 1)
        {
            //file << "SCC" << sccCounter << ": ";
            asprintf(&str, "%i", sccCounter);
            s = str;
            lines = "SCC" + s + ": ";

		    for (int COL = 0; COL < N; COL++) {
                if(gpuResults[ROW * N + COL] == 1)
                {
                    //file << COL << ", ";
                    asprintf (&str, "%i", COL);
                    s = str;
                    lines += s + ", ";
                    writtenSCCs[COL] = 1;
                }
                ind++;
            }
            sccCounter++;
            file << lines.substr(0, lines.size()-2); //get rid of the last ', '
            file << "\n";
            lines = "";
        }
    }
    
    file.close();
    free(str);
}

/*
* Fill vectors with 0's. Is needed for clean initialization because vectors are transformed when the algorithm is run.
*/
void cpu_clean_initialization_vectors(std::vector<int>& A, std::vector<int>& B, std::vector<int>& C)
{
    // Initialize output matrix with 0's
    std::fill(A.begin(), A.end(), 0);
    std::fill(B.begin(), B.end(), 0); 
    std::fill(C.begin(), C.end(), 0); 
}

/*
* Print adjacency matrix.
*/
void cpu_printMatrix(std::vector<int>& inputMatrix, const int N)
{
	for (int ROW = 0; ROW < N; ROW++) {
		for (int COL = 0; COL < N; COL++) {
			std::cout << inputMatrix[ROW * N + COL] << " ";
		}
		std::cout << std::endl;
	}
}
/*
* Matrix multiplication on CPU.
*/
void cpu_matMul(std::vector<int>& A, std::vector<int>& B, std::vector<int>& C, const int N)
{
	unsigned int sum;
	for (int row = 0; row < N; row++) {
		for (int col = 0; col < N; col++) {
			sum = 0;
			for (int n = 0; n < N; n++) {
				sum += A[row*N + n] * B[n*N + col];
            }
            // Take care of overflows 
            if(sum != 0)
                sum = 1;

			C[row*N + col] = sum;
		}
	}
}

/*
* Sum of two matrices of same length.
*/
void cpu_sumMat(std::vector<int>& OUTMAT, std::vector<int>& A, const int N)
{
    for (int row = 0; row < N * N; row++) {
        OUTMAT[row] = OUTMAT[row] + A[row];
    }
}

/*
* Transpose matrix.
*/
void cpu_matTranspose(std::vector<int>& A, std::vector<int>& B, const int N)
{
	for (int row = 0; row < N; row++) {
		for (int col = 0; col < N; col++) {
			A[row*N + col] = B[col*N + row];
		}
	}
}
/*
* Element-wise AND function of two matrices of same length.
*/
void cpu_matBitAnd(std::vector<int>& A, std::vector<int>& B, const int N)
{
	for (int i = 0; i < N*N; i++) {
		if (abs(A[i]) && abs(B[i]))
			A[i] = 1;
		else
			A[i] = 0;
	}
}