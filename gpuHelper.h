#ifndef GPUHELPER_H_
#define GPUHELPER_H_

#include <stdio.h>
#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>

template <class T>
class gpuHelper
{
// public functions
public:

    explicit gpuHelper() : begin(0), end(0)
    {}

    // Constructor
    explicit gpuHelper(size_t size, char memoryType) 
    { 
        if(memoryType == 'M') // M => cudaMallocHost
        {
            // Allcoate memory with cudaMallocHost
            allocate(size);  
        }
        else if (memoryType == 'P') // P => Pinned-Memory
        {
            allocatePinnedMemory(size); 
        }       
        else if (memoryType == 'Z') // Z => Zero-Copy
        {
            allocateZeroCopyMemory(size); 
        }   
    }
    // Destructor
    ~gpuHelper() 
    { 
        free(); 
    }

    // get the size of the array
    size_t getSize() const
    {
        return end - begin;
    }

    // get data
    const T* getData() const
    {
        return begin;
    }

    T* getData()
    {
        return begin;
    }

    /*
    * See cuda error types: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
    * under "enum cudaError"
    * Or see here: http://developer.download.nvidia.com/compute/cuda/3_1/toolkit/docs/online/group__CUDART__TYPES_g3f51e3575c2178246db0a94a430e0038.html
    * 
    */
    
    // set
    void set(const T* src, size_t size)
    {
        size_t min = std::min(size, getSize());
        cudaError_t cudaResult = cudaMemcpy(begin, src, min * sizeof(T), cudaMemcpyHostToDevice);
        cudaError_t cudaLastError = cudaGetLastError();
        handleCudaResult(cudaResult, cudaLastError);      
    }
    // get
    void get(T* dest, size_t size)
    {
        size_t min = std::min(size, getSize());
        cudaError_t cudaResult = cudaMemcpy(dest, begin, min * sizeof(T), cudaMemcpyDeviceToHost);
        cudaError_t cudaLastError = cudaGetLastError();
        handleCudaResult(cudaResult, cudaLastError);
    }
    // handle cuda result
    void handleCudaResult(cudaError_t result, cudaError_t lastError)
    {
        if(result != cudaSuccess)
        {
            printf("\nCuda error string: %s\n", cudaGetErrorString(result));
            printf("Cuda last error string: %s\n", cudaGetErrorString(lastError));
            throw std::runtime_error("Cuda error");
        }
    }

// private functions
private:
    // allocate memory on the device
    void allocate(size_t size)
    {
        cudaError_t cudaResult = cudaMallocHost((void**)&begin, size * sizeof(T));
        cudaError_t cudaLastError = cudaGetLastError();
        handleCudaResult(cudaResult, cudaLastError);
        end = begin + size;
    }

    // allocate pinned memory on the device
    void allocatePinnedMemory(size_t size)
    {
        cudaError_t cudaResult = cudaHostAlloc( (void**)&begin, size * sizeof(T), cudaHostAllocDefault );
        cudaError_t cudaLastError = cudaGetLastError();
        handleCudaResult(cudaResult, cudaLastError);
        end = begin + size;
    }

    // allocate pinned memory on the device
    void allocateZeroCopyMemory(size_t size)
    {
        cudaError_t cudaResult = cudaHostAlloc( (void**)&begin, size * sizeof(T), cudaHostAllocWriteCombined | cudaHostAllocMapped  );
        cudaError_t cudaLastError = cudaGetLastError();
        handleCudaResult(cudaResult, cudaLastError);
        end = begin + size;
    }

    // free memory on the device
    void free()
    {
        if (begin != 0)
        {
            cudaFree(begin);
            cudaFreeHost(begin);
            begin = end = 0;
        }
    }

    T* begin;
    T* end;
};

#endif