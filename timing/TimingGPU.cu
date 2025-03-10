// Timer implementations taken from: https://stackoverflow.com/questions/7876624/timing-cuda-operations
// Implemented from SO user: https://stackoverflow.com/users/1886641/jackolantern

/**************/
/* TIMING GPU */
/**************/

#include "TimingGPU.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

struct PrivateTimingGPU {
    cudaEvent_t     start;
    cudaEvent_t     stop;
};

// default constructor
TimingGPU::TimingGPU() { privateTimingGPU = new PrivateTimingGPU; }

// default destructor
TimingGPU::~TimingGPU() { }

void TimingGPU::StartCounter()
{
    cudaEventCreate(&((*privateTimingGPU).start));
    cudaEventCreate(&((*privateTimingGPU).stop));
    cudaEventRecord((*privateTimingGPU).start,0);
}

void TimingGPU::StartCounterFlags()
{
    int eventflags = cudaEventBlockingSync;

    cudaEventCreateWithFlags(&((*privateTimingGPU).start),eventflags);
    cudaEventCreateWithFlags(&((*privateTimingGPU).stop),eventflags);
    cudaEventRecord((*privateTimingGPU).start,0);
}

// Gets the counter in ms
float TimingGPU::GetCounter()
{
    float   time;
    cudaEventRecord((*privateTimingGPU).stop, 0);
    cudaEventSynchronize((*privateTimingGPU).stop);
    cudaEventElapsedTime(&time,(*privateTimingGPU).start,(*privateTimingGPU).stop);
    return time;
}
