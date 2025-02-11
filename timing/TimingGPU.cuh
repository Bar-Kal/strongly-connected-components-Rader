// Timer implementations taken from: https://stackoverflow.com/questions/7876624/timing-cuda-operations
// Implemented from SO user: https://stackoverflow.com/users/1886641/jackolantern

#ifndef __TIMING_CUH__
#define __TIMING_CUH__

/**************/
/* TIMING GPU */
/**************/

// Events are a part of CUDA API and provide a system independent way to measure execution times on CUDA devices with approximately 0.5
// microsecond precision.

struct PrivateTimingGPU;

class TimingGPU
{
    private:
        PrivateTimingGPU *privateTimingGPU;

    public:

        inline TimingGPU();
        inline ~TimingGPU();
        inline void StartCounter();
        inline void StartCounterFlags();
        inline float GetCounter();

}; // TimingCPU class

#endif
