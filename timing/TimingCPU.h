// Timer implementations taken from: https://stackoverflow.com/questions/7876624/timing-cuda-operations
// Implemented from SO user: https://stackoverflow.com/users/1886641/jackolantern

// 1 micro-second accuracy
// Returns the time in seconds

#ifndef __TIMINGCPU_H__
#define __TIMINGCPU_H__

#ifdef __linux__

    class TimingCPU {

        private:
            long cur_time_;

        public:

            TimingCPU();

            ~TimingCPU();

            void StartCounter();

            double GetCounter();
    };

#elif _WIN32 || _WIN64

    struct PrivateTimingCPU;

    class TimingCPU
    {
        private:
            PrivateTimingCPU *privateTimingCPU;

        public:

            inline TimingCPU();
            inline ~TimingCPU();
            inline void StartCounter();
            inline double GetCounter();

    }; // TimingCPU class

#endif

#endif
