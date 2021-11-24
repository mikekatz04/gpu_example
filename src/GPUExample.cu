#include "stdio.h"
#include "GPUExample.hh"
#include "math.h"



CUDA_KERNEL
void build_sine_waves(double* output, double* amplitudes, double* frequencies, double* phases, double dt, int num_t, int num_modes)
{
    int start1, increment1;
    #ifdef __CUDACC__
    start1 = blockIdx.y;
    increment1 = gridDim.y;  // number of blocks along y
    #else
    start1 = 0;
    increment1 = 1;
    #pragma omp parallel for
    #endif
    for (int mode_i = start1; mode_i < num_modes; mode_i += increment1)
    {
        // assign quantities to each mode
        double A = amplitudes[mode_i];
        double f = frequencies[mode_i];
        double phi = phases[mode_i];

        int start2, increment2;
        #ifdef __CUDACC__
        start2 = threadIdx.x + blockIdx.x * blockDim.x;
        increment2 = blockDim.x * gridDim.x;
        #else
        start2 = 0;
        increment2 = 1;
        #pragma omp parallel for
        #endif
        for (int i = start2; i < num_t; i += increment2)
        {
            // get t
            double t = dt * i;
            double val = A * sin(2. * PI * f * t + phi);

            int ind_out = mode_i * num_t + i;
            output[ind_out] = val;
        }
    }
}

void build_sine_waves_wrap(double* output, double* amplitudes, double* frequencies, double* phases, double dt, int num_t, int num_modes)
{
        #ifdef __CUDACC__
        int num_blocks = std::ceil((num_t + NUM_THREADS - 1)/NUM_THREADS);

        dim3 gridDim(num_blocks, num_modes);

        build_sine_waves<<<gridDim, NUM_THREADS>>>
                      (output, amplitudes, frequencies, phases, dt, num_t, num_modes);

        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());
        #else
        build_sine_waves (output, amplitudes, frequencies, phases, dt, num_t, num_modes);
        #endif
}
