import numpy as np
cimport numpy as np

from gpuexample.pointer_adjust import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "GPUExample.hh":
    void build_sine_waves_wrap(double* output, double* amplitudes, double* frequencies, double* phases, double dt, int num_t, int num_modes);

@pointer_adjust
def build_sine_waves(output, amplitudes, frequencies, phases, dt, num_t, num_modes):

    cdef size_t output_in = output
    cdef size_t amplitudes_in = amplitudes
    cdef size_t frequencies_in = frequencies
    cdef size_t phases_in = phases

    build_sine_waves_wrap(<double*> output_in, <double*> amplitudes_in, <double*> frequencies_in, <double*> phases_in, dt, num_t, num_modes)
