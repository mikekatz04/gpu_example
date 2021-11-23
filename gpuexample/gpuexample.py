import numpy as np

try:
    import cupy as xp
    from pygpuexample import build_sine_waves as build_sine_waves_gpu

    gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

    gpu_available = False

from pygpuexample_cpu import build_sine_waves as build_sine_waves_cpu
import time

YRSID_SI = 31558149.763545603


class pyGPUExample(object):
    """C/C++ Wrapper example"""
    def __init__(self, use_gpu=False):

        self.use_gpu = use_gpu
        if use_gpu:
            if not gpu_available:
                raise ValueError("No GPU available or cupy not installed.")
            self.xp = xp
            self.wave_gen = build_sine_waves_gpu

        else:
            self.xp = np
            self.wave_gen = build_sine_waves_cpu

    def __call__(self, A, f, phi, dt=10.0, T=1.0):

        A = self.xp.asarray(self.xp.atleast_1d(A))
        f = self.xp.asarray(self.xp.atleast_1d(f))
        phi = self.xp.asarray(self.xp.atleast_1d(phi))

        num_modes = len(A)
        num_t = int(T * YRSID_SI/dt)

        output_waves = self.xp.zeros((num_modes, num_t)).flatten()

        self.wave_gen(output_waves, A, f, phi, dt, num_t, num_modes)
        
        return output_waves.reshape(num_modes, num_t)
