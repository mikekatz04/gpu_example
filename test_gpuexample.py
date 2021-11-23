import matplotlib.pyplot as plt
import numpy as np
import time

try:
    import cupy as xp
    gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp
    gpu_available = False


from gpuexample.gpuexample import *

np.random.seed(1000)

num_modes = 10
T = 1.0
dt = 10.0

A = np.random.uniform(1.0, 10.0, size=(num_modes,))
f = np.random.uniform(1e-4, 1e-2, size=(num_modes,))
phi = np.random.uniform(0.0, 2 * np.pi, size=(num_modes,))


wave_generator = pyGPUExample(use_gpu=False)
output_wave = wave_generator(A, f, phi, dt=dt, T=T)

num = 10
# time it
st = time.perf_counter()
for _ in range(num):
    output_wave = wave_generator(A, f, phi, dt=dt, T=T)
et = time.perf_counter()

print(f"Duration per evaluation CPU: {(et - st)/num}")

if gpu_available:
    wave_generator = pyGPUExample(use_gpu=True)
    num = 100
    # time it
    st = time.perf_counter()
    for _ in range(num):
        output_wave = wave_generator_gpu(A, f, phi, dt=dt, T=T)
    et = time.perf_counter()

    print(f"Duration per evaluation GPU: {(et - st)/num}")
