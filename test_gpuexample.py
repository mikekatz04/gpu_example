import matplotlib.pyplot as plt
import numpy as np
import time

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

n = int(T * YRSID_SI / dt)
print(f"Duration per evaluation: {(et - st)/num}")
plt.plot(np.arange(n) * dt, output_wave.T)
plt.show()
