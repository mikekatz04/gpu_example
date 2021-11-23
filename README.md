# gpuexample: An example of GPU/CPU agnostic Python/C/CUDA codes

This package contains a basic setup for wrapping CPU/GPU agnostic C/C++/CUDA into CPU/GPU agnostic Python. The goal of the package is to compute a 2D array of sine waves with different amplitude, frequency, and initial phase. `src/GPUExample.cu` is the source C/C++/CUDA file. Its header file is `include/GPUExample.hh`. It is wrapped into Python with a Cython wrapper originally inspired by [this Github repository](https://github.com/rmcgibbo/npcuda-example). The Python file that contains the sine wave creation is `gpuexample/gpuexample.py`. The file `test_gpuexample/` will run the code on the CPU and, if available, on an NVIDIA GPU.

## Getting Started / Installing

Below is a quick set of instructions to get you started with `gpuexample`.

0) [Install Anaconda](https://docs.anaconda.com/anaconda/install/) if you do not have it.

1) Create a virtual environment. **Note**: There is no available `conda` compiler for Windows.

```
conda create -n gpuex_env -c conda-forge gcc_linux-64 gxx_linux-64 numpy Cython scipy python=3.9
conda activate few_env
```

    If on MACOSX, substitute `gcc_linux-64` and `gxx_linus-64` with `clang_osx-64` and `clangxx_osx-64`.

2) If using GPUs, use pip to [install cupy](https://docs-cupy.chainer.org/en/stable/install.html). If you have cuda version 9.2, for example:

```
pip install cupy-cuda92
```

3) Clone the repository.

```
git clone https://github.com/mikekatz04/gpu_example.git
cd gpu_example
```

3) Run install. If using GPUs, make sure CUDA is on your `PATH` or the `CUDAHOME` variable is set as an environment variable.

```
python setup.py install
```

3) Run the test file.

```
python test_gpuexample.py
```
5) To import `gpuexample`:

```
from gpuexample.gpuexample import pyGPUExample
```

### Prerequisites

To install this software for use with NVIDIA GPUs (compute capability >2.0), you need the [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [CuPy](https://cupy.chainer.org/). The CUDA toolkit must have cuda version >8.0. Be sure to properly install CuPy within the correct CUDA toolkit version. Make sure the nvcc binary is on `$PATH` or set it as the `CUDAHOME` environment variable.


## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/tags).

Current Version: 1.0.0

## Authors

* **Michael Katz**

## License

This project is licensed under the GNU License - see the [LICENSE.md](LICENSE.md) file for details.
