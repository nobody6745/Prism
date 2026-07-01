# Replication Package for ICSE2027

This repository contains the source code and experimental setup for the paper: **"PRISM: Practical and Fast Floating-Point Error
Estimation for Deep Learning Operators."** PRISM is designed for researchers to quantify numerical uncertainty and floating-point errors in Deep Learning  operators by injecting directed ULP perturbations.

## 1. Triton Environment Setup (`triton`)

This environment is used for running the Triton-based operator experiments and the PRISM Triton backend.

```
# Create and activate conda environment
conda create -n triton python=3.11 -y
conda activate triton

# Install core dependencies
pip install torch==2.4.0 numpy numba tabulate

# Build and install Triton from source
cd triton
pip install ninja cmake wheel pybind11  
pip install -e python                   
```

## 2. PyTorch GPU Environment (`torch-gpu`)

This environment is configured to build PyTorch from source with CUDA support enabled, specifically for GPU-based perturbation injection experiments.

```

conda create -n torch-gpu python=3.11 -y
conda activate torch-gpu
conda install -c conda-forge cmake=3.27 ninja -y
pip install pyyaml numpy typing_extensions
# Set compilation environment variables
cd /PRISM/torch-gpu/pytorch
export CUDA_HOME=/usr/local/cuda  # Update this to your local path if necessary
export CUDACXX=$CUDA_HOME/bin/nvcc
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
# Optimization: Disable unnecessary modules to speed up compilation
export USE_MKL=0
export USE_MKLDNN=0
export USE_DISTRIBUTED=0
export USE_MPI=0
export BUILD_TEST=0
export USE_XNNPACK=0
export USE_FBGEMM=0

# Compile and install
python setup.py install
```

------

## 3. PyTorch CPU Environment (`torch-cpu`)

This environment uses **OpenBLAS** as the backend for CPU-side operator testing and numerical analysis.

### Build OpenBLAS 

```
# In the OpenBLAS source directory
cd /PRISM/OpenBLAS
make clean
make TARGET=GENERIC USE_OPENMP=1 -j$(nproc)
make install PREFIX=$HOME/software/OpenBLAS
```

### Build PyTorch 

```
conda activate torch-cpu
cd /PRISM/torch-cpu/pytorch
# Configure backend and paths
export BLAS=OpenBLAS
export OpenBLAS_HOME=$HOME/software/OpenBLAS
export LD_PRELOAD=$OpenBLAS_HOME/lib/libopenblas.so

# Disable CUDA and enable OpenBLAS
export USE_CUDA=0
export USE_CUDNN=0
export USE_MKL=0
export USE_MKLDNN=0

# Build in development mode
python setup.py clean
python setup.py develop
```

## Run

To run the code, you can use the following commands:

##### Triton part:

```
cd /Prism/triton_scripts
python run_perturb.py # generate interval for each case
python judge_perturb.py # judge the result                         
```

## 

##### Pytorch part:

For cpu:.

```
cd /Prism/torch_scripts/torch-examples/xx 
conda activate triton
python3 xx.py 
export LD_PRELOAD=/OpenBLAS/lib/libopenblas.so
conda activate torch-cpu 
USE_MY_CUTLASS=1 python3 xx_perturb.py (perturb version)           
```

For gpu:

```
cd /Prism/torch_scripts/torch-examples/xx 
conda activate triton
python3 xx.py 
conda activate torch-gpu 
USE_MY_CUTLASS=1 python3 xx_perturb.py (perturb version)                     
```

## 
