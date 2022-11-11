# Parallel Approximate Matrix Multiplication

This repository implements the MADDNESS algorithm described [here](https://arxiv.org/abs/2106.10860) as part of the course _CSCI-GA 2945 Advanced Topic in Numerical Analysis: High Performance Computing_. We provide a serial version of the regression tree learning algorithm and serial, OpenMP parallelized, and CUDA parallelized versions of the inference algorithm as described in the paper.

## Running Experiments

Different branches are used to run experiments on CPU and GPU. Both use the same serial implementation as the baseline. Here is how one can reproduce the experiments:

### OpenMP Inference

- Checkout the ```optimize-omp-inference``` branch
- ```./run_experiments.sh``` will report the runtime and the numerical error for varying N (number of rows in A), D (width of A == number of rows in B), and R (width of B) across 4, 8, 16, 32, and 64 threads. Results can be found in corresponding ```.txt``` files in the directory ```omp_results```

### CUDA Inference

- Checkout the ```master``` branch
- ```./run_experiments.sh``` will report the runtime for varying N, D, and R as before. Results can be found in the directory ```gpu_results```

## Overview of our Implementation

The class ```RegressionTree``` in ```regressionTree.hpp``` implements the ```fit``` and ```predict``` methods which learn query indices and thresholds for internal nodes to compute the dot product of prototype vectors in each subspace with columns of B and compute the final approximate product of A and B respectively.

```predict_gpu``` in ```predict_gpu.cu``` implements the code for inference on GPU.
