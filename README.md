# Parallel Approximate Matrix Multiplication

This repository implements the MADDNESS algorithm described [here](https://arxiv.org/abs/2106.10860) as part of the course _CSCI-GA 3033 GPUs: Architecture and Programming course at NYU. We provide a serial version of the regression tree learning algorithm and serial, and CUDA (optimized) parallelized versions of the inference algorithm as described in the paper.

## Running Experiments

- Checkout the ```master``` branch
- ```./run_experiments.sh``` will report the runtime for varying N, D, and R. Results can be found in the directory ```gpu_results```

## Overview of our Implementation

The class ```RegressionTree``` in ```regressionTree.hpp``` implements the ```fit``` and ```predict``` methods which learn query indices and thresholds for internal nodes to compute the dot product of prototype vectors in each subspace with columns of B and compute the final approximate product of A and B respectively.

```predict_gpu``` in ```predict_gpu.cu``` implements the code for inference on GPU.
