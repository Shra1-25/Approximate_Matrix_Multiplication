#include<stdio.h>
#include<cuda.h>

__global__ void predict_gpu_branch(double* deviceMatrix,int N_test ,int D, int num_cols, int M,  double* precomputed_products, int* indices, double* thresholds, double* output, int c)
{
    int subspace = c;
    int row = blockIdx.x;
    int col = threadIdx.x;
    
    double thread_data=0;

    int cur_index = 0;
    for(int i =0;i< NUM_LEVELS-1;i++){
        if(deviceMatrix[indices[subspace*NUM_LEVELS+i] + row*D ] >= thresholds[subspace*NUM_NODES + cur_index])
	    cur_index = 2*cur_index + 2;
	else
	    cur_index = 2*cur_index + 1;
    }
    thread_data = precomputed_products[subspace*NUM_LEAVES*num_cols + cur_index*num_cols + col];   
    output[col * N_test + row] += thread_data;
}


__global__ void predict_gpu(double* deviceMatrix,int N_test ,int D, int num_cols, int M,  double* precomputed_products, int* indices, double* thresholds, double* output, int c)
{
    int subspace = c;
    int row = blockIdx.x;
    int col = threadIdx.x;
    
    int cur_index = 0;
    double thread_data=0;

    for(int i =0;i< NUM_LEVELS-1;i++){
        int b = deviceMatrix[indices[subspace*NUM_LEVELS+i] + row*D ] >= thresholds[subspace*NUM_NODES + cur_index];
        cur_index = 2*cur_index + 1 + b;
    }
    thread_data = precomputed_products[subspace*NUM_LEAVES*num_cols + cur_index*num_cols + col];   
    output[col * N_test + row] += thread_data;
}


__global__ void predict_gpu_shared(double* deviceMatrix,int N_test ,int D, int num_cols, int M,  double* precomputed_products, int* indices, double* thresholds, double* output, int c)
{
    // deviceMatrix : N_test (num_rows) x D
    // B(weights) : DxR
    // output : N_test x R = num_blocks x block_size
    // D in [256, 4096]
    // R in [16, 128] => R<D
    // N_test in [2000, 204800]
    // indices : CxNUM_LEVELS => indices_s : NUM_LEVELS (=4)
    // thresholds : CxNUM_NODES => thresholds_s : NUM_NODES (=15)
    
    int subspace = c;
    int row = blockIdx.x;	// every example goes in a different block. Shared memory will be for only 1 sample
    int col = threadIdx.x;	// R

    __shared__ int indices_s[NUM_LEVELS];
    for(int i=col; i<NUM_LEVELS; i+=num_cols)
        indices_s[i] = indices[subspace*NUM_LEVELS + i];

    __shared__ double thresholds_s[NUM_NODES];
    for(int i=col; i<NUM_NODES; i+=num_cols)
        thresholds_s[i] = thresholds[i + NUM_NODES*subspace];

    __syncthreads();

    int cur_index = 0;
    for(int i =0;i< NUM_LEVELS-1;i++){
        int b = (deviceMatrix[indices_s[i] + row*D] >= thresholds_s[cur_index]);
        cur_index = 2*cur_index + 1 + b;
    }
    double thread_data = precomputed_products[subspace*NUM_LEAVES*num_cols + cur_index*num_cols + col];   
    output[col * N_test + row] += thread_data;
}

__global__ void predict_gpu_opt(double* deviceMatrix,int N_test ,int D, int num_cols, int M,  double* precomputed_products, int* indices, double* thresholds, double* output, int c){
    // deviceMatrix: let's try shared memory: only store num_levels columns
    // precomputed_products: C x NUM_LEAVES x R = 8*16*R => optimized: 16*R
    //      max: 15*(128)*sizeof(double) = 16*2^7*2^3 bytes = 16*2^10B = 16KB => shared memory???
    // updated deviceMatrix = N_test x C(8) x NUM_LEVELS(4)	=> [2000, 1024000]*32*sizeof(float)

    int subspace = c;
    int row = blockIdx.x;	// every example goes in a different block. Shared memory will be for only 1 sample
    int col = threadIdx.x;	// R

    int cur_index = 0;
    for(int i =0;i< NUM_LEVELS-1;i++){
        int b = deviceMatrix[i + NUM_LEVELS*subspace+row*(8*NUM_LEVELS)] >= thresholds[cur_index+NUM_NODES*subspace];
        cur_index = 2*cur_index + 1 + b;
    }
    double thread_data = precomputed_products[subspace*NUM_LEAVES*num_cols + cur_index*num_cols + col];   
    output[col * N_test + row] += thread_data;

}

__global__ void predict_gpu_shared_opt(double* deviceMatrix,int N_test ,int D, int num_cols, int M,  double* precomputed_products, int* indices, double* thresholds, double* output, int c)
{
    // precomputed_products: C x NUM_LEAVES x R = 8*16*R => optimized: 16*R
    //      max: 15*(128)*sizeof(double) = 16*2^7*2^3 bytes = 16*2^10B = 16KB => shared memory???

    int subspace = c;
    int row = blockIdx.x;	// every example goes in a different block. Shared memory will be for only 1 sample
    int col = threadIdx.x;	// R

    __shared__ double thresholds_s[NUM_NODES];
    for(int i=col; i<NUM_NODES; i+=num_cols)
        thresholds_s[i] = thresholds[i + NUM_NODES*subspace];

    __shared__ double A_s[NUM_NODES];
    for(int i=col; i<NUM_LEVELS; i+=num_cols)
        A_s[i] = deviceMatrix[i + NUM_LEVELS*subspace+row*(8*NUM_LEVELS)];

    __syncthreads();

    int cur_index = 0;
    for(int i =0;i< NUM_LEVELS-1;i++){
        int b = A_s[i] >= thresholds_s[cur_index];
        cur_index = 2*cur_index + 1 + b;
    }
    double thread_data = precomputed_products[subspace*NUM_LEAVES*num_cols + cur_index*num_cols + col];   
    output[col * N_test + row] += thread_data;

}
