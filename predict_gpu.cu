#include<stdio.h>
#include<cuda.h>


__global__ void predict_gpu(double* deviceMatrix,int N_test ,int D, int num_cols, int M,  double* precomputed_products, int* indices, double* thresholds, double* output, int c)
{
    int subspace = c;
    int row = blockIdx.x;
    int col = threadIdx.x;
    
    int cur_index = 0;
    
    double thread_data=0;

    for(int i =0;i< NUM_LEVELS-1;i++)
    {
        int b = deviceMatrix[indices[subspace*NUM_LEVELS+i] + row*D ] >= thresholds[subspace*NUM_NODES + cur_index];

        cur_index = 2*cur_index + 1 + b;
    }

    thread_data = precomputed_products[subspace*NUM_LEAVES*num_cols + cur_index*num_cols + col];
   
    output[col * N_test + row] += thread_data;

}