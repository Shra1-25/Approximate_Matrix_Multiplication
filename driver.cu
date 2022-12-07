#include "datagenerator.hpp"
#include "regressionTree.hpp"
#include "train.hpp"
#include "utils.hpp"
#include "predict_gpu.cu"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

__host__
void print_matrix(double* mat, int rnum, int cnum) {
    for(int i = 0; i < rnum; i++) {
        printf("{");
        for(int j = 0; j < cnum; j++) {
            printf("%lf,", mat[j*rnum + i]);
        }
        printf("},");
    }
    printf("}\n");
}
__host__
void convert_to_row_major(double* input, double* output, int rows, int cols)
{
    for(int i =0;i<rows;i++)
    {
        for(int j = 0;j<cols;j++)
        {
            output[i*cols+j] = input[j*rows+i];
        }
    }
}

__host__
int main(int argc, char** argv) {
    int N = 1000; // number of examples in A_train
    int D = 256; // size of each example
    int C = 8; // number of subspaces
    int R = 32; // size of each row in B
    int NUM_RUNS = 10; // number of inference experiments to run
    
    int nthreads = 1;

    struct cudaDeviceProp * prop = (cudaDeviceProp*) malloc(sizeof(cudaDeviceProp));
    int device;
    cudaGetDeviceCount(&device);
    cudaGetDeviceProperties(prop, device-1);
    cerr<<"lol"<<endl;
    cerr<<prop->maxThreadsDim[0]<<endl;
    cerr<<prop->maxThreadsDim[1]<<endl;
    cerr<<prop->maxThreadsDim[2]<<endl;
    cerr<<prop->maxThreadsPerBlock<<endl;

    // handle arguments with getopt
    char *spec = NULL;
    int index;
    int c;
    char sched_algo;

    opterr = 0;

    while ((c = getopt (argc, argv, "n:d:r:c:")) != -1)
    switch (c)
    {
    case 'n':
        spec = optarg;
        sscanf(spec, "%d", &N);
        break;
    case 'd':
        spec = optarg;
        sscanf(spec, "%d", &D);
        break;
    case 'r':
        spec = optarg;
        sscanf(spec, "%d", &R);
        break;
    case 'c':
        spec = optarg;
        sscanf(spec, "%d", &C);
        break;

    case '?':
        if (optopt == 's')
            fprintf (stderr, "Option -%c requires an argument.\n", optopt);
        else if (isprint (optopt))
            fprintf (stderr, "Unknown option `-%c'.\n", optopt);
        else
            fprintf (stderr,
                    "Unknown option character `\\x%x'.\n",
                    optopt);
        exit(EXIT_FAILURE);
    
    default:
        printf("ERROR opt %c\n", c);
        exit(EXIT_FAILURE);
    }    
    
    double* A_train = generateExamples(N, D);
    double* B = generateExamples(D, R);

    Timer timer;

    RegressionTree* t = new RegressionTree(D, C);
    
    t->fit(A_train, N);
    t->precompute_products(B, R);

    double serial_time = 0.0, omp_time = 0.0;
    printf("Test matrix size\tSerial Time\t GPU Time\tGPU Speedup\tError\n");
    for(int N_test = 2000; N_test <= 1024000; N_test *= 2) {
        double* A_test = generateExamples(N_test, D);
        printf("%d x %d\t      ", N_test, D);
        
        double* output_cpu = new double[N_test * R];
        for(int i =0;i<N_test * R;i++)
            output_cpu[i] = 0;
        for(int i = 0; i < NUM_RUNS; i++) {
            timer.tic();
            // t->predict_cpu(A_test, N_test, output_cpu);
            omp_time += timer.toc();
        }

        double* output = new double[N_test * R];
        for(int i =0;i<N_test * R;i++)
            output[i] = 0;
        for(int i = 0; i < NUM_RUNS; i++) {
            timer.tic();
            t->predict(A_test, N_test, output);
            serial_time += timer.toc();
        }
        printf("%10lf\t", serial_time/NUM_RUNS);

        cudaEvent_t start,stop;
        float elapsedTime;

        cudaEventCreate (&start);
        cudaEventCreate (&stop);
        double* device_matrix, *device_products,  *device_thresholds, *device_output;
        int *device_indices;
        int x = cudaMalloc( (void**)&device_matrix, N_test*D* sizeof(double));
//        int x = cudaMalloc( (void**)&device_matrix, N_test*C*NUM_LEVELS* sizeof(double));
        if(x)
           cerr<<x<<endl;
        x = cudaMalloc( (void**)&device_products, C*NUM_LEAVES*R* sizeof(double));
        if(x)
           cerr<<x<<endl;
        x = cudaMalloc( (void**)&device_indices, C*NUM_LEVELS* sizeof(int));
        if(x)
           cerr<<x<<endl;
        x = cudaMalloc( (void**)&device_thresholds, C*NUM_NODES* sizeof(double));
        if(x)
           cerr<<x<<endl;
        x = cudaMalloc( (void**)&device_output, N_test*R*sizeof(double));
        if(x)
           cerr<<x<<endl;
        cudaMemset(device_output, 0, N_test*R*sizeof(double));

        double* A_test_row_major = new double[ N_test*D];
        convert_to_row_major(A_test, A_test_row_major, N_test, D);

	double* selcted_A_test_row_major = new double[N_test * C * NUM_LEVELS]; // C x NUM_LEVELS	
	for(int c=0; c<C;c++){
		for(int i=0; i<NUM_LEVELS-1; i++){
			for(int eg_idx=0; eg_idx<N_test; eg_idx++){
				selcted_A_test_row_major[c*NUM_LEVELS+i+eg_idx*NUM_LEVELS*C] = A_test_row_major[t->indices[c*NUM_LEVELS+i] + eg_idx*D];
			}
		}
	}
	
        cudaMemcpy((void*)device_matrix, (void*)A_test_row_major, N_test*D* sizeof(double) ,cudaMemcpyHostToDevice);
//        cudaMemcpy((void*)device_matrix, (void*)selcted_A_test_row_major, N_test*C*NUM_LEVELS* sizeof(double) ,cudaMemcpyHostToDevice);
        cudaMemcpy((void*)device_products, (void*)t->products,C*NUM_LEAVES*R *sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy((void*)device_indices, (void*)t->indices, C*NUM_LEVELS* sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy((void*)device_thresholds, (void*)t->thresholds,C*NUM_NODES* sizeof(double),cudaMemcpyHostToDevice);
        dim3 dimGrid(N_test);
        dim3 dimBlock(R);
        cudaEventRecord (start, 0);
        omp_time =0.0;
        for(int i =0;i< NUM_RUNS;i++){
            cudaMemset(device_output, 0, N_test*R*sizeof(double));
            for(int c = 0;c<C;c++){
                predict_gpu_shared<<<dimGrid, dimBlock>>>(device_matrix, N_test, D, R, D/C, device_products, device_indices, device_thresholds, device_output, c);
                cudaDeviceSynchronize();
            }
            cudaEventRecord (stop, 0);
            cudaEventSynchronize (stop);
            cudaEventElapsedTime ( &elapsedTime, start, stop);
            omp_time += elapsedTime * 1e-3;
        }

        double* host_result = new double[N_test*R];
        cudaMemcpy((void*)host_result, (void*)device_output,N_test*R*sizeof(double),cudaMemcpyDeviceToHost);

        double max_err = 0;
        for (long i = 0; i < N_test * R; i++) max_err = max(max_err, fabs(host_result[i] - output[i]));
        printf("%10lf\t", omp_time/NUM_RUNS);
        printf("%10lf\t", serial_time / omp_time);
        printf("%10e\n", max_err);
        cudaFree(device_matrix);
        cudaFree(device_products);
        cudaFree(device_indices);
        cudaFree(device_thresholds);
        cudaFree(device_output);
        free(A_test);
        free(A_test_row_major);
        free(output);
        free(selcted_A_test_row_major);
        free(host_result);
    }

}
