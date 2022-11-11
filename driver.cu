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
    // t->print_prototypes();

    t->precompute_products(B, R);
    // t->print_products();

    double serial_time = 0.0, omp_time = 0.0;

    // int N_test = 1000; // number of examples (rows) in test matrix A
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
        // printf("Time taken for omp predict %lf\n", omp_time/NUM_RUNS);


        double* output = new double[N_test * R];
        for(int i =0;i<N_test * R;i++)
        output[i] = 0;
        for(int i = 0; i < NUM_RUNS; i++) {
            timer.tic();
            t->predict(A_test, N_test, output);
            serial_time += timer.toc();
        }
        // printf("Time taken for serial predict %lf\n", serial_time/NUM_RUNS);
        printf("%10lf\t", serial_time/NUM_RUNS);
        // printf("Speedup omp over serial: %lf\n", serial_time / omp_time);

        // print predicted output
        // printf("\nApproximate product:\n{");
        // print_matrix(output, N_test, R);

        // calculate actual output
        // double* output_ref = new double[N_test * R];

        // for(int i = 0; i < R; i++) {
        //     for(int j = 0; j < N_test; j++) {
        //         output_ref[i*N_test + j] = 0.0;
        //         for(int k = 0; k < D; k++) {
        //             output_ref[i*N_test + j] += A_test[k*N_test + j] * B[i*D + k];
        //         }
        //     }
        // }

        // error
        // double max_err = 0;
        // for (long i = 0; i < N_test * R; i++) max_err = max(max_err, fabs(output[i] - output_cpu[i]));
        // printf(" %10e\n", max_err);


        cudaEvent_t start,stop;
        float elapsedTime;

        cudaEventCreate (&start);
        cudaEventCreate (&stop);
        double* device_matrix, *device_products,  *device_thresholds, *device_output;
        int *device_indices;
        cudaMalloc( (void**)&device_matrix, N_test*D* sizeof(double));
        cudaMalloc( (void**)&device_products, C*NUM_LEAVES*R* sizeof(double));
        cudaMalloc( (void**)&device_indices, C*NUM_LEVELS* sizeof(int));
        cudaMalloc( (void**)&device_thresholds, C*NUM_NODES* sizeof(double));
        cudaMalloc( (void**)&device_output, N_test*R*sizeof(double));
        cudaMemset(device_output, 0, N_test*R*sizeof(double));

        double* A_test_row_major = new double[ N_test*D];
        double* A_test_double_transpose = new double[N_test*D];

        convert_to_row_major(A_test, A_test_row_major, N_test, D);
        // convert_to_row_major(A_test_row_major, A_test_double_transpose, D, N_test);
        // max_err = 0;
        // for (long i = 0; i < N_test * D; i++) max_err = max(max_err, fabs(A_test[i] - A_test_double_transpose[i]));
        // printf("Error from transpose: %10e\n", max_err);

        cudaMemcpy((void*)device_matrix, (void*)A_test_row_major, N_test*D* sizeof(double) ,cudaMemcpyHostToDevice);
        cudaMemcpy((void*)device_products, (void*)t->products,C*NUM_LEAVES*R *sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy((void*)device_indices, (void*)t->indices, C*NUM_LEVELS* sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy((void*)device_thresholds, (void*)t->thresholds,C*NUM_NODES* sizeof(double),cudaMemcpyHostToDevice);
        dim3 dimGrid(N_test);
        dim3 dimBlock(R);
        cudaEventRecord (start, 0);
        omp_time =0.0;
        for(int i =0;i< NUM_RUNS;i++)
        {
            cudaMemset(device_output, 0, N_test*R*sizeof(double));
            for(int c = 0;c<C;c++)
            {
                predict_gpu<<<dimGrid, dimBlock>>>(device_matrix, N_test, D, R, D/C, device_products, device_indices, device_thresholds, device_output, c);
                cudaDeviceSynchronize();
            }
            cudaEventRecord (stop, 0);
            cudaEventSynchronize (stop);
            cudaEventElapsedTime ( &elapsedTime, start, stop);
            omp_time += elapsedTime * 1e-3;
        }

        // cudaEventRecord (stop, 0);
        // cudaEventSynchronize (stop);
        // cudaEventElapsedTime ( &elapsedTime, start, stop);

        double* host_result = new double[N_test*R];
        cudaMemcpy((void*)host_result, (void*)device_output,N_test*R*sizeof(double),cudaMemcpyDeviceToHost);

        double max_err = 0;
        for (long i = 0; i < N_test * R; i++) max_err = max(max_err, fabs(host_result[i] - output[i]));
        printf("%10lf\t", omp_time/NUM_RUNS);
        printf("%10lf\t", serial_time / omp_time);
        // printf("Error from GPU: %10e\n", max_err);
        printf("%10e\n", max_err);
    }

}
