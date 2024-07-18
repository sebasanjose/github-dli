#include <math.h>
#include <time.h>
#include <iostream>
#include <stdexcept>
#include "cuda_runtime.h"

// declare the vectors' number of elements and their size in bytes
static const int n_el = 10000000; // 10 million
static const size_t size = n_el * sizeof(float);
static const int n_streams = 4; // Number of streams

// function for computing sum on CPU
void CPU_sum(const float* A, const float* B, float* C, int n_el) {
    for (int i = 0; i < n_el; i++) {
        C[i] = A[i] + B[i];
    }
}

// kernel
__global__ void kernel_sum(const float* A, const float* B, float* C, int n_el) {
    // calculate the unique thread index
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    // perform tid-th elements addition 
    if (tid < n_el) C[tid] = A[tid] + B[tid];
}

// function which invokes the kernel using multiple streams
void GPU_sum(const float* A, const float* B, float* C, int n_el) {
    // declare device vectors in the device (GPU) memory
    float *d_A, *d_B, *d_C;

    // allocate device vectors in the device (GPU) memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // declare the number of blocks per grid and the number of threads per block
    int threadsPerBlock = 512;
    int blocksPerGrid = (n_el + threadsPerBlock - 1) / threadsPerBlock;

    // create CUDA streams
    cudaStream_t streams[n_streams];
    for (int i = 0; i < n_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // calculate the size of each chunk and the size in bytes
    int chunk_size = n_el / n_streams;
    size_t chunk_bytes = chunk_size * sizeof(float);

    // launch kernels in parallel streams
    for (int i = 0; i < n_streams; ++i) {
        int offset = i * chunk_size;

        // transfer data from CPU to GPU for each stream
        cudaMemcpyAsync(d_A + offset, A + offset, chunk_bytes, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_B + offset, B + offset, chunk_bytes, cudaMemcpyHostToDevice, streams[i]);

        // invoke the kernel
        kernel_sum<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(d_A + offset, d_B + offset, d_C + offset, chunk_size);

        // transfer data from GPU to CPU for each stream
        cudaMemcpyAsync(C + offset, d_C + offset, chunk_bytes, cudaMemcpyDeviceToHost, streams[i]);
    }

    // wait for all streams to complete
    for (int i = 0; i < n_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    // declare and allocate input vectors h_A and h_B in the host (CPU) memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // initialize input vectors
    for (int i = 0; i < n_el; i++) {
        h_A[i] = sin(i);
        h_B[i] = cos(i);
    }

    /************ CPU Version ***********/

    clock_t tstart, tend;
    float cpu_duration;
    // compute on CPU
    tstart = clock();

    /////////////////////////////////
    // call kernel function
    /////////////////////////////////
    CPU_sum(h_A, h_B, h_C, n_el);
    /////////////////////////////////

    tend = clock();
    cpu_duration = ((float)(tend - tstart)) / CLOCKS_PER_SEC;
    printf("Total  time for sum on CPU: %f seconds\n", cpu_duration);

    /************ GPU Version ***********/

    clock_t tstart_total;
    tstart_total = clock();

    float gpu_duration;
    tstart = clock();

    /////////////////////////////////
    // call kernel function using multiple streams
    /////////////////////////////////
    GPU_sum(h_A, h_B, h_C, n_el);
    /////////////////////////////////

    tend = clock();
    gpu_duration = ((float)(tend - tstart)) / CLOCKS_PER_SEC;
    printf("Kernel time for sum on GPU: %f seconds\n", gpu_duration);

    tend = clock();
    gpu_duration = ((float)(tend - tstart_total)) / CLOCKS_PER_SEC;
    printf("Total  time for sum on GPU: %f seconds\n", gpu_duration);

    /************ Check correctness using RMS Error ***********/

    // compute the squared error of the result
    // using double precision for good accuracy
    double err = 0;
    for (int i = 0; i < n_el; i++) {
        double diff = double((h_A[i] + h_B[i]) - h_C[i]);
        err += diff * diff;
        // print results for manual checking.
        //printf("%f=%f,",h_A[i]+h_B[i],h_C[i]);
    }
    // compute the RMS error
    err = sqrt(err / double(n_el));
    printf("error: %f\n", err);

    printf("speed-up: %.2fx", cpu_duration / gpu_duration);

    // free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
