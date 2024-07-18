#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

// CPU function to add two vectors
void addVectorsCPU(const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C, int N) {
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

// GPU kernel to add two vectors
__global__ void addVectorsGPU(const int* A, const int* B, int* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// Wrapper function to call the GPU kernel
void addVectorsOnGPU(const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C, int N) {
    int *d_A, *d_B, *d_C;
    size_t size = N * sizeof(int);

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_A, size);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating memory for d_A: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    err = cudaMalloc((void**)&d_B, size);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating memory for d_B: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        return;
    }
    err = cudaMalloc((void**)&d_C, size);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating memory for d_C: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        return;
    }

    // Copy vectors from host memory to device memory
    err = cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying memory to d_A: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }
    err = cudaMemcpy(d_B, B.data(), size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying memory to d_B: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    // Set up the execution configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    addVectorsGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Synchronize to check for any errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    // Wait for GPU to finish before accessing on host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA synchronize failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    // Copy the result vector from device memory to host memory
    err = cudaMemcpy(C.data(), d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Error copying memory from d_C: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    // Set the size of the vectors
    int N = 5;

    // Initialize the vectors
    std::vector<int> A(N), B(N), C_CPU(N), C_GPU(N);

    // Fill the vectors with random values
    std::srand(std::time(0));
    for (int i = 0; i < N; ++i) {
        A[i] = std::rand() % 100;
        B[i] = std::rand() % 100;
    }

    // Add vectors on CPU
    addVectorsCPU(A, B, C_CPU, N);

    // Add vectors on GPU
    addVectorsOnGPU(A, B, C_GPU, N);

    // Verify the results
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (C_CPU[i] != C_GPU[i]) {
            success = false;
            std::cout << "Mismatch at index " << i << ": CPU = " << C_CPU[i] << ", GPU = " << C_GPU[i] << std::endl;
            break;
        }
    }

    if (success) {
        std::cout << "Vectors were added correctly!" << std::endl;
    } else {
        std::cout << "There was an error in the vector addition." << std::endl;
    }

    return 0;
}
