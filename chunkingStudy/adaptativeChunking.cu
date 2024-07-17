#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void vectorAddKernel(int *a, int *b, int *c, int chunkSize, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < offset + chunkSize) {
        c[idx] = a[idx] + b[idx];
    }
}

void addVectors(int *a, int *b, int *c, int size){
    for (int i=0; i<size; i++) {
        c[i] = a[i] + b[i];
    }
}


void adaptiveChunking(int *h_a, int *h_b, int *h_c, int size) {
    int minChunkSize = 256; // Minimum chunk size
    int maxChunkSize = 1024; // Maximum chunk size
    int currentChunkSize = minChunkSize;
    int numChunks = size / currentChunkSize;
    int remainingSize = size % currentChunkSize;

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size * sizeof(int));
    cudaMalloc(&d_b, size * sizeof(int));
    cudaMalloc(&d_c, size * sizeof(int));

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numChunks; ++i) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        int offset = i * currentChunkSize;
        cudaMemcpyAsync(d_a + offset, h_a + offset, currentChunkSize * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_b + offset, h_b + offset, currentChunkSize * sizeof(int), cudaMemcpyHostToDevice, stream);
        vectorAddKernel<<<(currentChunkSize + 255) / 256, 256, 0, stream>>>(d_a, d_b, d_c, currentChunkSize, offset);
        cudaMemcpyAsync(h_c + offset, d_c + offset, currentChunkSize * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);

        // Adaptive adjustment of chunk size based on performance feedback
        // This is a simplified example; in a real scenario, you'd measure performance metrics
        if (i % 2 == 0 && currentChunkSize < maxChunkSize) {
            currentChunkSize *= 2; // Increase chunk size
        } else if (i % 2 != 0 && currentChunkSize > minChunkSize) {
            currentChunkSize /= 2; // Decrease chunk size
        }
    }

    // Handle the remaining data
    if (remainingSize > 0) {
        cudaStream_t lastStream;
        cudaStreamCreate(&lastStream);
        int offset = numChunks * currentChunkSize;
        cudaMemcpyAsync(d_a + offset, h_a + offset, remainingSize * sizeof(int), cudaMemcpyHostToDevice, lastStream);
        cudaMemcpyAsync(d_b + offset, h_b + offset, remainingSize * sizeof(int), cudaMemcpyHostToDevice, lastStream);
        vectorAddKernel<<<(remainingSize + 255) / 256, 256, 0, lastStream>>>(d_a, d_b, d_c, remainingSize, offset);
        cudaMemcpyAsync(h_c + offset, d_c + offset, remainingSize * sizeof(int), cudaMemcpyDeviceToHost, lastStream);
        cudaStreamSynchronize(lastStream);
        cudaStreamDestroy(lastStream);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "Total execution time: " << duration.count() << " ms" << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    int size = 4096;
    int *h_a = new int[size];
    int *h_b = new int[size];
    int *h_c = new int[size];

    // Initialize h_a and h_b
    for (int i = 0; i < size; ++i) {
        h_a[i] = i;
        h_b[i] = size - i;
    }

    //adaptiveChunking(h_a, h_b, h_c, size);
    addVectors(h_a, h_b, h_c, size);

    // Verify results
    bool correct = true;
    for (int i = 0; i < size; ++i) {
        if (h_c[i] != size) {
            correct = false;
            std::cout << "Error at index " << i << ": " << h_c[i] << std::endl;
            break;
        }
    }
    if (correct) {
        std::cout << "All results are correct." << std::endl;
    }

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    return 0;
}
