#include <hip/hip_runtime.h>
#include <iostream>
#include <memory>

#include "example_utils.hpp"

#define N 512
#define BLOCK_SIZE 256

// Declaration of the HIP kernel
__global__ void vector_add(float *C, const float *A, const float *B) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int hello(int dev_id) {
    float *A, *B, *C;
    float *d_A, *d_B, *d_C; // Device pointers

    // Allocate memory on the host
    // HIP_CHECK(hipSetDevice(dev_id));
    A = (float*)malloc(N*sizeof(float));
    B = (float*)malloc(N*sizeof(float));
    C = (float*)malloc(N*sizeof(float));

    // Initialize vectors on the host
    for(int i = 0; i < N; i++) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(i);
    }

    // Allocate memory on the device
    std::cout << "1" << std::endl;
    HIP_CHECK(hipMalloc(&d_A, N*sizeof(float)));
    std::cout << "2" << std::endl;
    HIP_CHECK(hipMalloc(&d_B, N*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, N*sizeof(float)));

    // Copy host vectors to device
    std::cout << "3" << std::endl;
    HIP_CHECK(hipMemcpy(d_A, A, N*sizeof(float), hipMemcpyHostToDevice));
    std::cout << "4" << std::endl;
    HIP_CHECK(hipMemcpy(d_B, B, N*sizeof(float), hipMemcpyHostToDevice));
    std::cout << "5" << std::endl;
    std::cout << "launch kernel" << std::endl;
    // Launch the vector addition kernel
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(vector_add, dim3(numBlocks), dim3(BLOCK_SIZE), 0, 0, d_C, d_A, d_B);

    // Copy result back to host
    HIP_CHECK(hipMemcpy(C, d_C, N*sizeof(float), hipMemcpyDeviceToHost));

    // Display the result
    for(int i = 0; i < 5; i++) {
        std::cout << "C[" << i << "] = " << C[i] << std::endl;
    }

    // Free device memory
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    // Free host memory
    free(A);
    free(B);
    free(C);

    return 0;
}
