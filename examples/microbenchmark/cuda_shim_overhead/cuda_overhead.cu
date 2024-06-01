#include <stdio.h>
#include <stdlib.h>

#include "cuda_overhead.h"

// Size of array
#define NRUN 120000

// Main program
extern "C" void test_cuda()
{
	// checkCudaErrors(cudaSetDevice(device_id));
	// Number of bytes to allocate for N doubles
	size_t bytes = 4096;

	// Allocate memory for arrays A, B, and C on host
	double *A = (double*)malloc(bytes);

	// Allocate memory for arrays d_A, d_B, and d_C on device
	double *d_A,
	checkCudaErrors(cudaMalloc(&d_A, bytes));

	// Fill host arrays A and B
	for(int i=0; i < 4096 / sizeof(double*); i++)
	{
		A[i] = 1.0;
	}

	for(int i=0; i < 1; i++) {
		cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
	}
}