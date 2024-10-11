#include <stdlib.h>
#include <stdio.h>
#include <cblas.h>
#include <iostream>
#include <chrono>

int main(int argc, char **argv){
    int size = 2000,
        num_itr = 100;

    double *A = (double*) malloc(size * size * sizeof(double)),
           *B = (double*) malloc(size * size * sizeof(double)),
           *C = (double*) malloc(size * size * sizeof(double));

    for(int i=0; i<size*size; i++){
        A[i] = 0.1;
    }
    for(int i=0; i<size*size; i++){
        B[i] = 1.2;
    }

    // run openBLAS
    auto t1 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<num_itr; i++){
        // BLAS call
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, size, size, size, 1.0, A, size, B, size, 0.0, C, size);    // C <- A @ B.T
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "finish in " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;

    free(A), free(B), free(C);

    return 0;
}