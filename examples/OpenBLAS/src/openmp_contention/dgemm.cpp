#include <stdlib.h>
#include <stdio.h>
#include <cblas.h>
#include <iostream>
#include <chrono>

void dgemm(int num_itr, int size, double *A, double *B, double *C) {
    for(int i=0; i<num_itr; i++){
        // BLAS call
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, size, size, size, 1.0, A, size, B, size, 0.0, C, size);    // C <- A @ B.T
    }

}