#include <cblas.h>
#include <stdio.h>
#include <algorithm>

void matmul(int N)
{
    double *A = (double *) malloc(N * N * sizeof(double));   
    double *B = (double *) malloc(N * N * sizeof(double));
    double *C = (double *) malloc(N * N * sizeof(double));
    std::fill_n(A, N * N, 1.0);
    std::fill_n(B, N * N, 2.0);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, A, N, B, N, 0, C, N);

    free(A);
    free(B);
    free(C);
}