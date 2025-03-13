#include <cblas.h>
#include <stdio.h>
#include <algorithm>

void matmul(int N, double *A, double *B, double *C)
{
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, A, N, B, N, 0, C, N);
}