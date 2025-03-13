#include <thread>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <pthread.h>
#include "matmul.h"

int main() {
    int N = 8192;
    int num_small = 20;

    double *A = (double *) malloc(N * N * sizeof(double));   
    double *B = (double *) malloc(N * N * sizeof(double));
    double *C = (double *) malloc(N * N * sizeof(double));
    std::fill_n(A, N * N, 1.0);
    std::fill_n(B, N * N, 2.0);

    int N_small = N_big/4;
    double *A_small = (double *) malloc(N_small * N_small * sizeof(double));   
    double *B_small = (double *) malloc(N_small * N_small * sizeof(double));
    double *C_small = (double *) malloc(N_small * N_small * sizeof(double));
    std::fill_n(A_small, N_small * N_small, 3.0);
    std::fill_n(B_small, N_small * N_small, 4.0);

    auto t1 = std::chrono::high_resolution_clock::now();
    matmul(N, A, B, C);

    for (int i = 0; i < num_small; i++) {
        matmul(N_small, A_small, B_small, C_small);
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "matmul compute time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;

    free(A);
    free(B);
    free(C);
    free(A_small);
    free(B_small);
    free(C_small);

    return 0;
}