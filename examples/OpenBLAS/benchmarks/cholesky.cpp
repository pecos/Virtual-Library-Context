/**
 * This is modified from https://github.com/heilokchow/math_lib_benchmark/blob/master/bench.cpp
 */

#include <iostream>
#include <chrono>
#include <stdio.h>
#include <random>
#include <string>

#include <cblas.h>
#include <lapack.h>

void F_POTRF(double* const& y, const int& n, const int& nrep, pthread_barrier_t* barrier) {
    double t = 0.0;
    double* y1 = new double[n * n];
    double* y2 = new double[n * n];

    
    // std::random_device device;
    std::mt19937 generator(101);
    std::normal_distribution<double> normal(0.0, 1.0);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int p = i * n + j;
            int q = j * n + i;
            y1[p] = normal(generator);
            y2[q] = y1[p];
        }
    }
    
    pthread_barrier_wait(barrier);
    auto start_time = std::chrono::system_clock::now();
    for (int i = 0; i < nrep; i++) {   
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, y1, n, y2, n, 0.0, y, n);

        int info;
        // Call dpotrf_ directly
        char uplo = 'U'; // Use 'U' for upper triangular or 'L' for lower triangular
        dpotrf_(&uplo, &n /* N */, y /* A */, &n /* LDA */, &info, 1);

        if (info != 0) {
            std::cerr << "Error: dpotrf_ failed with info = " << info << "\n";
        }
    }
    auto end_time = std::chrono::system_clock::now();
    std::cout << "cholesky runtime: " << ((std::chrono::duration<double>) (end_time - start_time)).count() << "s\n";
    
    delete[] y1;
    delete[] y2;
}

int cholesky(int argc, char** argv, pthread_barrier_t* barrier)
{
    if (argc != 3) {
        std::cout << "No. of input: " << argc << std::endl;
        puts("./a.out <dim> <nrep>");
        exit(0);
    }

    int n = std::stoi(argv[1]);
    int nrep = std::stoi(argv[2]);

    if (n >= 46341) {
        std::cout << "n should be less than 46341 \n";
        exit(0);
    }

    double* y = new double[n*n];
    int p = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            p = i * n + j;
            y[p] = 0.0;
        }
    }   

    // // Perform Cholesky Decomposition
    F_POTRF(y, n, nrep, barrier);

    delete[] y;
    return 0;
}
