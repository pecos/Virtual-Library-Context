/**
 * This is modified from https://github.com/heilokchow/math_lib_benchmark/blob/master/bench.cpp
 */

#include <iostream>
#include <chrono>
#include <stdio.h>
#include <random>
#include <string>
#include <cstring>

#include <cblas.h>
#include <lapack.h>

void F_GEMM(double* const& x, double* const& y, double* const& z, const int& n, const int& nrep, pthread_barrier_t* barrier) {
    double t = 0.0;
    int p = 0;
    // std::random_device device;
    std::mt19937 generator(101);
    std::normal_distribution<double> normal(0.0, 1.0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            p = i * n + j;
            y[p] = normal(generator);
            z[p] = normal(generator);
        }
    }

    pthread_barrier_wait(barrier);
    auto start_time = std::chrono::system_clock::now();
    for (int i = 0; i < nrep; i++)
    {
        memset(x, 0.0,  sizeof(double) * n * n);

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, y, n, z, n, 0.0, x, n);
    }
    auto end_time = std::chrono::system_clock::now();
    std::cout << "gemm runtime: " << ((std::chrono::duration<double>) (end_time - start_time)).count() << "s\n";
}

int gemm(int argc, char** argv, pthread_barrier_t* barrier)
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

    double* x = new double[n*n];
    double* y = new double[n*n];
    double* z = new double[n*n];
    int p = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            p = i * n + j;
            x[p] = 0.0;
            y[p] = 0.0;
            z[p] = 0.0;
        }
    }   

    // Perform Martix Multiplication
    F_GEMM(x, y, z, n, nrep, barrier);

    delete[] x;
    delete[] y;
    delete[] z;
    return 0;
}
