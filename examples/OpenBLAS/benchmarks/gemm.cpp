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

void F_GEMM(double* const& x, double* const& y, double* const& z, const int& n, const int& nrep) {
    double t = 0.0;
    int p = 0;
    auto t0 = std::chrono::system_clock::now();
    auto t1 = std::chrono::system_clock::now();
    std::random_device device;
    std::mt19937 generator(device());
    std::normal_distribution<double> normal(0.0, 1.0);
    for (int i = 0; i < nrep; i++)
    {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                p = i * n + j;
                y[p] = normal(generator);
                z[p] = normal(generator);
            }
        }

        t0 = std::chrono::system_clock::now();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, y, n, z, n, 0.0, x, n);
        t1 = std::chrono::system_clock::now();

        std::chrono::duration<double> elapsed = t1 - t0;
        std::cout << "DGEMM elapsed time: " << elapsed.count() << "s\n";
        t += static_cast<double>(elapsed.count());
    }
    std::cout << "DGEMM average time: " << t / nrep << "s\n";
}

int main(int argc, char** argv)
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
    F_GEMM(x, y, z, n, nrep);

    delete[] x;
    delete[] y;
    delete[] z;
    return 0;
}
