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

void F_GESV(double* const& v, double* const& y, double* const& b, const int& n, const int& nrep) {
    double t = 0.0;
    double* y1 = new double[n * n];
    double* y2 = new double[n * n];
    auto t0 = std::chrono::system_clock::now();
    auto t1 = std::chrono::system_clock::now();
    std::random_device device;
    std::mt19937 generator(device());
    std::normal_distribution<double> normal(0.0, 1.0);

    for (int i = 0; i < nrep; i++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int p = i * n + j;
                int q = j * n + i;
                y1[p] = normal(generator);
                y2[q] = y1[p];
            }
            b[i] = normal(generator);
        }
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, y1, n, y2, n, 0.0, y, n);

        int* ipiv = new int[n];
        int info;
        t0 = std::chrono::system_clock::now();
        // Call dgesv_ directly
        int j = 0;
        dgesv_(&n, &j /* assuming nrhs is 1 */, y, &n /* LDA */, ipiv, b, &n /* LDB */, &info);
        t1 = std::chrono::system_clock::now();

        if (info != 0) {
            std::cerr << "Error: dgesv_ failed with info = " << info << "\n";
        }

        delete[] ipiv;

        std::chrono::duration<double> elapsed = t1 - t0;
        std::cout << "DGESV elapsed time: " << elapsed.count() << "s\n";
        t += static_cast<double>(elapsed.count());
    }

    std::cout << "DGESV average time: " << t / nrep << "s\n";

    delete[] y1;
    delete[] y2;
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

    double* y = new double[n*n];
    double* v = new double[n];
    double* b = new double[n];
    int p = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            p = i * n + j;
            y[p] = 0.0;
        }
        v[i] = 0;
        b[i] = 0;
    }   


    // Perform Matrix Inversion
    F_GESV(v, y, b, n, nrep);

    delete[] y;
    delete[] v;
    delete[] b;
    return 0;
}