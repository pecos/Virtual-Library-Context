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

#include "VLC/runtime.h"
#include "VLC/loader.h"

std::chrono::_V2::system_clock::time_point runtime_init_start;
std::chrono::_V2::system_clock::time_point runtime_init_end;
std::chrono::_V2::system_clock::time_point vlc_init_start;
std::chrono::_V2::system_clock::time_point vlc_init_end;

void F_GESV(int vlc_id, double* const& y, double* const& b, const int& n, const int& nrep) {
    vlc_init_start = std::chrono::high_resolution_clock::now();
    VLC::Context vlc(vlc_id, gettid());
    vlc.avaliable_cpu("0-23");
    VLC::register_vlc(&vlc);
    VLC::Loader loader("/lib/x86_64-linux-gnu/libopenblas.so.0", vlc_id, true);
    vlc_init_end = std::chrono::high_resolution_clock::now();
    std::cout << "PERF: VLC init finished in " << std::chrono::duration_cast<std::chrono::milliseconds>((vlc_init_end - vlc_init_start) + (runtime_init_end - runtime_init_start)).count() << "ms" << std::endl;
     
    double t = 0.0;
    double* y1 = new double[n * n];
    double* y2 = new double[n * n];
    auto t0 = std::chrono::system_clock::now();
    auto t1 = std::chrono::system_clock::now();
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
        b[i] = normal(generator);
    }

    for (int i = 0; i < nrep; i++) {
        memset(y, 0.0,  sizeof(double) * n * n);
        int* ipiv = new int[n];
        int info;

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, y1, n, y2, n, 0.0, y, n);

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
    runtime_init_start = std::chrono::high_resolution_clock::now();
    VLC::Runtime vlc; // initialize VLC environment
    vlc.initialize();
    runtime_init_end = std::chrono::high_resolution_clock::now();

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
    double* b = new double[n];
    int p = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            p = i * n + j;
            y[p] = 0.0;
        }
        b[i] = 0;
    }   

    std::thread t;

    // Perform Martix Multiplication
    t = std::thread(F_GESV, 1, y, b, n, nrep);
    t.join();

    delete[] y;
    delete[] b;
    return 0;
}