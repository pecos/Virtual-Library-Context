/**
 * This is modified from https://github.com/heilokchow/math_lib_benchmark/blob/master/bench.cpp
 */

#include <iostream>
#include <chrono>
#include <stdio.h>
#include <random>
#include <string>
#include <thread>

#include <cblas.h>
#include <lapack.h>

#include "VLC/runtime.h"
#include "VLC/loader.h"

std::chrono::_V2::system_clock::time_point runtime_init_start;
std::chrono::_V2::system_clock::time_point runtime_init_end;
std::chrono::_V2::system_clock::time_point vlc_init_start;
std::chrono::_V2::system_clock::time_point vlc_init_end;

void F_POTRF(int vlc_id, double* const& y, const int& n, const int& nrep) {
    vlc_init_start = std::chrono::high_resolution_clock::now();
    VLC::Context vlc(vlc_id, gettid());
    vlc.avaliable_cpu("0-23");
    VLC::register_vlc(&vlc);
    VLC::Loader loader("/lib/x86_64-linux-gnu/libopenblas64.so.0", vlc_id, true);
    vlc_init_end = std::chrono::high_resolution_clock::now();
    std::cout << "PERF: VLC init finished in " << std::chrono::duration_cast<std::chrono::milliseconds>((vlc_init_end - vlc_init_start) + (runtime_init_end - runtime_init_start)).count() << "ms" << std::endl;

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
        }
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, y1, n, y2, n, 0.0, y, n);

        lapack_int info;
        t0 = std::chrono::system_clock::now();
        // Call dpotrf_ directly
        char uplo = 'U'; // Use 'U' for upper triangular or 'L' for lower triangular
        dpotrf_(&uplo, &n /* N */, y /* A */, &n /* LDA */, &info, 1);
        t1 = std::chrono::system_clock::now();

        if (info != 0) {
            std::cerr << "Error: dpotrf_ failed with info = " << info << "\n";
        }

        std::chrono::duration<double> elapsed = t1 - t0;
        std::cout << "DPOTRF elapsed time: " << elapsed.count() << "s\n";
        t += static_cast<double>(elapsed.count());
    }

    std::cout << "DPOTRF average time: " << t / nrep << "s\n";

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
    int p = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            p = i * n + j;
            y[p] = 0.0;
        }
    }   

    std::thread t;

    t = std::thread(F_POTRF, 1, y, n, nrep);
    t.join();

    delete[] y;
    return 0;
}
