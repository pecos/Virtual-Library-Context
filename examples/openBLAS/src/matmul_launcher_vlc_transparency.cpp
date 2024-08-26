#include <thread>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>

#include "VLC/runtime.h"
#include "VLC/loader.h"

#include <cblas.h>
#include <algorithm>

std::chrono::_V2::system_clock::time_point t0;
std::chrono::_V2::system_clock::time_point t1;

void matmul(int N)
{
    // transparency
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

void launcher(int vlc_id, int N)
{
    std::cout << "VLC " << vlc_id << " is created" << std::endl;
    VLC::Context vlc(vlc_id, gettid());

    // please change the number based on your system
    if (vlc_id == 1)
        vlc.avaliable_cpu("0-22");
    else
        vlc.avaliable_cpu("23");
    VLC::register_vlc(&vlc);

    VLC::Loader loader("/lib/x86_64-linux-gnu/libopenblas64.so.0", vlc_id, true);

    if (vlc_id == 1) {  // one larger matmul
        t0 = std::chrono::high_resolution_clock::now();
        matmul(N);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "matmul " << N << " finish in " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t0).count() << "ms" << std::endl;
    } else {
        t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 20; i++) {  // 8 smaller matmul
            matmul(N / 8);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "matmul " << N / 8 << " finish in " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;
    }
}

int main() {
    // initialize VLC environment
    VLC::Runtime vlc;
    vlc.initialize();

    int N = 8192;

    std::cout << "Begin!" << std::endl;
    int num_work = 2;

    std::vector<std::thread> t(num_work);

    for (int i = 0; i < num_work; i++) {
        t[i] = std::thread(launcher, i+1, N);
    }

    for (int i = 0; i < num_work; i++) {
        t[i].join();
    }

    return 0;
}