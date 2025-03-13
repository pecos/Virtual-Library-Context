#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <thread>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <pthread.h>
#include <dlfcn.h>
#include <unistd.h>
#include <fstream>
#include <numeric>

#include "VLC/runtime.h"
#include "VLC/loader.h"

typedef void (*matmul_t)(int N, double *A, double *B, double *C);

void register_functions() {
   std::unordered_map<std::string, std::string> names{
      {"matmul", "_Z6matmuliPdS_S_"}};
   VLC::Loader::register_func_names(names);
}

void launch(int vlc_id, int N, double *A, double *B, double *C) {
    VLC::Context vlc(vlc_id, gettid());

    // please change the number based on your system
    if (vlc_id == 1)
        vlc.avaliable_cpu("7-23");
    else
        vlc.avaliable_cpu("0-6");
    VLC::register_vlc(&vlc);

    VLC::Loader loader("libmatmul.so", vlc_id, false);

    // load functions from libraries
    auto matmul = loader.load_func<matmul_t>("matmul");

    auto start = std::chrono::high_resolution_clock::now();
    if (vlc_id == 1) {  // one larger matmul
        matmul(N, A, B, C);
    } else {
        for (int i = 0; i < 20; i++) {  // 8 smaller matmul
            matmul(N, A, B, C);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "matmul " << N << " compute time:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

}

int main() {
    VLC::Runtime vlc;
    vlc.initialize();
    register_functions();

    int N = 8192;

    int N_big = 8192;
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
    auto large_matmul = std::thread(launch, 1, N, A, B, C);
    auto small_matmul = std::thread(launch, 2, N_small, A_small, B_small, C_small);

    large_matmul.join();
    small_matmul.join();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "matmul finish in: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;

    free(A);
    free(B);
    free(C);
    free(A_small);
    free(B_small);
    free(C_small);

    return 0;
}
