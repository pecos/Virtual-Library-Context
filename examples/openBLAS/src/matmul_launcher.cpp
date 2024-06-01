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

typedef void (*matmul_t)(int N);

pthread_barrier_t barrier;

std::chrono::_V2::system_clock::time_point t0;
std::chrono::_V2::system_clock::time_point t1;

void register_functions() {
   std::unordered_map<std::string, std::string> names{
      {"matmul", "_Z6matmuli"}};
   VLC::register_func_names(names);
}

void launch(int vec_id, int N) {
    void *handle = dlmopen(LM_ID_NEWLM, "libmatmul.so", RTLD_NOW);
    if (handle == NULL) {
        fprintf(stderr, "Error in `dlmopen`: %s\n", dlerror());
        return;
    }

    // load functions from libraries
    auto matmul = VLC::load_func<matmul_t>(handle, "matmul");

    if (vec_id == 0) {  // one larger matmul
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

    // register functions used in VLC
    register_functions();

    std::cout << "Begin!" << std::endl;
    int num_work = 2;

    pthread_barrier_init(&barrier, NULL, num_work);

    std::vector<std::thread> t(num_work);

    
    for (int i = 0; i < num_work; i++) {
        t[i] = std::thread(launch, i, N);
    }

    for (int i = 0; i < num_work; i++) {
        t[i].join();
    }

    return 0;
}