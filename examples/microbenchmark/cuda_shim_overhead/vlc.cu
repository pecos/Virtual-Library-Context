/**
 * This App are try to count the overhead of VLC Service with CUDA
*/
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

typedef void (*test_cuda_t)();

pthread_barrier_t barrier;

void register_functions() {
    std::unordered_map<std::string, std::string> names{
    {"test_cuda", "test_cuda"}};
    VLC::register_func_names(names);
}

void launch() {
    void *handle = dlmopen(LM_ID_NEWLM, "libcudaoverhead.so", RTLD_NOW);
    if (handle == NULL) {
        fprintf(stderr, "Error in `dlmopen`: %s\n", dlerror());
        return;
    }


    // load functions from libraries
    auto test_cuda = VLC::load_func<test_cuda_t>(handle, "test_cuda");

    std::chrono::_V2::system_clock::time_point cuda_begin = std::chrono::high_resolution_clock::now();
    test_cuda();
    std::chrono::_V2::system_clock::time_point cuda_end  = std::chrono::high_resolution_clock::now();
    std::cout << "average cudaMemcpy finish in " << std::chrono::duration_cast<std::chrono::microseconds>(cuda_end - cuda_begin).count() / 120000.0 << "us" << std::endl;
}

int main() {
    // initialize VLC environment
    VLC::Runtime vlc;
    vlc.initialize();

    // to use cuda in this app so ensure VLCs Service is using
    cudaSetDevice(0);

    // register functions used in VLC
    register_functions();

    std::cout << "Begin!" << std::endl;
    int num_work = 1;


    std::vector<std::thread> t(num_work);
    std::cout << "declare thread!" << std::endl;

    for (int i = 0; i < num_work; i++) {
        t[i] = std::thread(launch);
    }
    for (int i = 0; i < num_work; i++) {
        t[i].join();
    }

    return 0;
}