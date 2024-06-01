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

#include "hello.h"

typedef void (*hello_t)();

pthread_barrier_t barrier;

void register_functions() {
    std::unordered_map<std::string, std::string> names{
    {"hello", "hello"}};
    VLC::register_func_names(names);
}

int main() {
    // initialize VLC environment
    VLC::Runtime vlc;
    vlc.initialize();

    // to use cuda in this app so ensure VLCs stub is using
    hello();

    // register functions used in VLC
    register_functions();

    std::cout << "Begin!" << std::endl;
    int num_work = 1;

    std::vector<std::thread> t(num_work);
    std::cout << "declare thread!" << std::endl;

    void *handle = dlmopen(LM_ID_NEWLM, "libhello.so", RTLD_NOW);
    if (handle == NULL) {
        fprintf(stderr, "Error in `dlmopen`: %s\n", dlerror());
    }

    // load functions from libraries
    auto hello_vlc = VLC::load_func<hello_t>(handle, "hello");
    
    std::chrono::_V2::system_clock::time_point hello_begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 120000; i++) { 
        hello_vlc();
    }
    std::chrono::_V2::system_clock::time_point hello_end = std::chrono::high_resolution_clock::now();
    std::cout << "average hello finish in " << std::chrono::duration_cast<std::chrono::microseconds>(hello_end - hello_begin).count() / 120000.0 << "us" << std::endl;

    return 0;
}