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

std::chrono::_V2::system_clock::time_point t0;
std::chrono::_V2::system_clock::time_point t1;

typedef int (*eign_t)(int N, int N_ev);

void register_functions() {
   std::unordered_map<std::string, std::string> names{
      {"eign", "_Z4eignii"}};
   VLC::register_func_names(names);
}

void launch(int vec_id, int N, int N_ev) {
    void *handle = dlmopen(LM_ID_NEWLM, "libeign.so", RTLD_NOW);
    if (handle == NULL) {
        fprintf(stderr, "Error in `dlmopen`: %s\n", dlerror());
        return;
    }

    // load functions from libraries
    auto eign = VLC::load_func<eign_t>(handle, "eign");

    t0 = std::chrono::high_resolution_clock::now();
    eign(N, N_ev);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "eign finish in " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t0).count() << "ms" << std::endl;
}

int main() {
    // initialize VLC environment
    VLC::Runtime vlc;
    vlc.initialize();

    int N = 10000;
    int N_ev = 10;

    // register functions used in VLC
    register_functions();

    std::cout << "Begin!" << std::endl;
    int num_work = 2;

    std::vector<std::thread> t(num_work);
    
    for (int i = 0; i < num_work; i++) {
        t[i] = std::thread(launch, i, N, N_ev);
    }

    for (int i = 0; i < num_work; i++) {
        t[i].join();
    }

    return 0;
}