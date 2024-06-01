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

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

typedef int (*test_cuda_t)(int);

pthread_barrier_t barrier;

void register_functions() {
    std::unordered_map<std::string, std::string> names{
    {"test_cuda", "test_cuda"}};
    VLC::register_func_names(names);
}

void launch(int vec_id, int dev_id) {
    std::cout << "vec " << vec_id << " begin!" << std::endl;

    void *handle = dlmopen(LM_ID_NEWLM, "libhelloworld.so", RTLD_NOW);
    if (handle == NULL) {
        fprintf(stderr, "Error in `dlmopen`: %s\n", dlerror());
        return;
    }

    std::cout << "vec " << vec_id << " loaded!" << std::endl;

    // load functions from libraries
    auto test_cuda = VLC::load_func<test_cuda_t>(handle, "test_cuda");
    std::cout << "test_cuda " << vec_id << " loaded!" << std::endl;

    // if (dev_id == 0)
    //     std::this_thread::sleep_for(std::chrono::seconds(10));
    int result = test_cuda(dev_id);
    std::cout << "[" << dev_id << "] " << "test_cuda: result=" << result << std::endl;
    
}

int main() {
    // initialize VLC environment
    // VLC::Runtime vlc;
    // vlc.initialize();

    // checkCudaErrors((cudaSetDevice(0)));

    // register functions used in VLC
    register_functions();

    std::cout << "Begin!" << std::endl;
    int num_work = 2;

    pthread_barrier_init(&barrier, NULL, num_work); 
    std::cout << "pthread_barrier_init!" << std::endl;

    std::vector<std::thread> t(num_work);
    std::cout << "declare thread!" << std::endl;

    for (int i = 0; i < num_work; i++) {
        t[i] = std::thread(launch, i, i);
        std::cout << "launched thread" << i << std::endl;
        // t[i].join();
        // checkCudaErrors(cudaDeviceSynchronize());
    }

    for (int i = 0; i < num_work; i++) {
        t[i].join();
    }

    // print_mem_info();

    return 0;
}