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

#include "cuda_overhead.h"

int main() {
    cudaSetDevice(0);

    std::cout << "Begin!" << std::endl;

    std::chrono::_V2::system_clock::time_point cuda_begin = std::chrono::high_resolution_clock::now();
    test_cuda();
    std::chrono::_V2::system_clock::time_point cuda_end  = std::chrono::high_resolution_clock::now();
    std::cout << "average cudaMemcpy finish in " << std::chrono::duration_cast<std::chrono::microseconds>(cuda_end - cuda_begin).count() / 120000.0 << "us" << std::endl;

    return 0;
}