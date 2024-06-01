#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <thread>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <dlfcn.h>
#include <unistd.h>
#include <fstream>
#include <numeric>

#include "eign.h"

std::chrono::_V2::system_clock::time_point t0;
std::chrono::_V2::system_clock::time_point t1;

int main() {
    int N = 10000;
    int N_ev = 10;

    std::cout << "Begin!" << std::endl;
    int num_work = 2;
    
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_work; i++) {
        eign(N, N_ev);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "matmul finish in " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t0).count() << "ms" << std::endl;

    return 0;
}