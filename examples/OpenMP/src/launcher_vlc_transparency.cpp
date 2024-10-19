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
#include <omp.h>
#include <unistd.h>
#include <fstream>
#include <sys/mman.h>

#include "VLC/runtime.h"
#include "VLC/loader.h"

const int REPEAT = 400;
const int NUM_CORE = 64;

int add(int * first, int * second, int * result, int num_items) {
    int th_id;

    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel private(th_id)
    {   
        // printf("addmp: cpu_id: %d\n", sched_getcpu());
        th_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int batch_size = num_items / num_threads;

        for (int j = 0; j < REPEAT; j++) {
            for (int i = 0; i < batch_size; i++) {
                result[th_id * batch_size + i] = first[th_id * batch_size + i] + second[th_id * batch_size + i];
            }
        }

    }
    
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "addmp: finish in " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;

    return 0;
}

void launch(int vlc_id) {
    std::cout << "VLC " << vlc_id << " is created" << std::endl;
    VLC::Context vlc(vlc_id, gettid());

    // please change the number based on your system
    if (vlc_id == 1)
        vlc.avaliable_cpu("0-11");
    else
        vlc.avaliable_cpu("12-23");
    VLC::register_vlc(&vlc);

    VLC::Loader loader("/usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so", vlc_id, true);

    int size = 12000000;
    int * v1 = (int *) mmap(NULL, size * sizeof(int), PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    int * v2 = (int *) mmap(NULL, size * sizeof(int), PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    int * result = (int *) mmap(NULL, size * sizeof(int), PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);

    // std::vector<int> result(size, 0);
    // std::vector<int> v1(size, 1);
    // std::vector<int> v2(size, 1);

    add(v1, v2, result, size);

    printf("%d: quit\n", vlc_id);
}


int main() {
    // initialize VLC environment
    VLC::Runtime vlc;
    vlc.initialize();

    std::cout << "Begin!" << std::endl;

    int num_work = 2;

    std::vector<std::thread> t(num_work);
    std::cout << "declare thread!" << std::endl;

    for (int i = 0; i < num_work; i++) {
        t[i] = std::thread(launch, i+1);
    }

    for (int i = 0; i < num_work; i++) {
        t[i].join();
    }

    return 0;
}