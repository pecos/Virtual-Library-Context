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

typedef int (*sum_t)(std::vector<int> &);
typedef int (*add_t)(int *first, int *second, int *result, int num_items);

void register_functions() {
   std::unordered_map<std::string, std::string> names{
      {"sum", "_Z3sumRSt6vectorIiSaIiEE"},
      {"add", "_Z3addPiS_S_i"}};
   VLC::Loader::register_func_names(names);
}

pthread_barrier_t barrier;

void launch_sum(int vlc_id) {
    std::cout << "VLC " << vlc_id << " is created" << std::endl;
    VLC::Context vlc(vlc_id, gettid());

    // please change the number based on your system
    vlc.avaliable_cpu("0-11");
    VLC::register_vlc(&vlc);

    VLC::Loader loader("libsummp.so", vlc_id, false);

    auto sum = loader.load_func<sum_t>("sum");

    int size = 12000000;
    std::vector<int> v0(size, 1);

    printf("%d: sum() starts\n", vlc_id);
    sum(v0);
    
    pthread_barrier_wait(&barrier);
    printf("%d: quit\n", vlc_id);
}

void launch_add(int vlc_id) {
    std::cout << "VLC " << vlc_id << " is created" << std::endl;
    VLC::Context vlc(vlc_id, gettid());

    // please change the number based on your system
    vlc.avaliable_cpu("12-23");
    VLC::register_vlc(&vlc);
   
    VLC::Loader loader("libaddmp.so", vlc_id, false);

    auto add = loader.load_func<add_t>("add");

    int size = 12000000;
    int * v1 = (int *) mmap(NULL, size * sizeof(int), PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    int * v2 = (int *) mmap(NULL, size * sizeof(int), PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    int * result = (int *) mmap(NULL, size * sizeof(int), PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);

    printf("%d: add() starts\n", vlc_id);
    add(v1, v2, result, size);

    pthread_barrier_wait(&barrier);
    printf("%d: quit\n", vlc_id);
}

int main() {
    // initialize VLC environment
    VLC::Runtime vlc;
    vlc.initialize();

    register_functions();

    std::cout << "Begin!" << std::endl;

    pthread_barrier_init(&barrier, NULL, 2);

    std::vector<std::thread> t(2);

    t[0] = std::thread(launch_sum, 1);
    t[1] = std::thread(launch_add, 2);
    
    t[0].join();
    t[1].join();

    return 0;
}