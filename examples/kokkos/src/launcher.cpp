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

typedef void (*kokkos_init_t)(int dev);

typedef void (*kokkos_finalize_t)();

typedef double (*kokkos_reduction_copy_to_device_t)(double* array, const int N, const int dev_id);

pthread_barrier_t barrier;

void register_functions() {
   std::unordered_map<std::string, std::string> names{
      {"kokkos_init", "_Z11kokkos_initi"},
      {"kokkos_finalize", "_Z15kokkos_finalizev"},
      {"kokkos_reduction_copy_to_device", "_Z31kokkos_reduction_copy_to_devicePdii"}};
   VLC::register_func_names(names);
}

void launch(int vec_id, int dev_id) {
    std::cout << "vec " << vec_id << " begin!" << std::endl;

    void *handle = dlmopen(LM_ID_NEWLM, "libkokkos_compute.so", RTLD_NOW);
    if (handle == NULL) {
        fprintf(stderr, "Error in `dlmopen`: %s\n", dlerror());
        return;
    }

    std::cout << "vec " << vec_id << " loaded!" << std::endl;

    // load functions from libraries
    auto kokkos_init = VLC::load_func<kokkos_init_t>(handle, "kokkos_init");
    std::cout << "kokkos_init " << vec_id << " loaded!" << std::endl;
    auto kokkos_finalize = VLC::load_func<kokkos_finalize_t>(handle, "kokkos_finalize");
    std::cout << "kokkos_finalize " << vec_id << " loaded!" << std::endl;
    auto kokkos_reduction_copy_to_device = VLC::load_func<kokkos_reduction_copy_to_device_t>(handle, "kokkos_reduction_copy_to_device");
    std::cout << "kokkos_reduction_copy_to_device " << vec_id << " loaded!" << std::endl;

    kokkos_init(dev_id);
    std::cout << "kokkos_init " << vec_id << " finished!" << std::endl;

    int n = 10000;

    std::vector<double> v(n);
    std::iota(v.begin(), v.end(), 1.0);

    double result = kokkos_reduction_copy_to_device(v.data(), n, dev_id);
    std::cout << "kokkos: result=" << result << std::endl;

    kokkos_finalize();

    pthread_barrier_wait(&barrier);
}

int main() {
   // initialize VLC environment
   VLC::Runtime vlc;
   vlc.initialize();

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
   }

   for (int i = 0; i < num_work; i++) {
      t[i].join();
   }

   // print_mem_info();

   return 0;
}