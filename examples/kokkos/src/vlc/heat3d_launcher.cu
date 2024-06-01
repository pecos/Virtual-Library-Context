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
#include <cuda.h>

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

typedef void (*kokkos_initialize_t)(int dev);
typedef void (*kokkos_finalize_t)();
typedef void *(*initialize_system_t)(int me, int nranks, int argc, char* argv[]);
typedef void (*finalize_system_t)(void *sys);
typedef int (*get_N_t)(void *sys);
typedef int (*get_I_t)  (void *sys);
typedef void (*heat3d_phase1_t)(void *sys, int t);
typedef void (*heat3d_exchange_data_t)(void *this_sys, void *other_sys);
typedef double (*heat3d_phase2_t)(void *sys);

pthread_barrier_t barrier;

int NUM_DEVICE = 2; // only support two GPUs

// shared data
std::vector<void *> heat3d_sys(NUM_DEVICE, NULL);
std::vector<double> local_T_ave(NUM_DEVICE, 0);

void register_functions() {
    std::unordered_map<std::string, std::string> names{
        {"kokkos_initialize", "_Z17kokkos_initializei"},
        {"kokkos_finalize", "_ZN6Kokkos8finalizeEv"},
        {"initialize_system", "_Z17initialize_systemiiiPPc"},
        {"finalize_system", "_Z15finalize_systemPv"},
        {"get_N", "_Z5get_NPv"},
        {"get_I", "_Z5get_IPv"},
        {"heat3d_phase1", "_Z13heat3d_phase1Pvi"},
        {"heat3d_exchange_data", "_Z20heat3d_exchange_dataPvS_"},
        {"heat3d_phase2", "_Z13heat3d_phase2Pv"},};
    VLC::register_func_names(names);
}

void launch(int vec_id, int dev_id, int argc, char* argv[]) {
    std::cout << "vec " << vec_id << " begin!" << std::endl;

    void *handle = dlmopen(LM_ID_NEWLM, "libheat3d.so", RTLD_NOW);
    if (handle == NULL) {
        fprintf(stderr, "Error in `dlmopen`: %s\n", dlerror());
        return;
    }

    std::cout << "vec " << vec_id << " loaded!" << std::endl;

    // load functions from libraries
    auto kokkos_initialize = VLC::load_func<kokkos_initialize_t>(handle, "kokkos_initialize");
    auto kokkos_finalize = VLC::load_func<kokkos_finalize_t>(handle, "kokkos_finalize");
    auto initialize_system = VLC::load_func<initialize_system_t>(handle, "initialize_system");
    auto finalize_system = VLC::load_func<finalize_system_t>(handle, "finalize_system");
    auto get_N = VLC::load_func<get_N_t>(handle, "get_N");
    auto get_I = VLC::load_func<get_I_t>(handle, "get_I");
    auto heat3d_phase1 = VLC::load_func<heat3d_phase1_t>(handle, "heat3d_phase1");
    auto heat3d_exchange_data = VLC::load_func<heat3d_exchange_data_t>(handle, "heat3d_exchange_data");
    auto heat3d_phase2 = VLC::load_func<heat3d_phase2_t>(handle, "heat3d_phase2");

    kokkos_initialize(dev_id);
    heat3d_sys[dev_id] = initialize_system(dev_id, NUM_DEVICE, argc, argv);
    std::cout << "heat3d system " << vec_id << " initialized!" << std::endl;
   
    int I = get_I(heat3d_sys[dev_id]);
    int N = get_N(heat3d_sys[dev_id]);

    auto old_time = std::chrono::high_resolution_clock::now();
    auto time_begin = old_time;
    // main loop
    for (int t = 0; t <= N; t++) {
        heat3d_phase1(heat3d_sys[dev_id], t);
        pthread_barrier_wait(&barrier);
        checkCudaErrors(cudaDeviceSynchronize());
        heat3d_exchange_data(heat3d_sys[dev_id], heat3d_sys[1 - dev_id]);  // assume only 2 devices
        local_T_ave[dev_id] = heat3d_phase2(heat3d_sys[dev_id]);
        pthread_barrier_wait(&barrier);
        double T_ave = std::reduce(local_T_ave.begin(), local_T_ave.end());
        // std::cout << "T_ave=" << T_ave << std::endl;
        if ((t % I == 0 || t == N) && (dev_id == 0)) {
            auto time = std::chrono::high_resolution_clock::now();
            if ((t == N) && (dev_id == 0)) {
                std::cout << "heat3D, Kokkos+VLC, t=" << t << ", T_ave=" << T_ave 
                    << ", time last iter(ms)=" << std::chrono::duration_cast<std::chrono::milliseconds>(time - old_time).count() 
                    << ", current runtime(ms)=" << std::chrono::duration_cast<std::chrono::milliseconds>(time - time_begin).count() << std::endl;
                old_time = time;
            }
        }
    }

    finalize_system(heat3d_sys[dev_id]);

    kokkos_finalize();
    pthread_barrier_wait(&barrier);
}

int main(int argc, char* argv[]) {
   // initialize VLC environment
   VLC::Runtime vlc;
   vlc.initialize();

   // register functions used in VLC
   register_functions();

   std::cout << "Begin!" << std::endl;
   int num_work = 2;  // only support 2 worker in this application

   pthread_barrier_init(&barrier, NULL, num_work);

   std::vector<std::thread> t(num_work);

    for (int i = 0; i < num_work; i++) {
        t[i] = std::thread(launch, i, i, argc, argv);
    }

    for (int i = 0; i < num_work; i++) {
        t[i].join();
    }

    return 0;
}