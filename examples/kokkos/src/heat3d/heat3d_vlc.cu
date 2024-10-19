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

std::chrono::_V2::system_clock::time_point vlc_init_start;
std::chrono::_V2::system_clock::time_point vlc_init_end;

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

int NUM_DEVICE = 1; // only support at most two GPUs

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
    VLC::Loader::register_func_names(names);
}

void launch(int vlc_id, int dev_id, int argc, char* argv[]) {
    std::cout << "VLC " << vlc_id << " is created" << std::endl;
    VLC::Context vlc(vlc_id, gettid());
    vlc.avaliable_cpu("0-23"); // please change the number based on your system
    VLC::register_vlc(&vlc);
    VLC::Loader loader("libheat3d.so", vlc_id, false);

    // load functions from libraries
    auto kokkos_initialize = loader.load_func<kokkos_initialize_t>("kokkos_initialize");
    auto kokkos_finalize = loader.load_func<kokkos_finalize_t>("kokkos_finalize");
    auto initialize_system = loader.load_func<initialize_system_t>("initialize_system");
    auto finalize_system = loader.load_func<finalize_system_t>("finalize_system");
    auto get_N = loader.load_func<get_N_t>("get_N");
    auto get_I = loader.load_func<get_I_t>("get_I");
    auto heat3d_phase1 = loader.load_func<heat3d_phase1_t>("heat3d_phase1");
    auto heat3d_exchange_data = loader.load_func<heat3d_exchange_data_t>("heat3d_exchange_data");
    auto heat3d_phase2 = loader.load_func<heat3d_phase2_t>("heat3d_phase2");
    vlc_init_end = std::chrono::high_resolution_clock::now();
    std::cout << "PERF: VLC init finished in " << std::chrono::duration_cast<std::chrono::milliseconds>(vlc_init_end - vlc_init_start).count() << "ms" << std::endl;

    kokkos_initialize(dev_id);
    heat3d_sys[dev_id] = initialize_system(dev_id, NUM_DEVICE, argc, argv);
    std::cout << "heat3d system " << vlc_id << " initialized!" << std::endl;
   
    int I = get_I(heat3d_sys[dev_id]);
    int N = get_N(heat3d_sys[dev_id]);

    auto old_time = std::chrono::high_resolution_clock::now();
    auto time_begin = old_time;
    // main loop
    for (int t = 0; t <= N; t++) {
        heat3d_phase1(heat3d_sys[dev_id], t);
        pthread_barrier_wait(&barrier);
        checkCudaErrors(cudaDeviceSynchronize());
        if (NUM_DEVICE != 1) {
            heat3d_exchange_data(heat3d_sys[dev_id], heat3d_sys[1 - dev_id]);  // assume only 2 devices
        }
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
}

int main(int argc, char* argv[]) {
    vlc_init_start = std::chrono::high_resolution_clock::now();
    // initialize VLC environment
    VLC::Runtime vlc;
    vlc.initialize();

    // register functions used in VLC
    register_functions();

    std::cout << "Begin!" << std::endl;
    int num_work = NUM_DEVICE;  // only support at most 2 worker in this application

    pthread_barrier_init(&barrier, NULL, num_work);

    std::vector<std::thread> t(num_work);

    for (int i = 0; i < num_work; i++) {
        t[i] = std::thread(launch, i+1, i, argc, argv);
    }

    for (int i = 0; i < num_work; i++) {
        t[i].join();
    }

    return 0;
}