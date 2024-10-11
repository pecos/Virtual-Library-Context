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

#include "heat3d.h"

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

pthread_barrier_t barrier;

int NUM_DEVICE = 1; // only support at most one GPUs

// shared data
std::vector<void *> heat3d_sys(NUM_DEVICE, NULL);
std::vector<double> local_T_ave(NUM_DEVICE, 0);

void launch(int vlc_id, int dev_id, int argc, char* argv[]) {
    kokkos_initialize(dev_id);
    heat3d_sys[dev_id] = initialize_system(dev_id, NUM_DEVICE, argc, argv);
    std::cout << "heat3d system " << vlc_id << " initialized!" << std::endl;
   
    int I = get_I(heat3d_sys[dev_id]);
    int N = get_N(heat3d_sys[dev_id]);

    auto old_time = std::chrono::high_resolution_clock::now();
    auto time_begin = old_time;
    // main loop
    for (int t = 0; t <= N; t++) {
        std::cout << "!!" << std::endl;
        heat3d_phase1(heat3d_sys[dev_id], t);
        pthread_barrier_wait(&barrier);
        std::cout << "!!!" << std::endl;
        checkCudaErrors(cudaDeviceSynchronize());
        std::cout << "!!!!" << std::endl;
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
    std::cout << "Begin!" << std::endl;
    int num_work = NUM_DEVICE;  // only support at most 2 worker in this application

    std::vector<std::thread> t(num_work);

    launch(1, 0, argc, argv);

    return 0;
}