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

typedef void (*reduction_t)(int num_itr, int size, double *A, double *B, double *v);
typedef void (*dgemm_t)(int num_itr, int size, double *A, double *B, double *C);

void register_functions() {
   std::unordered_map<std::string, std::string> names{
      {"reduction", "_Z9reductioniiPdS_S_"},
      {"dgemm", "_Z5dgemmiiPdS_S_"}};
   VLC::register_func_names(names);
}

void launch_openmp(int vec_id, int num_itr, int size, double *A, double *B, double *v){
    void *handle = dlmopen(LM_ID_NEWLM, "libreduction.so", RTLD_NOW);
    if (handle == NULL) {
        fprintf(stderr, "Error in `dlmopen`: %s\n", dlerror());
        return;
    }

    // load functions from libraries
    auto reduction = VLC::load_func<reduction_t>(handle, "reduction");
    reduction(num_itr, size, A, B, v);
}

void launch_openblas(int vec_id, int num_itr, int size, double *A, double *B, double *C) {
    void *handle = dlmopen(LM_ID_NEWLM, "libdgemm.so", RTLD_NOW);
    if (handle == NULL) {
        fprintf(stderr, "Error in `dlmopen`: %s\n", dlerror());
        return;
    }

    // load functions from libraries
    auto dgemm = VLC::load_func<dgemm_t>(handle, "dgemm");
    dgemm(num_itr, size, A, B, C);
}


int main() {
    // initialize VLC environment
    VLC::Runtime vlc;
    vlc.initialize();

    // register functions used in VLC
    register_functions();

    std::cout << "Begin!" << std::endl;

    int size_mp = 10000,
    num_itr_mp = 100;

    double *A_mp = (double*) malloc(size_mp * size_mp * sizeof(double)),
           *B_mp = (double*) malloc(size_mp * size_mp * sizeof(double));

    for(int i=0; i<size_mp*size_mp; i++){
        A_mp[i] = 0.1;
    }
    for(int i=0; i<size_mp*size_mp; i++){
        B_mp[i] = 1.2;
    }

    double v = 0.0;

    int size_blas = 2000,
    num_itr_blas = 100;

    double *A_blas = (double*) malloc(size_blas * size_blas * sizeof(double)),
           *B_blas = (double*) malloc(size_blas * size_blas * sizeof(double)),
           *C_blas = (double*) malloc(size_blas * size_blas * sizeof(double));

    for(int i=0; i<size_blas*size_blas; i++){
        A_blas[i] = 0.1;
    }
    for(int i=0; i<size_blas*size_blas; i++){
        B_blas[i] = 1.2;
    }

    // run openMP and openBLAS in parallel
    std::vector<std::thread> t(2); 
    
    auto t1 = std::chrono::high_resolution_clock::now();

    t[0] = std::thread(launch_openblas, 0, num_itr_blas, size_blas, A_blas, B_blas, C_blas);
    t[1] = std::thread(launch_openmp, 1, num_itr_mp, size_mp, A_mp, B_mp,  &v);

    t[0].join();
    t[1].join();

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "finish in " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;

    std::cout << "result = " << v << std::endl;

    free(A_blas), free(B_blas), free(C_blas), free(A_mp), free(B_mp);

    return 0;
}