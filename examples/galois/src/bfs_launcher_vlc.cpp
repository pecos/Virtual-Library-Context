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

#include "VLC/runtime.h"
#include "VLC/loader.h"

pthread_barrier_t barrier;

std::vector<void*> graph_ptr(2, nullptr);

typedef void (*init_galois_t)(int);
typedef void* (*load_file_t)(const std::string&);
typedef void (*bfs_t)(void*, int, int);
typedef unsigned int (*read_distance_t)(void*, int, int);

void register_functions() {
    std::unordered_map<std::string, std::string> names{
        {"init_galois", "_Z11init_galoisi"},
        {"load_file", "_Z9load_fileRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE"},
        {"bfs", "_Z3bfsPN6galois6graphs12LC_CSR_GraphI8NodeDatavLb1ELb0ELb0EvEEii"},
        {"read_distance", "_Z13read_distancePN6galois6graphs12LC_CSR_GraphI8NodeDatavLb1ELb0ELb0EvEEii"}};
    VLC::Loader::register_func_names(names);
}

void launch(int vlc_id, int rounds, int skip_rounds, std::string filename) {
    std::cout << "VLC " << vlc_id << " begin!" << std::endl;
    VLC::Context vlc(vlc_id, gettid());

    // please change the number based on your system
    if (vlc_id == 1)
        vlc.avaliable_cpu("0-11");
    else
        vlc.avaliable_cpu("12-23");
    VLC::register_vlc(&vlc);

    VLC::Loader loader("libbfsgalois.so", vlc_id, false);

    // load functions from libraries
    auto init_galois = loader.load_func<init_galois_t>("init_galois");
    auto load_file = loader.load_func<load_file_t>("load_file");
    auto bfs = loader.load_func<bfs_t>("bfs");
    auto read_distance = loader.load_func<read_distance_t>("read_distance");

    printf("%d: init_galois() starts\n", vlc_id);
    init_galois(10000);

    int done = 0;
    for (int rd = 0; rd < rounds + 1; rd++) {
        // sync so everyone runs concurrently
        pthread_barrier_wait(&barrier);
        if (vlc_id == 1) {
            // graph is shared between instances of galois
            printf("%d: load_file() starts\n", vlc_id);
            graph_ptr[done % 2] = load_file(filename);
        }
        
        if (rd < skip_rounds || done == rounds) {
            continue;
        }
        int source = 0;
        int report = 5;
        int slot = vlc_id-1;

        printf("%d: bfs() starts\n", vlc_id);
        if (vlc_id == 1)
            bfs(graph_ptr[done % 2], source, slot);
        else {
            bfs(graph_ptr[done % 2], source, slot);
            unsigned int d = read_distance(graph_ptr[done % 2], report, slot);
            printf("VLC %d round %d: distance from %d to %d at slot %d is %d\n", vlc_id, rd, source, report, slot, d);
            free(graph_ptr[done % 2]);
        }
        done++;
    }
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
    std::vector<std::thread> t(num_work);

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_work; i++) {
        t[i] = std::thread(launch, i+1, 10, i, "/var/local/yyan/graphs/twitter.gr");
        std::cout << "launched vec" << i << std::endl;
    }

    for (int i = 0; i < num_work; i++) {
        t[i].join();
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "App finish in " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;

    return 0;
}