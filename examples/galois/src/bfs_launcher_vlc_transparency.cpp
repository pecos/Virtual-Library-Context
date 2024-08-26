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

#include "bfs.h"

pthread_barrier_t barrier;

std::vector<Graph *> graph_ptr(2, nullptr);

std::chrono::_V2::system_clock::time_point t0;
std::chrono::_V2::system_clock::time_point t1;

void launch(int id, int rounds, int skip_rounds, std::string filename) {
    std::cout << "VLC " << id << " is created" << std::endl;
    VLC::Loader loder("/home/yyan/vlc/examples/galois/lib/Galois/build/libgalois/libgalois_shmem.so", id, false);

    init_galois(10000);

    int done = 0;
    for (int rd = 0; rd < rounds + 1; rd++) {
        // sync so everyone runs concurrently
        pthread_barrier_wait(&barrier);
        if (id == 1) {
            // graph is shared between instances of galois
            printf("%d: load_file() starts\n", id);
            graph_ptr[done % 2] = load_file(filename);
        }
        
        if (rd < skip_rounds || done == rounds) {
            continue;
        }
        int source = 0;
        int report = 5;
        int slot = id - 1;

        printf("%d: bfs() starts\n", id);
        if (id == 1)
            bfs(graph_ptr[done % 2], source, slot);
        else {
            bfs(graph_ptr[done % 2], source, slot);
            unsigned int d = read_distance(graph_ptr[done % 2], report, slot);
            printf("VLC %d round %d: distance from %d to %d at slot %d is %d\n", id, rd, source, report, slot, d);
            free(graph_ptr[done % 2]);
        }
        done++;
    }
}

int main() {
    // initialize VLC environment
    VLC::Runtime vlc;
    vlc.initialize();

    std::cout << "Begin!" << std::endl;
    int num_work = 2;

    std::vector<std::thread> t(num_work);
    
    for (int i = 0; i < num_work; i++) {
        t[i] = std::thread(launch, i+1, 10, i, "/var/local/yyan/graphs/rmat26.gr");
        std::cout << "launched vlc" << i << std::endl;
   }

    for (int i = 0; i < num_work; i++) {
        t[i].join();
    }

    return 0;
}