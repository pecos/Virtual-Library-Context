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
#include "testgalois.h"
#include <unistd.h>
#include <fstream>

pthread_barrier_t barrier;

static void print_mem_info() {
    int tSize = 0, resident = 0, share = 0;
    std::ifstream buffer("/proc/self/statm");
    buffer >> tSize >> resident >> share;
    buffer.close();

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    double rss = resident * page_size_kb;
    std::cout << "RSS - " << rss << " kB\n";

    double shared_mem = share * page_size_kb;
    std::cout << "Shared Memory - " << shared_mem << " kB\n";

    std::cout << "Private Memory - " << rss - shared_mem << "kB\n";
}

void launch(int tag) {
    std::cout << "thread " << tag << " begin!" << std::endl;

    printf("%d: test() starts\n", tag);
    test();
    
    printf("%d: quit\n", tag);
    pthread_barrier_wait(&barrier);
}

int main() {
    std::cout << "Begin!" << std::endl;
    int size = 120000000;

    int num_work = 2;

    pthread_barrier_init(&barrier, NULL, num_work);
    std::cout << "pthread_barrier_init!" << std::endl;

    std::vector<std::thread> t(num_work);
    std::cout << "declare thread!" << std::endl;

    for (int i = 0; i < num_work; i++) {
        t[i] = std::thread(launch, i);
        std::cout << "launched thread" << i << std::endl;
    }

    for (int i = 0; i < num_work; i++) {
        t[i].join();
    }

    // print_mem_info();

    return 0;
}