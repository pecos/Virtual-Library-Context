#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <dlfcn.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <sched.h>
#include <fcntl.h>
#include <sys/mman.h>

#include "VLC/runtime.h"
#include "VLC/loader.h"

int main() {
    cpu_set_t mask;
    int fd;
    std::chrono::_V2::system_clock::time_point initialize_begin = std::chrono::high_resolution_clock::now();

    // initialize VLC environment
    VLC::Runtime vlc;
    vlc.initialize();

    std::chrono::_V2::system_clock::time_point initialize_end = std::chrono::high_resolution_clock::now();

    void *handle = dlmopen(LM_ID_NEWLM, NULL, RTLD_NOW);  // create VLC without any content
    if (handle == NULL) {
        fprintf(stderr, "Error in `dlmopen`: %s\n", dlerror());
    }

    std::chrono::_V2::system_clock::time_point create_end = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 120000; i++) {
        sched_getaffinity(0, sizeof(cpu_set_t), &mask);
    }
    std::chrono::_V2::system_clock::time_point setaffinity_end = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < 120000; i++) {
    //     fd = open("/proc/cpuinfo", O_RDONLY);
    //     close(fd);
    // }
    std::chrono::_V2::system_clock::time_point readfile_end  = std::chrono::high_resolution_clock::now();

    void *ptr = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);

    for (int i = 0; i < 120000; i++) {
        mmap(ptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
    }

    std::chrono::_V2::system_clock::time_point mmap_end  = std::chrono::high_resolution_clock::now();

    std::cout << "initialize runtime finish in " << std::chrono::duration_cast<std::chrono::microseconds>(initialize_end - initialize_begin).count() << "us" << std::endl;
    std::cout << "create VLC finish in " << std::chrono::duration_cast<std::chrono::microseconds>(create_end - initialize_end).count() << "us" << std::endl;
    std::cout << "average sched_getaffinity finish in " << std::chrono::duration_cast<std::chrono::microseconds>(setaffinity_end - create_end).count() / 120000.0 << "us" << std::endl;
    std::cout << "average open finish in " << std::chrono::duration_cast<std::chrono::microseconds>(readfile_end - setaffinity_end).count() / 120000.0 << "us" << std::endl;
    std::cout << "average mmap finish in " << std::chrono::duration_cast<std::chrono::microseconds>(mmap_end - readfile_end).count() / 120000.0 << "us" << std::endl;

    return 0;
}