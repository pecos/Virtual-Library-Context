#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <fcntl.h>
#include <sched.h>
#include <iostream>
#include <filesystem>

#include "VLC/runtime.h"

int main() {
    // initialize VLC environment
    VLC::Runtime vlc;
    vlc.initialize();

    // Test 1
    std::cout << "Test 1: read file directly" << std::endl;
    std::ifstream mem_file("/proc/meminfo");
    if (!mem_file.is_open()) {
        VLC_DIE("VLC: unable to open /proc/meminfo");
    }

    std::string line;
    while (std::getline(mem_file, line)) {
        std::cout << "1:" << line << std::endl;
    }

    return 0;
}