#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <fcntl.h>
#include <sched.h>
#include <iostream>
#include <filesystem>

#include "VLC.h"

int main() {
    // initialize VLC environment
    VLC::Runtime vlc;
    vlc.initialize();

    // Test 1
    std::cout << "Test 1: read file directly" << std::endl;
    std::string path = "/sys/devices/system/cpu";
    for (const auto & entry : std::filesystem::directory_iterator(path)) {
        std::cout << entry.path() << std::endl;
    }

    // Test 2: 
    std::cout << "\nTest 2: std::thread::hardware_concurrency()" << std::endl;
    std::cout << "result is " << std::thread::hardware_concurrency() << std::endl;

    // Test 3:
    std::cout << "\nTest 3: sysconf(_SC_NPROCESSORS_ONLN)" << std::endl;
    std::cout << "result is " << sysconf(_SC_NPROCESSORS_ONLN) << std::endl;

    // Test 4:
    std::cout << "\nTest 4: sysconf(_SC_NPROCESSORS_CONF)" << std::endl;
    std::cout << "result is " << sysconf(_SC_NPROCESSORS_CONF) << std::endl;

    return 0;
}