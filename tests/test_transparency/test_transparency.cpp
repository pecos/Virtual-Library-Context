#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <fcntl.h>
#include <sched.h>
#include <iostream>
#include <filesystem>
#include <thread>
#include <vector>

#include "VLC/runtime.h"
#include "VLC/loader.h"
#include "foobar.h"

void launch(int id) {
    std::cout << "VLC " << id << " is created" << std::endl;
    VLC::load_vlc("build/libfoobar.so", id);
    // transparency
    foo();
    bar();
    return;
}

int main() {
    // initialize VLC environment
    VLC::Runtime vlc;
    vlc.initialize();

    std::cout << "Begin!" << std::endl;
    int num_work = 2;

    std::vector<std::thread> t(num_work);
  
    for (int i = 0; i < num_work; i++) {
        t[i] = std::thread(launch, i+1);
    }

    for (int i = 0; i < num_work; i++) {
        t[i].join();
    }

    return 0;
}