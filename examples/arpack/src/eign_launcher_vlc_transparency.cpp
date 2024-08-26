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

#include "eign.h"

std::chrono::_V2::system_clock::time_point t0;
std::chrono::_V2::system_clock::time_point t1;

void launch(int vlc_id, int N, int N_ev) {
    std::cout << "VLC " << vlc_id << " is created" << std::endl;
    VLC::Context vlc(vlc_id, gettid());

    // please change the number based on your system
    if (vlc_id == 1)
        vlc.avaliable_cpu("0-11");
    else
        vlc.avaliable_cpu("12-23");
    VLC::register_vlc(&vlc);

    VLC::Loader loader("/usr/local/lib/libarpack.so.2", vlc_id, true);

    t0 = std::chrono::high_resolution_clock::now();
    eign(N, N_ev);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "eign finish in " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t0).count() << "ms" << std::endl;
}

int main() {
    // initialize VLC environment
    VLC::Runtime vlc;
    vlc.initialize();

    int N = 10000;
    int N_ev = 10;

    std::cout << "Begin!" << std::endl;
    int num_work = 2;

    std::vector<std::thread> t(num_work);
    
    for (int i = 0; i < num_work; i++) {
        t[i] = std::thread(launch, i+1, N, N_ev);
    }

    for (int i = 0; i < num_work; i++) {
        t[i].join();
    }

    return 0;
}