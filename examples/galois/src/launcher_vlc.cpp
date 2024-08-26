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

typedef int (*test_t)();

pthread_barrier_t barrier;

void register_functions() {
    std::unordered_map<std::string, std::string> names{
        {"test", "_Z4testv"}};
    VLC::Loader::register_func_names(names);
}

void launch(int vlc_id) {
    std::cout << "VLC " << vlc_id << " begin!" << std::endl;
    VLC::Context vlc(vlc_id, gettid());

    // please change the number based on your system
    if (vlc_id == 1)
        vlc.avaliable_cpu("0-11");
    else
        vlc.avaliable_cpu("12-23");
    VLC::register_vlc(&vlc);

    VLC::Loader loader("libbfsgalois.so", vlc_id, false);

    printf("%d: try dlsym\n", vlc_id);
    auto test = loader.load_func<test_t>("_Z4testv");

    printf("%d: test() starts\n", vlc_id);
    test();
    
    printf("%d: quit\n", vlc_id);
    
    // BUG: openmp destruct will let main process quit too
    pthread_barrier_wait(&barrier);
    // dlclose(handle);
}

int main() {
    // initialize VLC environment
    VLC::Runtime vlc;
    vlc.initialize();

    std::cout << "Begin!" << std::endl;
    int num_work = 2;

    pthread_barrier_init(&barrier, NULL, num_work);
    std::cout << "pthread_barrier_init!" << std::endl;

    std::vector<std::thread> t(num_work);
    std::cout << "declare thread!" << std::endl;

    for (int i = 0; i < num_work; i++) {
        t[i] = std::thread(launch, i+1);
        std::cout << "launched thread" << i << std::endl;
    }

    for (int i = 0; i < num_work; i++) {
        t[i].join();
    }

    return 0;
}