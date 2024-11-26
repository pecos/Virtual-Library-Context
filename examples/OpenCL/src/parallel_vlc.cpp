#include <thread>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <dlfcn.h>
#include <unistd.h>

#include "VLC/runtime.h"
#include "VLC/loader.h"

// pthread_barrier_t barrier;

typedef int (*hello_t)();

auto start_time = std::chrono::system_clock::now();

void register_functions() {
   std::unordered_map<std::string, std::string> names{
      {"hello", "_Z5hellov"}};
   VLC::Loader::register_func_names(names);
}

void launch_hello(int vlc_id, const char* cpu_str) {
    void *handle = dlmopen(LM_ID_NEWLM, "hello.so", RTLD_LAZY | RTLD_LOCAL);
    if (!handle) {
        fprintf(stderr, "%s\n", dlerror());
        exit(EXIT_FAILURE);
    }

    auto hello = (hello_t) dlsym(handle, "_Z5hellov");
    if (hello == NULL) {
        fprintf(stderr, "%s\n", dlerror());
        exit(EXIT_FAILURE);
    }

    hello();

    auto t1 = std::chrono::system_clock::now();
    std::cout << "IS end time: " << ((std::chrono::duration<double>) (t1 - start_time)).count() << "s\n";
}

int main(int argc, char * argv[]) {
    // // initialize VLC environment
    // VLC::Runtime vlc;
    // vlc.initialize();

    // register_functions();

    std::cout << "Begin!" << std::endl;

    // pthread_barrier_init(&barrier, NULL, 1);

    auto hello = std::thread(launch_hello, 1, "0-11");
    // auto hello2 = std::thread(launch_hello, 2, "12-23");
    
    hello.join();
    // hello2.join();

    return 0;
}