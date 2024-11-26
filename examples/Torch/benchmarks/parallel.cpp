#include <thread>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <dlfcn.h>
#include <unistd.h>
#include <vector>

#include "VLC/tuning.h"

pthread_barrier_t barrier;

typedef int (*run_t)(int argc, char * argv[], pthread_barrier_t * barrier);

auto start_time = std::chrono::system_clock::now();

void launch(VLC::TuningConfig * config) {
    void *handle = dlopen(config->path, RTLD_LAZY | RTLD_LOCAL);
    if (!handle) {
        fprintf(stderr, "%s\n", dlerror());
        exit(EXIT_FAILURE);
    }
    
    auto run = (run_t) dlsym(handle, config->entry_point_symbol);
    if (run == NULL) {
        fprintf(stderr, "%s\n", dlerror());
        exit(EXIT_FAILURE);
    }
    
    run(config->argc, config->argv, &barrier);
    auto t1 = std::chrono::system_clock::now();

    std::cout << config->name << " end time: " << ((std::chrono::duration<double>) (t1 - start_time)).count() << "s\n";
}

int main(int argc, char * argv[]) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " config_file" << std::endl;
        return -1;
    }

    std::vector<VLC::TuningConfig> config = VLC::parse_config(argv[1], argv[0]);
    int num_task = config.size();

    start_time = std::chrono::system_clock::now();

    pthread_barrier_init(&barrier, NULL, num_task);

    std::vector<std::thread> task(num_task);
    for (int i = 0; i < num_task; i++) {
        task[i] = std::thread(launch, &config[i]);
    }
    
    for (int i = 0; i < num_task; i++) {
        task[i].join();
    }
    
    return 0;
}