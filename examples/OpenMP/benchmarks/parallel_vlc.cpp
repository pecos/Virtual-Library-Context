#include <thread>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <dlfcn.h>
#include <unistd.h>
#include <vector>

#include "VLC/runtime.h"
#include "VLC/tuning.h"

pthread_barrier_t barrier;

typedef int (*run_t)(int argc, char * argv[], pthread_barrier_t * barrier);

auto start_time = std::chrono::system_clock::now();

void launch_vlc(VLC::TuningConfig * config, int vlc_id, const char* cpu_str) {
    std::cout << "VLC " << vlc_id << "(" << config->name << ") is created." << std::endl;
    VLC::Context vlc(vlc_id, gettid());

    // please change the number based on your system
    vlc.avaliable_cpu(cpu_str);
    VLC::register_vlc(&vlc);

    VLC::Loader loader(config->path, vlc_id, false);
    auto run = loader.load_func<run_t>(std::string(config->name));
    
    run(config->argc, config->argv, &barrier);
    auto t1 = std::chrono::system_clock::now();

    std::cout << "VLC " << vlc_id << "(" << config->name << ") end time: " << ((std::chrono::duration<double>) (t1 - start_time)).count() << "s\n";
}

int main(int argc, char * argv[]) {
    // initialize VLC environment
    VLC::Runtime vlc;
    vlc.initialize();

    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << "config_file cpuset1 (cpuset2 cpuset3 ...)" << std::endl;
        return -1;
    }

    std::vector<VLC::TuningConfig> config = VLC::parse_config(argv[1], argv[0]);
    int num_task = config.size();

    if (num_task + 2 != argc) {
        std::cerr << "usage: " << argv[0] << "config_file cpuset1 (cpuset2 cpuset3 ...)" << std::endl;
        return -1;
    }

    std::cout << "VLCs Tuning Start." << std::endl;
    start_time = std::chrono::system_clock::now();

    pthread_barrier_init(&barrier, NULL, num_task);

    std::vector<std::thread> task(num_task);
    for (int i = 0; i < num_task; i++) {
        task[i] = std::thread(launch_vlc, &config[i], i + 1, argv[i + 2]);
    }
    
    for (int i = 0; i < num_task; i++) {
        task[i].join();
    }
    
    return 0;
}