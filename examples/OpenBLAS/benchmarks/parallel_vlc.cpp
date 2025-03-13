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

/**
 * Launches a Virtual Library Context (VLC) with the specified configuration.
 *
 * This function initializes a VLC environment, assigns available CPU cores,
 * and loads the specified library for execution within the VLC.
 *
 * @param config Pointer to a VLC::TuningConfig object containing the configuration
 *               details such as the library path, entry point symbol, and arguments.
 * @param vlc_id An integer representing the VLC identifier.
 * @param cpu_str A string specifying the CPU cores to be allocated for this VLC.
 *
 * The function outputs the creation and end time of the VLC execution.
 */
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
        std::cerr << "usage: " << argv[0] << " config_file cpuset1 (cpuset2 cpuset3 ...)" << std::endl;
        return -1;
    }

    std::vector<VLC::TuningConfig> config = VLC::parse_config(argv[1], argv[0]);
    int num_task = config.size();

    if (num_task + 2 != argc) {
        std::cerr << "usage: " << argv[0] << " config_file cpuset1 (cpuset2 cpuset3 ...)" << std::endl;
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