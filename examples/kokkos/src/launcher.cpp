#include <thread>
#include <iostream>
#include <vector>
#include <numeric>

#include "VLC/runtime.h"
#include "VLC/loader.h"

std::chrono::_V2::system_clock::time_point vlc_init_start;
std::chrono::_V2::system_clock::time_point vlc_init_end;

typedef void (*kokkos_init_t)(int dev);
typedef void (*kokkos_finalize_t)();
typedef double (*kokkos_reduction_copy_to_device_t)(double* array, const int N, const int dev_id);

void register_functions() {
    std::unordered_map<std::string, std::string> names{
        {"kokkos_init", "_Z11kokkos_initi"},
        {"kokkos_finalize", "_Z15kokkos_finalizev"},
        {"kokkos_reduction_copy_to_device", "_Z31kokkos_reduction_copy_to_devicePdii"}};
    VLC::Loader::register_func_names(names);
}

void launch(int vlc_id) {
    VLC::Context vlc(vlc_id, gettid());
    vlc.avaliable_cpu("0-23"); // please change the number based on your system
    VLC::register_vlc(&vlc);
    VLC::Loader loader("libkokkos_compute.so", vlc_id, false);

    auto kokkos_init = loader.load_func<kokkos_init_t>("kokkos_init");
    auto kokkos_finalize = loader.load_func<kokkos_finalize_t>("kokkos_finalize");
    auto kokkos_reduction_copy_to_device = loader.load_func<kokkos_reduction_copy_to_device_t>("kokkos_reduction_copy_to_device");
    vlc_init_end = std::chrono::high_resolution_clock::now();
    std::cout << "PERF: VLC init finished in " << std::chrono::duration_cast<std::chrono::milliseconds>(vlc_init_end - vlc_init_start).count() << "ms" << std::endl;

    int dev_id = 0;

    kokkos_init(dev_id);

    int n = 100000;

    std::vector<double> v(n);
    std::iota(v.begin(), v.end(), 1.0);

    double result = kokkos_reduction_copy_to_device(v.data(), n, dev_id);
    std::cout << "kokkos: result=" << result << std::endl;

    kokkos_finalize();
}

int main() {
    vlc_init_start = std::chrono::high_resolution_clock::now();
    VLC::Runtime vlc;     // initialize VLC environment
    vlc.initialize();
    register_functions();     // register functions used in VLC
    std::thread t;
    t = std::thread(launch, 1);
    t.join();
}
