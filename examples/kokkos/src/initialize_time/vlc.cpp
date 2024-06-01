#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <dlfcn.h>
#include <unistd.h>

#include "VLC/runtime.h"
#include "VLC/loader.h"

typedef void (*kokkos_init_t)(int dev);

void register_functions() {
    std::unordered_map<std::string, std::string> names{
        {"kokkos_init", "_Z11kokkos_initi"}};
    VLC::register_func_names(names);
}

int main() {
    // initialize VLC environment
    VLC::Runtime vlc;
    vlc.initialize();

    // register functions used in VLC
    register_functions();

    void *handle = dlmopen(LM_ID_NEWLM, "libkokkos_compute.so", RTLD_NOW);
    if (handle == NULL) {
        fprintf(stderr, "Error in `dlmopen`: %s\n", dlerror());
    }

    // load functions from libraries
    auto kokkos_init = VLC::load_func<kokkos_init_t>(handle, "kokkos_init");
    printf("%p\n", kokkos_init);

    return 0;
}