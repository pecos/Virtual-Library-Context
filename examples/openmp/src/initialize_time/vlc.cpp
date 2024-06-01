#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <dlfcn.h>
#include <unistd.h>
#include <fstream>
#include <numeric>

#include "VLC/runtime.h"
#include "VLC/loader.h"

typedef int (*add_t)(int *first, int *second, int *result, int num_items);

std::chrono::_V2::system_clock::time_point t0;
std::chrono::_V2::system_clock::time_point t1;

void register_functions() {
    std::unordered_map<std::string, std::string> names{
        {"add", "_Z3addPiS_S_i"}};
    VLC::register_func_names(names);
}

int main() {
    // initialize VLC environment
    VLC::Runtime vlc;
    vlc.initialize();

    // register functions used in VLC
    register_functions();

    void *handle = dlmopen(LM_ID_NEWLM, "libaddmp.so", RTLD_NOW);
    if (handle == NULL) {
        fprintf(stderr, "Error in `dlmopen`: %s\n", dlerror());
    }

    auto add = VLC::load_func<add_t>(handle, "add");
    printf("%p\n", add);

    return 0;
}