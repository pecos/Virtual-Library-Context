#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <dlfcn.h>
#include <unistd.h>

#include "VLC/runtime.h"
#include "VLC/loader.h"

void register_functions() {
    std::unordered_map<std::string, std::string> names{
        {"TORCH_VERSION_MAJOR", "TORCH_VERSION_MAJOR"}};
    VLC::Loader::register_func_names(names);
}

int main() {
    // initialize VLC environment
    VLC::Runtime vlc;
    vlc.initialize();

    // register functions used in VLC
    register_functions();

    VLC::Loader loader("libtorch.so", 1, false);

    return 0;
}