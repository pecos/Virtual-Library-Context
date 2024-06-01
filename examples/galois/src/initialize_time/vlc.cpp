#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <dlfcn.h>
#include <unistd.h>

#include "VLC/runtime.h"
#include "VLC/loader.h"

typedef void (*bfs_t)(void*, int, int);

void register_functions() {
    std::unordered_map<std::string, std::string> names{
            {"bfs", "_Z3bfsPN6galois6graphs12LC_CSR_GraphI8NodeDatavLb1ELb0ELb0EvEEii"}};
    VLC::register_func_names(names);
}

int main() {
    // initialize VLC environment
    VLC::Runtime vlc;
    vlc.initialize();

    // register functions used in VLC
    register_functions();

    void *handle = dlmopen(LM_ID_NEWLM, "libbfsgalois.so", RTLD_NOW);
    if (handle == NULL) {
        fprintf(stderr, "Error in `dlmopen`: %s\n", dlerror());
    }

    // load functions from libraries
    auto bfs = VLC::load_func<bfs_t>(handle, "bfs");
    printf("%p\n", bfs);

    return 0;
}