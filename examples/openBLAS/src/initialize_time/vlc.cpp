#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <dlfcn.h>
#include <unistd.h>

#include "VLC/runtime.h"
#include "VLC/loader.h"

typedef void (*matmul_t)(int N);

void register_functions() {
    std::unordered_map<std::string, std::string> names{
        {"matmul", "_Z6matmuli"}};
    VLC::register_func_names(names);
}

int main() {
    // initialize VLC environment
    VLC::Runtime vlc;
    vlc.initialize();

    // register functions used in VLC
    register_functions();

    void *handle = dlmopen(LM_ID_NEWLM, "libmatmul.so", RTLD_NOW);
    if (handle == NULL) {
        fprintf(stderr, "Error in `dlmopen`: %s\n", dlerror());
    }

    auto matmul = VLC::load_func<matmul_t>(handle, "matmul");
    printf("%p\n", matmul);

    return 0;
}