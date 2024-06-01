#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <dlfcn.h>
#include <unistd.h>

#include "VLC/runtime.h"
#include "VLC/loader.h"

typedef int (*eign_t)(int N, int N_ev);

void register_functions() {
   std::unordered_map<std::string, std::string> names{
      {"eign", "_Z4eignii"}};
   VLC::register_func_names(names);
}

int main() {
    // initialize VLC environment
    VLC::Runtime vlc;
    vlc.initialize();

    // register functions used in VLC
    register_functions();

    void *handle = dlmopen(LM_ID_NEWLM, "libeign.so", RTLD_NOW);
    if (handle == NULL) {
        fprintf(stderr, "Error in `dlmopen`: %s\n", dlerror());
    }

    auto eign = VLC::load_func<eign_t>(handle, "eign");
    printf("%p\n", eign);

    return 0;
}