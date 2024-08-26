#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <pthread.h>
#include <dlfcn.h>
#include <unistd.h>
#include <fstream>


typedef int (*test_t)();

static void print_mem_info() {
    int tSize = 0, resident = 0, share = 0;
    std::ifstream buffer("/proc/self/statm");
    buffer >> tSize >> resident >> share;
    buffer.close();

    long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
    double rss = resident * page_size_kb;
    std::cout << "RSS - " << rss << " kB\n";

    double shared_mem = share * page_size_kb;
    std::cout << "Shared Memory - " << shared_mem << " kB\n";

    std::cout << "Private Memory - " << rss - shared_mem << "kB\n";
}

void launch(int tag) {
    std::cout << "thread " << tag << " begin!" << std::endl;

    void *handle = dlmopen(LM_ID_NEWLM, "/home/yyan/vlc/examples/galois/libtestgalois.so", RTLD_NOW);
    if (handle == NULL) {
        fprintf(stderr, "Error in `dlmopen`: %s\n", dlerror());
        return;
    }

    printf("%d: try dlsym\n", tag);
    test_t test = (test_t) dlsym(handle, "_Z4testv");
    if (test == NULL) {
        fprintf(stderr, "Error in `dlsym`: %s\n", dlerror());
        printf("quit\n");
        return;
    }

    printf("%d: load test() from ./libtestgalois.so\n", tag);

    printf("%d: test() starts\n", tag);
    test();
    
    printf("%d: quit\n", tag);
}

int main() {

    std::cout << "Begin!" << std::endl;

    launch(0);

    std::cout << "End!" << std::endl;

    return 0;
}