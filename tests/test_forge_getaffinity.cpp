#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <sched.h>
#include <iostream>

#include "VLC.h"

int main() {
    // initialize VLC environment
    VLC::Runtime vlc;
    vlc.initialize();

    cpu_set_t mask;

    for (int i = 0; i < 10; i++) {
        if (sched_getaffinity(0, sizeof(cpu_set_t), &mask) == -1) {
            std::cerr << "APP: unable to determine cpu set, " << strerror(errno) << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    std::cout << "APP: Visible CPU are ";

    for (int i = 0; i < sizeof(cpu_set_t) * 8; i++) {
        if (CPU_ISSET(i, &mask)) {
            std::cout << i << " ";
        }
    }
    std::cout << std::endl;

    return 0;
}