#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <iostream>
#include <vector>

#include "VLC.h"

typedef int (*pthread_create_t)(pthread_t *, const pthread_attr_t *,
                          void *(*)(void *), void *);

void *threadfunction2(void *arg) {
    printf("pthread created!\n");
    return 0;
}

void *threadfunction(void *arg)
{
    void *handle = dlmopen(LM_ID_NEWLM, "libpthread.so.0", RTLD_LAZY);

    pthread_create_t pthread_create_ptr = (pthread_create_t) dlsym(handle, "pthread_create");

    pthread_t thread3;
    int createerror3 = pthread_create_ptr(&thread3, NULL, threadfunction2, NULL);

    pthread_join(createerror3, NULL);

    printf("Hello, World!\n"); /*printf() is specified as thread-safe as of C11*/
    fflush(stdout);
    return 0;
}

int main(void)
{
    VLC::Runtime vlc;
    vlc.initialize();

    std::cout << "Begin!" << std::endl;
    std::vector<pthread_t> threads(16);
    int createerror = 0;
    for (int i = 0; i < 16; i++) {
        createerror += pthread_create(&threads[i], NULL, threadfunction2, NULL);
    }
    /*creates a new thread with default attributes and NULL passed as the argument to the start routine*/
    if (!createerror) { /*check whether the thread creation was successful*/
        for (int i = 0; i < 16; i++) {
            pthread_join(threads[i], NULL);     /*wait until the created thread terminates*/
        }
        return 0;
    }

    // TODO: provide __errno_location symbol defined in shim
    //   std::cerr << "App: unable to create pthread, " << strerror(errno) << std::endl;
    return 1;
}
