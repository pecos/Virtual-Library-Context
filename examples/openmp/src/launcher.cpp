#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <thread>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <pthread.h>
#include <dlfcn.h>
#include <omp.h>

#include "VLC.h"

typedef int (*sum_t)(std::vector<int> &);
typedef std::vector<int> (*add_t)(std::vector<int> &, std::vector<int> &);
typedef void (*register_thread_t)(pthread_t, int);
typedef void (*enable_t)(pthread_t);

pthread_barrier_t barrier;

void launch0(std::vector<int> first, int tag) {
   std::cout << "thread 0 begin!" << std::endl;
   // std::this_thread::sleep_for(std::chrono::seconds(5));
   pthread_t tid = pthread_self();
   printf("%d: thread %08lx starts\n", tag, tid);

   void *handle = dlmopen(LM_ID_NEWLM, "libsummp.so", RTLD_NOW);
   if (handle == NULL) {
      fprintf(stderr, "Error in `dlmopen`: %s\n", dlerror());
      return;
   }

   printf("%d: try dlsym\n", tag);
   sum_t sum = (sum_t) dlsym(handle, "_Z3sumRSt6vectorIiSaIiEE");
   if (sum == NULL) {
      fprintf(stderr, "Error in `dlsym`: %s\n", dlerror());
      printf("quit\n");
      return;
   }

   printf("%d: load sum() from ./libsummp.so\n", tag);

   printf("%d: sum() starts\n", tag);
   sum(first);
   
   pthread_barrier_wait(&barrier);
   printf("%d: quit\n", tag);
}

void launch1(std::vector<int> first, std::vector<int> second, int tag) {
   std::cout << "thread 1 begin!" << std::endl;
   pthread_t tid = pthread_self();
   printf("%d: thread %08lx starts\n", tag, tid);

   printf("%d: try dlmopn\n", tag);
   
   void *handle = dlmopen(LM_ID_NEWLM, "libaddmp.so", RTLD_NOW);
   if (handle == NULL) {
      fprintf(stderr, "Error in `dlmopen`: %s\n", dlerror());
      return;
   }
   

   printf("%d: try dlsym\n", tag);
   add_t add = (add_t) dlsym(handle, "_Z3addRSt6vectorIiSaIiEES2_");

   if (add == NULL) {
      fprintf(stderr, "Error in `dlsym`: %s\n", dlerror());
      printf("quit\n");
      return;
   }

   printf("%d: load add() from ./libaddmp.so\n", tag);

   printf("%d: add() starts\n", tag);
   add(first, second);

   pthread_barrier_wait(&barrier);
   printf("%d: quit\n", tag);
}

int main() {
   // initialize VLC environment
   VLC::Runtime vlc;
   vlc.initialize();

   std::cout << "Begin!" << std::endl;
   int size = 12000000;
   std::vector<int> v0(size, 1);
   std::vector<int> v1(size, 1);
   std::vector<int> v2(size, 1);

   pthread_barrier_init(&barrier, NULL, 2);
   std::cout << "pthread_barrier_init!" << std::endl;

   std::vector<std::thread> t(2);
   std::cout << "declare thread!" << std::endl;

   t[0] = std::thread(launch0, v0, 0);
   std::cout << "launched thread 0!" << std::endl;
   t[1] = std::thread(launch1, v1, v2, 1);
   std::cout << "launched thread 1!" << std::endl;
   
   t[0].join();
   t[1].join();

   return 0;
}