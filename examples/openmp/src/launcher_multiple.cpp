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
#include <unistd.h>
#include <fstream>

#include "VLC.h"

typedef int (*add_t)(std::vector<int> *, std::vector<int> *, std::vector<int> *); 
typedef int (*sum_t)(std::vector<int> &);
typedef int (*power_t)(int);

pthread_barrier_t barrier;

static void print_mem_info() {
   int tSize = 0, resident = 0, share = 0;
   std::ifstream buffer("/proc/self/statm");
   buffer >> tSize >> resident >> share;
   buffer.close();

   long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
   double rss = resident * page_size_kb;
   std::cout << "RSS - " << rss << " kB" << std::endl;

   double shared_mem = share * page_size_kb;
   std::cout << "Shared Memory - " << shared_mem << " kB" << std::endl;

   std::cout << "Private Memory - " << rss - shared_mem << "kB" << std::endl;
}

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
   
   printf("%d: quit\n", tag);
   
   // BUG: openmp destruct will let main process quit too
   pthread_barrier_wait(&barrier);
   // dlclose(handle);
}

void launch1(std::vector<int> *first, std::vector<int> *second, int tag) {
   std::cout << "thread 1 begin!" << std::endl;
   // std::this_thread::sleep_for(std::chrono::seconds(5));
   pthread_t tid = pthread_self();
   printf("%d: thread %08lx starts\n", tag, tid);

   void *handle = dlmopen(LM_ID_NEWLM, "libaddmp.so", RTLD_NOW);
   if (handle == NULL) {
      fprintf(stderr, "Error in `dlmopen`: %s\n", dlerror());
      return;
   }
   
   printf("%d: try dlsym\n", tag);
   add_t add = (add_t) dlsym(handle, "_Z3addPSt6vectorIiSaIiEES2_S2_");

   if (add == NULL) {
      fprintf(stderr, "Error in `dlsym`: %s\n", dlerror());
      printf("quit\n");
      return;
   }

   printf("%d: load add() from ./libaddmp.so\n", tag);

   printf("%d: add() starts\n", tag);
   std::vector<int> result(first->size());
   add(first, second, &result);
   
   printf("%d: quit\n", tag);
   
   // BUG: openmp destruct will let main process quit too
   pthread_barrier_wait(&barrier);
   // dlclose(handle);
}

void launch2(int times, int tag) {
   std::cout << "thread 0 begin!" << std::endl;
   // std::this_thread::sleep_for(std::chrono::seconds(5));
   pthread_t tid = pthread_self();
   printf("%d: thread %08lx starts\n", tag, tid);

   void *handle = dlmopen(LM_ID_NEWLM, "libpowermp.so", RTLD_NOW);
   if (handle == NULL) {
      fprintf(stderr, "Error in `dlmopen`: %s\n", dlerror());
      return;
   }

   printf("%d: try dlsym\n", tag);
   power_t power = (power_t) dlsym(handle, "_Z5poweri");
   if (power == NULL) {
      fprintf(stderr, "Error in `dlsym`: %s\n", dlerror());
      printf("quit\n");
      return;
   }

   printf("%d: load power() from ./libpowermp.so\n", tag);

   printf("%d: power() starts\n", tag);
   power(times);
   
   printf("%d: quit\n", tag);
   
   // BUG: openmp destruct will let main process quit too
   pthread_barrier_wait(&barrier);
   // dlclose(handle);
}

int main() {
   printf("My 1 pid:%d\n", getpid());
   // initialize VLC environment
   VLC::Runtime vlc;
   vlc.initialize(); 

   printf("My 2 pid:%d\n", getpid());

   std::cout << "Begin!" << std::endl;
   int size = 12000000;
   std::vector<int> v1(size, 1);
   std::vector<int> v2(size, 1);

   int num_lib = 4;

   pthread_barrier_init(&barrier, NULL, num_lib);
   std::cout << "pthread_barrier_init!" << std::endl;

   std::vector<std::thread> t(num_lib);
   std::cout << "declare thread!" << std::endl;

   for (int i = 0; i < num_lib; i++) {
      t[i] = std::thread(launch1, &v1, &v2, i);
      std::cout << "launched work " << i << std::endl;
   }

   for (int i = 0; i < num_lib; i++) {
      t[i].join();
   }

   print_mem_info();

   return 0;
}