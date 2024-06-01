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
#include "addmp.h"
#include "summp.h"
#include "powermp.h"
#include <unistd.h>
#include <fstream>
#include <sys/mman.h>

pthread_barrier_t barrier;

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

void launch0(std::vector<int> first, int tag) {
   std::cout << "thread 0 begin!" << std::endl;

   printf("%d: sum() starts\n", tag);
   sum(first);
   
   printf("%d: quit\n", tag);
   pthread_barrier_wait(&barrier);
}

void launch1(int tag) {
   std::cout << "thread 1 begin!" << std::endl;

   printf("%d: add() starts\n", tag);

   int size = 12000000;
   int * v1 = (int *) mmap(NULL, size * sizeof(int), PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
   int * v2 = (int *) mmap(NULL, size * sizeof(int), PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
   int * result = (int *) mmap(NULL, size * sizeof(int), PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);

   add(v1, v2, result, size);

   printf("%d: quit\n", tag);
   pthread_barrier_wait(&barrier);
}

void launch2(int times, int tag) {
   std::cout << "thread 0 begin!" << std::endl;

   printf("%d: power() starts\n", tag);
   power(times);
   
   printf("%d: quit\n", tag);
   pthread_barrier_wait(&barrier);
}

int main() {
   std::cout << "Begin!" << std::endl;

   int num_work = 4;

   pthread_barrier_init(&barrier, NULL, num_work);
   std::cout << "pthread_barrier_init!" << std::endl;

   std::vector<std::thread> t(num_work);
   std::cout << "declare thread!" << std::endl;

   for (int i = 0; i < num_work; i++) {
      t[i] = std::thread(launch1, i);
      std::cout << "launched thread 1!" << std::endl;
   }

   for (int i = 0; i < num_work; i++) {
      t[i].join();
   }

   // print_mem_info();

   return 0;
}