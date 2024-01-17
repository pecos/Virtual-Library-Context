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

pthread_barrier_t barrier;

void launch0(std::vector<int> first, int tag) {
   std::cout << "thread 0 begin!" << std::endl;

   printf("%d: sum() starts\n", tag);
   sum(first);
   
   pthread_barrier_wait(&barrier);
   printf("%d: quit\n", tag);
}

void launch1(std::vector<int> first, std::vector<int> second, int tag) {
   std::cout << "thread 1 begin!" << std::endl;

   printf("%d: add() starts\n", tag);
   add(first, second);

   pthread_barrier_wait(&barrier);
   printf("%d: quit\n", tag);
}

int main() {
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