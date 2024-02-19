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
#include <unistd.h>
#include <fstream>

#include "VLC.h"

typedef void (*init_galois_t)(int);
typedef void* (*load_file_t)(const std::string&);
typedef void (*bfs_t)(void*, int, int);
typedef unsigned int (*read_distance_t)(void*, int, int);

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

void launch(int tag, std::string filename, int slot) {
   std::cout << "thread " << tag << " begin!" << std::endl;

   void *handle = dlmopen(LM_ID_NEWLM, "libbfsgalois.so", RTLD_NOW);
   if (handle == NULL) {
      fprintf(stderr, "Error in `dlmopen`: %s\n", dlerror());
      return;
   }

   printf("%d: try dlsym\n", tag);
   init_galois_t init_galois = (init_galois_t) dlsym(handle, "_Z11init_galoisi");
   if (init_galois == NULL) {
      fprintf(stderr, "Error in `dlsym`: %s\n", dlerror());
      printf("quit\n");
      return;
   }

   printf("%d: load init_galois() from ./libbfsgalois.so\n", tag);
   printf("%d: init_galois() starts\n", tag);
   init_galois(10000);

   load_file_t load_file = (load_file_t) dlsym(handle, "_Z9load_fileRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
   if (load_file == NULL) {
      fprintf(stderr, "Error in `dlsym`: %s\n", dlerror());
      printf("quit\n");
      return;
   }

   printf("%d: load load_file() from ./libbfsgalois.so\n", tag);
   printf("%d: load_file() starts\n", tag);
   void* graph_ptr = load_file(filename);

   bfs_t bfs = (bfs_t) dlsym(handle, "_Z3bfsPN6galois6graphs12LC_CSR_GraphI8NodeDatavLb1ELb0ELb0EvEEii");
   if (load_file == NULL) {
      fprintf(stderr, "Error in `dlsym`: %s\n", dlerror());
      printf("quit\n");
      return;
   }

   printf("%d: load bfs() from ./libbfsgalois.so\n", tag);
   printf("%d: bfs() starts\n", tag);
   int source = 0;
   bfs(graph_ptr, source, slot);

   read_distance_t read_distance = (read_distance_t) dlsym(handle, "_Z13read_distancePN6galois6graphs12LC_CSR_GraphI8NodeDatavLb1ELb0ELb0EvEEii");
   if (load_file == NULL) {
      fprintf(stderr, "Error in `dlsym`: %s\n", dlerror());
      printf("quit\n");
      return;
   }

   printf("%d: load read_distance() from ./libbfsgalois.so\n", tag);
   printf("%d: read_distance() starts\n", tag);

   int report = 5;
   unsigned int d = read_distance(graph_ptr, report, slot);
   printf("VLC %d round 1: distance from %d to %d at slot %d is %d\n", tag, source, report, slot, d);
   
   printf("%d: quit\n", tag);

   // BUG: openmp destruct will let main process quit too
   pthread_barrier_wait(&barrier);
   // dlclose(handle);
}

int main() {
   // initialize VLC environment
   VLC::Runtime vlc;
   vlc.initialize();

   std::cout << "Begin!" << std::endl;
   int num_work = 4;

   pthread_barrier_init(&barrier, NULL, num_work);
   std::cout << "pthread_barrier_init!" << std::endl;

   std::vector<std::thread> t(num_work);
   std::cout << "declare thread!" << std::endl;

   for (int i = 0; i < num_work; i++) {
      t[i] = std::thread(launch, i, "/home/yyan/Parla.py/examples/galois_multiload_example/inputs/r.gr", i);
      std::cout << "launched thread" << i << std::endl;
   }

   for (int i = 0; i < num_work; i++) {
      t[i].join();
   }

   // print_mem_info();

   return 0;
}