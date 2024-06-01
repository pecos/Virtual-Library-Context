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

#include "VLC/loader.h"

typedef void (*init_galois_t)(int);
typedef void* (*load_file_t)(const std::string&);
typedef void (*bfs_t)(void*, int, int);
typedef unsigned int (*read_distance_t)(void*, int, int);

void register_functions() {
   std::unordered_map<std::string, std::string> names{
      {"init_galois", "_Z11init_galoisi"},
      {"load_file", "_Z9load_fileRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE"},
      {"bfs", "_Z3bfsPN6galois6graphs12LC_CSR_GraphI8NodeDatavLb1ELb0ELb0EvEEii"},
      {"read_distance", "_Z13read_distancePN6galois6graphs12LC_CSR_GraphI8NodeDatavLb1ELb0ELb0EvEEii"}};
   VLC::register_func_names(names);
}

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

void launch(int rounds, std::string filename) {

   void *handle = dlopen("libbfsgalois.so", RTLD_NOW);
   if (handle == NULL) {
      fprintf(stderr, "Error in `dlopen`: %s\n", dlerror());
      return;
   }

   // load functions from libraries
   auto init_galois = VLC::load_func<init_galois_t>(handle, "init_galois");
   auto load_file = VLC::load_func<load_file_t>(handle, "load_file");
   auto bfs = VLC::load_func<bfs_t>(handle, "bfs");
   auto read_distance = VLC::load_func<read_distance_t>(handle, "read_distance");

   printf("init_galois() starts\n");
   init_galois(10000);

   int done = 0;
   for (int rd = 0; rd < rounds; rd++) {
      printf("load_file() starts\n");
      void * graph_ptr = load_file(filename);
      
      int source = 0;
      int report = 5;

      printf("bfs() starts phase 0\n");
      bfs(graph_ptr, source, 0);
      
      printf("bfs() starts phase 1\n");
      bfs(graph_ptr, source, 1);
      unsigned int d = read_distance(graph_ptr, report, 1);
      printf("round %d: distance from %d to %d at slot %d is %d\n", rd + 1, source, report, 1, d);
      free(graph_ptr);

      done++;
   }
}

int main() {
   // register functions used in VLC
   register_functions();

   std::cout << "Begin!" << std::endl;
   auto t1 = std::chrono::high_resolution_clock::now();
   launch(10, "/var/local/yyan/graphs/road-USA.gr");

   auto t2 = std::chrono::high_resolution_clock::now();
   std::cout << "App finish in " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;

   // print_mem_info();

   return 0;
}