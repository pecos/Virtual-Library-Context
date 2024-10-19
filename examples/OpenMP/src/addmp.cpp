#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include "addmp.h"
#include <unistd.h>
#include <fstream>
#include <thread>
#include <stdio.h>
#include <string.h>

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

const int REPEAT = 400;
const int NUM_CORE = 64;

#define COUNT_CORE  // if define, will print thread mapping bitmap
// #define BIND_CORE   // if define, will bind thread to core

int add(int *first, int *second, int *result, int num_items) {
    int th_id;

    #ifdef COUNT_CORE
    std::vector<int> core_count(NUM_CORE, 0);
    std::vector<int> core_count2(NUM_CORE, 0);  // assume 48 core
    #endif

    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel private(th_id)
    {   
        // printf("addmp: cpu_id: %d\n", sched_getcpu());
        th_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int batch_size = num_items / num_threads;

        #ifdef BIND_CORE
        // bind thread to core manually
        cpu_set_t my_set;
        CPU_ZERO(&my_set);
        // CPU_SET(th_id, &my_set);
        CPU_SET(th_id + 12, &my_set);
        sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
        #endif

        #ifdef COUNT_CORE
        int last_core_id = sched_getcpu();
        core_count[last_core_id] += 1;
        #endif

        for (int j = 0; j < REPEAT; j++) {
            for (int i = 0; i < batch_size; i++) {
                result[th_id * batch_size + i] = first[th_id * batch_size + i] + second[th_id * batch_size + i];
            }
        }

        #ifdef COUNT_CORE
        core_count2[sched_getcpu()] += 1;
        #endif
    }
    
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "addmp: finish in " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;

    #ifdef COUNT_CORE
    std::cout << "addmp: core map1:" << std::endl;
    for (int i : core_count) {
        std::cout << i << ", ";
    }
    std::cout << std::endl;

    std::cout << "addmp: core map2:" << std::endl;
    for (int i : core_count2) {
        std::cout << i << ", ";
    }
    std::cout << std::endl;
    #endif

    print_mem_info();
    return 0;
}

// std::vector<int> *global_first;
// std::vector<int> *global_second;
// std::vector<int> *global_result;

// int task(int th_id, int batch_size) {
//     for (int j = 0; j < REPEAT; j++) {
//         for (int i = 0; i < batch_size; i++) {
//            (*global_result)[th_id * batch_size + i] = (*global_first)[th_id * batch_size + i] + (*global_second)[th_id * batch_size + i];
//         }
//     }
//     return 0;
// }

// int add(std::vector<int> *first, std::vector<int> *second, std::vector<int> *result) {
//     std::cout << "begin" << std::endl;

//     int num_items = first->size();

//     int num_threads = 48;

//     std::vector<std::thread> t(num_threads);

//     global_first = first;
//     global_second = second;
//     global_result = result;

//     int batch_size = num_items / num_threads;

//     auto t1 = std::chrono::high_resolution_clock::now();
//     for (int i = 0; i < num_threads; i++) {
//         t[i] = std::thread(task, i, batch_size);
//     }

//     for (int i = 0; i < num_threads; i++) {
//         t[i].join();
//     }
//     auto t2 = std::chrono::high_resolution_clock::now();

//     // print_mem_info();
//     return 0;
// }