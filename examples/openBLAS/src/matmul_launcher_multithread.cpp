#include <thread>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <pthread.h>
#include "matmul.h"

int main() {
    int N = 8192;
    auto t1 = std::chrono::high_resolution_clock::now();
    int num_work = 20;

    std::vector<std::thread> t(num_work+1);

    
    for (int i = 0; i < num_work; i++) {
        t[i] = std::thread(matmul, N / 8);
    }

    t[num_work] = std::thread(matmul, N);
 
    for (int i = 0; i < num_work + 1; i++) {
        t[i].join();
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "matmul finish in " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;


    return 0;
}