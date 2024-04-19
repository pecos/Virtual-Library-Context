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
    matmul(N);
        
    for (int i = 0; i < 36; i++) {  // 8 smaller matmul
        matmul(N / 8);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "matmul finish in " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;

    return 0;
}