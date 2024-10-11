#include <thread>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <pthread.h>
#include "addmp.h"

int main() {
    // ensure it is indeed loaded
    printf("%p\n", add);
    return 0;
}