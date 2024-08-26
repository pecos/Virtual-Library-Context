/**
 * This App are try to count the overhead of VLC transparency machanism
*/

#include <thread>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>

#include "VLC/runtime.h"
#include "VLC/loader.h"

#include "hello.h"


int main() {
    // initialize VLC environment
    VLC::Runtime vlc;
    vlc.initialize();

    VLC::load_vlc("helloworld_stub_overhead/build/libhello.so", 1);

    
    std::chrono::_V2::system_clock::time_point hello_begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 120000; i++) { 
        hello();
    }
    std::chrono::_V2::system_clock::time_point hello_end = std::chrono::high_resolution_clock::now();
    std::cout << "average hello finish in " << std::chrono::duration_cast<std::chrono::microseconds>(hello_end - hello_begin).count() / 120000.0 << "us" << std::endl;

    return 0;
}