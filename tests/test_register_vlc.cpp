#include <unistd.h>
#include <iostream>
#include <thread>
#include <vector>

#include "VLC/runtime.h"
#include "VLC/loader.h"

void launch(int vlc_id) {
    std::cout << "VLC " << vlc_id << " is created" << std::endl;
    VLC::Context vlc(vlc_id, gettid());
    if (vlc_id == 1)
        vlc.avaliable_cpu("0-11");
    else
        vlc.avaliable_cpu("12-23");
    VLC::register_vlc(&vlc);
}

int main() {
    // initialize VLC environment
    VLC::Runtime vlc;
    vlc.initialize();

    std::cout << "Begin!" << std::endl;
    int num_work = 2;

    std::vector<std::thread> t(num_work);
  
    for (int i = 0; i < num_work; i++) {
        t[i] = std::thread(launch, i+1);
    }

    for (int i = 0; i < num_work; i++) {
        t[i].join();
    }

    return 0;
}