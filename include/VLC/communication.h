#ifndef _VLC_COMMUNICATION_H_
#define _VLC_COMMUNICATION_H_

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <iostream>
#include <signal.h>

#include "VLC/info.h"
#include "VLC/resource.h"

namespace VLC {
namespace Internal {

static pid_t MONITOR_PID;

class SharedMem {
public:
    int id;
    size_t size;

    /**
     * Create a shared memory
     * 
     * @param size the shared memory segment size 
    */
    SharedMem(size_t size) {
        this->size = size;

        if ((id = shmget(IPC_PRIVATE, size, IPC_CREAT | IPC_EXCL | S_IRUSR | S_IWUSR)) < 0) {
            VLC_DIE("VLC: unable to create shared memory, %s", strerror(errno));
        }
    }

    /**
     * Create a shared memory by given ID and size
     * 
     * @param size the shared memory segment size
     * @param id the id of a already created shared memory
    */
    SharedMem(size_t size, int id) {
        this->size = size;
        this->id = id;
    }

    /**
     * Default Constructor
    */
    SharedMem() {
        this->size = 0;
        this->id = 0;
    }

    /* use default copy/delete/move constructors should be fine here */

    /**
     * Write data to shared memory
     * 
     * @param data where the data is write from
    */
    void write(void *data) {
        void *shm_data;

        if ((shm_data = shmat(id, NULL, 0)) == (void *) -1) {
            VLC_DIE("VLC: unable to attach to shared memory, %s", strerror(errno));
        }

        memcpy(shm_data, data, size);
        shmdt(shm_data);
    }

    /**
     * Read data from shared memory
     * 
     * @param data where the data is read to, shoule have enough size.
    */
    void read(void *data) {
        void *shm_data;

        if ((shm_data = shmat(id, NULL, 0)) == (void *) -1) {
            VLC_DIE("VLC: unable to attach to shared memory, %s", strerror(errno));
        }

        memcpy(data, shm_data, size);
        shmdt(shm_data);
    }

    /**
     * Destroy this shared memory.
    */
    void destory() {
        shmctl(id, IPC_RMID, 0);
    }
};


static SharedMem VLC_SHARED_MEM;

/**
 * Retrive the already created shared memory object.
 * 
 * This is done internally by saving the ID to a static field.
*/
SharedMem get_vlc_shared_mem() {
    return VLC_SHARED_MEM;
}

/**
 * A signal handler for monitor's use.
 * 
 * This hanlder expect recive a signal from application
 * which indicates application has updated shared memory content.
*/
void monitor_sig_hanlder(int sig) {
    signal(SIGUSR1, monitor_sig_hanlder); //reset signal hanlder

    VLC_DEBUG("VLC: recived signal");

    VLC::Context vlc_config;
    VLC_SHARED_MEM.read((void *) &vlc_config);

    // save the vlc config 
    pid_to_vlc_id[vlc_config.thread_id] = vlc_config.id;

    // parse cpu str
    std::string cpu_str(vlc_config.cpu_str);
    if (!cpu_str.empty()) {  // if string is empty, skip this step
        std::vector<std::string> substrings;
        std::vector<int> core_map;

        std::stringstream ss(cpu_str);

        while(ss.good()) {
            std::string substr;
            getline(ss, substr, ',' );
            substrings.push_back(substr);
        }

        for (auto substr: substrings) {
            std::stringstream ss(substr);
            int begin;
            ss >> begin;

            // for format: "0-4"
            if (ss.peek() == '-') {
                ss.ignore();
                int end;
                ss >> end;

                for (int i = begin; i <= end; i++) {
                    core_map.push_back(i);
                }
            } else {  // format: "0"
                core_map.push_back(begin);
            }
        }

        VLC::Internal::vlc_id_to_core_map[vlc_config.id] = std::move(core_map);
    }

    // enforce affinity
    create_virtual_affinity(vlc_config.thread_id);
    enfore_virtual_affinity(vlc_config.thread_id);

    // send SIGUSR1 signal back to application
    // so it knows the content is already processed by VLC monitor
    // kill(vlc_config.thread_id, SIGUSR1);
    vlc_config.reset();
    VLC_SHARED_MEM.write((void *) &vlc_config);
    VLC_DEBUG("VLC: notify %d that message is read", vlc_config.thread_id);
}

/**
 * A signal handler for application's use.
 * 
 * This hanlder expect recive a signal from monitor
 * which indicates monitor has read shared memory content.
*/
void application_sig_handler(int sig) {
    signal(SIGUSR1, application_sig_handler);
    VLC_DEBUG("application: recived signal");
}

/**
 * Application process use this method to notify Minitor process
 * that there is a new content in shared memory.
 * 
 * It will blocking wait for a response of minitor that the message is read.
*/
void notify_monitor_and_wait() {
    //reset signal hanlder
    signal(SIGUSR1, application_sig_handler); 

    VLC_DEBUG("application: send signal");
    kill(MONITOR_PID, SIGUSR1);
    
    // waiting monitor reset the memory
    // which indicate the message is read and processed
    VLC::Context vlc_config;
    while (true) {
        VLC_SHARED_MEM.read((void *) &vlc_config);
        if (!vlc_config.is_valid()) {
            break;
        }
    }
}

}
}

#endif // _VLC_COMMUNICATION_H_