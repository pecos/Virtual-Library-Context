#ifndef _VLC_RESOURCE_H_
#define _VLC_RESOURCE_H_

#include <unordered_map>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <vector>

#include "VLC/info.h"

static const std::string VLC_RESOURCE_ROOT = "/tmp/vlc/";

namespace VLC {
namespace Internal {

std::unordered_map<pid_t, int> pid_to_vlc_id;
std::unordered_map<int, std::vector<int>> vlc_id_to_core_map;
std::unordered_map<pid_t, cpu_set_t> virtual_cpu_sets;
cpu_set_t system_cpu_set;

/**
 * Create a virtualized affinity if not created before
 */
void create_virtual_affinity(pid_t pid) {
    if (virtual_cpu_sets.find(pid) == virtual_cpu_sets.end()) {
        cpu_set_t virtual_cpu_set = system_cpu_set;

        // find the core_map from info we saved from register_vlc()
        std::vector<int> core_map = vlc_id_to_core_map[pid_to_vlc_id[pid]];

        CPU_ZERO(&virtual_cpu_set);
        for (auto i: core_map) {
            CPU_SET(i, &virtual_cpu_set);
        }

        virtual_cpu_sets[pid] = std::move(virtual_cpu_set);
    }
}

void enfore_virtual_affinity(pid_t pid) {
    if (sched_setaffinity(pid, sizeof(virtual_cpu_sets[pid]), &virtual_cpu_sets[pid]) == -1) {
        VLC_DIE("VLC: unable to set cpu set, %s", strerror(errno));
    }
}

class Resource {
public:
    Resource() = default;

    ~Resource() = default;

    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;
    Resource(Resource&&) = delete;
    Resource& operator=(Resource&&) = delete;
    
    /**
     * update the avalible mem for a VLC.
     * 
     * @param vlc_id the ID of the VLC
     * @param mem_size the amount of memory to be assigned (in GB).
     *     
    */
    void set_avaliable_mem(int vlc_id, int mem_size) {
        mem_limit[vlc_id] = mem_size;
    }

    /**
     * check if there is a mem limit set on the VLC
     * 
     * @param vlc_id the ID of the VLC
     * @return True if there is a mem limit on this VLC
     * 
    */
    bool has_mem_limit(int vlc_id) {
        return mem_limit.find(vlc_id) != mem_limit.end();
    }

    /**
     * generate a temporal file in replace of /proc/meminfo.
     * if has generated before, do nothing and return the previous file path
     * 
     * @param vlc_id the ID of the VLC
     * @param file_path a pointer to char array of the path to generated mem file
     *
     * @note assume ```mem_limit[vlc_id]``` is set
    */
    void generate_mem_info_file(int vlc_id, char *file_path) {
        auto result = mem_file_path.find(vlc_id);
        if (result != mem_file_path.end()) {
            std::strcpy(file_path, result->second);
            return;
        }

        // create temporal resource folder if not exist
        std::string resource_folder = VLC_RESOURCE_ROOT + std::to_string(vlc_id);
        std::string resource_path = resource_folder + "/meminfo"; 
        std::filesystem::create_directories(resource_folder);

        std::ifstream original_mem_file("/proc/meminfo");
        if (!original_mem_file.is_open()) {
            VLC_DIE("VLC: unable to open /proc/meminfo");
        }

        std::ofstream forged_mem_file(resource_path);
        if (!forged_mem_file.is_open()) {
            VLC_DIE("VLC: cannot create %s", resource_path.c_str());
        }

        int avaliable_mem_in_KB = mem_limit[vlc_id] * 1024 * 1024;

        std::string line;
        int lineCount = 0;
        while (std::getline(original_mem_file, line)) {
            if (lineCount == 0) {
                forged_mem_file << "MemTotal:       " << avaliable_mem_in_KB << " kB" << std::endl;
            } else if (lineCount == 1) {
                forged_mem_file << "MemFree:        " << avaliable_mem_in_KB << " kB" << std::endl;
            } else if (lineCount == 2) {
                forged_mem_file << "MemAvailable:   " << avaliable_mem_in_KB << " kB" << std::endl;
            } else {
                forged_mem_file << line << std::endl;
            }
            lineCount++;
        }

        original_mem_file.close();
        forged_mem_file.close();

        // convert the file path to a new allocated cstring
        char* cstr = new char[resource_path.size() + 1];  // Allocate memory for the C-style string
        std::memcpy(cstr, resource_path.c_str(), resource_path.size() + 1);        // Copy the contents of the std::string to the C-style string
        // cstr[resource_path.size()] = 0;
        mem_file_path[vlc_id] = cstr;
        std::strcpy(file_path, cstr);
        return;
    }

private:
    std::unordered_map<int, int> mem_limit;  // the virtualized avalible mem for VLCs 
    std::unordered_map<int, char*> mem_file_path;  // path to forged mem files
};

}

struct Context {
    int id;
    pid_t thread_id;
    char cpu_str[256] = {0};
    bool valid = false;

    Context() = default;

    Context(int vlc_id) {
        id = vlc_id;
        valid = true;
    }

    Context(int vlc_id, pid_t thread_id) {
        id = vlc_id;
        this->thread_id = thread_id;
        valid = true;
    }

    inline void register_thread(pid_t thread_id) {
        this->thread_id = thread_id;
    }

    /**
     * configure the cores avalible to this VLC
     * 
     * @param cpu_str a string represent the avalibale cpu ranges.
     *      Format:
     *          "0-3": cpu 0 1 2 3
     *          "0,1,2,3": cpu 0 1 2 3
     *          "0-1,7-8": cpu 0 1 7 8
     * @note maximum string length is 256
    */
    inline void avaliable_cpu(const char *cpu_str) {
        strcpy(this->cpu_str, cpu_str);
    }

    inline void reset() {
        valid = false;
    }

    inline bool is_valid() {
        return valid;
    }
};

}

#endif // _VLC_RESOURCE_H_