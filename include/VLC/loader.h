#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#ifndef _VLC_LOADER_H_
#define _VLC_LOADER_H_

#include <thread>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <pthread.h>
#include <dlfcn.h>
#include <unistd.h>
#include <fstream>
#include <numeric>
#include <filesystem>
#include <mutex>

#include <string>
#include <unordered_map>

#include "VLC/info.h"
#include "VLC/resource.h"
#include "VLC/communication.h"

static std::mutex register_vlc_mutex;
static std::unordered_map<std::string, std::string> function_names_map;

namespace VLC {

/**
 * Create a VLC loader which loads an set of libraries 
 * into a virtualized environment. There are two mode:
 * 
 * Mode 1: Manual
 * Function pointers has to be aquired by ```load_func``` muanually.
 * And all functions need to be pre-registered by ```register_func_names```
 * 
 * Mode 2: Transparent Mode
 * Functions are transparently resolved and replaced with symbols from VLC's shim.
 * There is no need to use ```load_func``` manually.
 * 
*/
class Loader {
public:

    /**
     * @param use_transparency when true, use transparency mode, 
     *          otherwise manual mode.
     * @param libpath path to shared object of a library, 
     *      when transparent mode (a shim) is used, please give an absolute path.
     *      Otherwise linker may be confused between the real library and the shim.
     * @param vlc_id the ID of the VLC that libraries should be loaded to
     *      0 means the main() namespace, user should pick a number other than 0 for a new VLC.
    */
    Loader(const char * libpath, int vlc_id, bool use_transparent_mode) {
        if (use_transparent_mode) {
            load_vlc_transparent_mode(libpath, vlc_id);
        } else {
            lib_handle = dlmopen(LM_ID_NEWLM, libpath, RTLD_NOW);
            if (lib_handle == NULL) {
                VLC_DIE("VLC: error in `dlmopen`: %s\n", dlerror());
            }
        }
    }

    /**
     * Load a funtion pointer from VLC by name.
     * The function should have been registered by ```register_func_names```
     * 
     * This is a wrapper around ```dlsym()```
     * @param function_name a name of function to load
    */
    template <typename FuncType>
    FuncType load_func(std::string function_name) {
        auto result = function_names_map.find(function_name);
        if (result == function_names_map.end()) {
            VLC_DIE("VLC: try to load a function which is not registered, name=%s", function_name.c_str());
        }

        std::string mangled_name = result->second;
        VLC_DEBUG("VLC: try to load function %s\n", function_name.c_str());
        FuncType func_ptr = (FuncType) dlsym(lib_handle, mangled_name.c_str());
        if (func_ptr == NULL) {
            VLC_DIE("VLC: failed to load function %s, error in dlsym=%s", function_name.c_str(), dlerror());
        }

        return func_ptr;
    }

    /**
     * Register a map between function name and mangled name (from libraries symbol table)
     * 
     * @param names_map a map that key is unmangled name or function and value is mangled name
    */
    static void register_func_names(std::unordered_map<std::string, std::string> names_map) {
        function_names_map = names_map;
    }

private:
    void *lib_handle;  // a handler to libraries (created by dlmopen)

    static void load_vlc_transparent_mode(const char * libpath, int vlc_id) {
        char *error;

        void *lib_handle = dlmopen(LM_ID_NEWLM, libpath, RTLD_NOW);
        if (lib_handle == NULL) {
            VLC_DIE("Error in `dlmopen`: %s\n", dlerror());
        }

        std::string line;
        std::ifstream symbol_file("symbols.txt");
        if (!symbol_file.is_open()) {
            VLC_DIE("Error in open symbols.txt file\n");
        }

        // dump the address of functions to file
        std::ofstream address_file;
        address_file.open("address." + std::to_string(vlc_id), std::fstream::trunc);
        
        while (std::getline(symbol_file, line)) {
            void *func_addr = dlsym(lib_handle, line.c_str());
            address_file << (unsigned long) func_addr << std::endl;
        }

        address_file.close();
        symbol_file.close();

        typedef void (*reload_library_symbols_t)(int );
        // open the shim object (which has the same filename as the loaded library)
        auto shim_handler = dlopen(std::filesystem::path(libpath).filename().c_str(), RTLD_LAZY | RTLD_LOCAL | RTLD_DEEPBIND);
        if ((error = dlerror()) != NULL)  {
            VLC_DIE("Error in dlopen: %s\n", error);
        }
        auto reload_library_symbols = (reload_library_symbols_t) dlsym(shim_handler, "reload_library_symbols");
        if ((error = dlerror()) != NULL)  {
            VLC_DIE("Error in dlsym: %s\n", error);
        }
        reload_library_symbols(vlc_id);
    }
};


void register_vlc(VLC::Context *vlc_config_ptr) {
    register_vlc_mutex.lock();
    Internal::SharedMem shared_mem = Internal::get_vlc_shared_mem();

    shared_mem.write((void *) vlc_config_ptr);

    Internal::notify_monitor_and_wait();
    register_vlc_mutex.unlock();
}

}

#endif // _VLC_LOADER_H_