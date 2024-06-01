#ifndef _VLC_LOADER_H_
#define _VLC_LOADER_H_

#include <string>
#include <unordered_map>

#include "VLC/info.h"

static std::unordered_map<std::string, std::string> function_names_map;

namespace VLC {

/**
 * Register a map between function name and mangled name (from libraries symbol table)
 * 
 * @param names_map a map that key is unmangled name or function and value is mangled name
*/
void register_func_names(std::unordered_map<std::string, std::string> names_map) {
    function_names_map = names_map;
}

/**
 * Load a funtion pointer from VLC by name.
 * The function should have been registered by ```register_func_names```
 * 
 * This is a wrapper around ```dlsym()```
 * @param handle a handle of loaded libraries in a VLC
 * @param function_name a name of function to load
*/
template <typename FuncType>
FuncType load_func(void *handle, std::string function_name) {
    auto result = function_names_map.find(function_name);
    if (result == function_names_map.end()) {
        VLC_DIE("VLC: try to load a function which is not registered, name=%s", function_name.c_str());
    }

    std::string mangled_name = result->second;
    VLC_DEBUG("VLC: try to load function %s\n", function_name.c_str());
    FuncType func_ptr = (FuncType) dlsym(handle, mangled_name.c_str());
    if (func_ptr == NULL) {
        VLC_DIE("VLC: failed to load function %s, error in dlsym=%s", function_name.c_str(), dlerror());
    }

    return func_ptr;
}

}

#endif // _VLC_LOADER_H_