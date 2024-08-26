# Introduction

**Virtual Library Contexts (VLCs)** is a library-level virtualization framework that encapsulate sets of libraries and provide performance isolation between them.

What VLCs **DO** for applications:
- partition compute resources between libraries to avoid contention
- compose potentially incompatible libraries
- safely parallelize thread-unsafe code
- enable nested parallelism that is not supported natively

What VLCs **NOT REQUIRE**:
- Library source code modification
- Library recompilation
- Major applications code changes
- Privileged OS features

# System Requirement

VLCs are designed only for GNU/Linux. 

VLCs are tested on a system with the following environment:
- Ubuntu 22.04
- Kernel 5.15.0
- glibc 2.35 (*for system with glibc < 2.34, a special setting of VLC is required.*)
- g++ 11.4.0
- Python 3.9

# Dependencies

A modified version of [Implib.so](https://github.com/yinengy/Implib.so) shim generator.

# Installation

VLCs is a header only library and no installation is required. To use it, simply include the header file "VLC/runtime.h" and "VLC/loader.h" in the application.

# Overview

A VLC is a virtualized context (environment) that libraries execute in. An application typically composed with several VLCs each has different resource partition.

If a library is loaded into multiple VLCs, it will be replicated and each copy has their own private states by replicating its static fields. It make the parallel execution of thread-unsafe code become possible by putting a library in several VLCs.

A typical use of VLCs involve the following step:
1. Initialize VLCs Runtime
2. Create threads for each VLC
3. assign avaliable resource for each VLC
4. load libraries into each VLC
5. call library APIs to make a computation (where the application code is)

To migrate an application to use VLCs, users could keep the existing application code unchanged on step 5 and add code for step 1-4 on the beginning of the application.

# Code Example
Currently VLCs support two modes: Manual Mode and Transparent Mode. User could pick anyone that is convenient for them.

## Transparent Mode

(NOTE: This is an experimental feature and not all libraries works on this mode (e.g. Galois).)

First, the `VLC::Runtime` should be declared and initialized by calling `vlc.initialize()` at the begging of main().

Then we make threads so each VLC could run in parallel.

To assign resource like available CPU cores to each VLC, we need to define a 
`VLC::Context` object with VLC ID (start from 1, since 0 is preserved for default namespace) and tid of current thread. `avaliable_cpu()` is called to set the number of CPUs and `register_vlc()` is called to tell runtime how the context should be.

To load libraries in to VLC, simply initialized a `VLC::Loader` object with VLC ID and an **absolute path** to the library and the last parameter is `true` to indicate we are using transparent mode.

The computation kernel code could then be executed directly. VLC Runtime will resolve the symbols automatically in transparent mode.

### Build shim
To use transparent mode, a shim is required to be generated for libraries in VLCs. We provide a tool chain to do that and there is a script for each of the application in examples. To make your own shim, please reference the script.

```C++
void launcher(int vlc_id) {
    // configure avalibale resouce for VLC
    VLC::Context vlc(vlc_id, gettid());
    if (vlc_id == 1)
        vlc.avaliable_cpu("0-11");
    else
        vlc.avaliable_cpu("12-23");
    VLC::register_vlc(&vlc);

    // load libraries into VLC
    VLC::Loader loader("/lib/x86_64-linux-gnu/libopenblas64.so.0", vlc_id, true);

    // run compute kernel
    do_some_compute(100);
}

int main() {
    // initialize VLC environment
    VLC::Runtime vlc;
    vlc.initialize();

    std::vector<std::thread> t(num_vlc);

    // make a thread for each VLC
    for (int i = 0; i < num_vlc; i++) 
        t[i] = std::thread(launcher, i+1);
    
    for (int i = 0; i < num_vlc; i++) 
        t[i].join();
}
```

## Manual Mode

This mode requires manually load all function symbols of the libraries used in the application code. This provides better compatibility than transparent mode.

User need to define the function pointer types of libraries they want to use at the top of the application code. And then put a mapping of the function name to its mangled symbols names in `register_functions()`.

When loading libraries into VLC, user need to get the function pointers manually from VLC by calling `loader.load_func<function_type>("function_name");`, and all libraries API used in the application need to be replaced with these function pointers. If a API in the libraries loaded into VLC is not used, there is no need to load that unused function pointer.

```C++
typedef void (*compute_t)(int N);

void register_functions() {
    // a map to the function symbol names (mangled)
    std::unordered_map<std::string, std::string> names{
        {"compute", "_Z6computei"}};  
    VLC::Loader::register_func_names(names);
}

void launcher(int vlc_id) {
    // configure avalibale resouce for VLC
    VLC::Context vlc(vlc_id, gettid());
    if (vlc_id == 1)
        vlc.avaliable_cpu("0-11");
    else
        vlc.avaliable_cpu("12-23");
    VLC::register_vlc(&vlc);

    // load libraries into VLC
    VLC::Loader loader("/lib/x86_64-linux-gnu/libopenblas64.so.0", vlc_id, false);
    // load functions from libraries
    auto compute = loader.load_func<compute_t>("compute");

    // run compute kernel
    compute(0);
}

int main() {
    // initialize VLC environment
    VLC::Runtime vlc;
    vlc.initialize();

    register_functions();

    std::vector<std::thread> t(num_vlc);

    // make a thread for each VLC
    for (int i = 0; i < num_vlc; i++) 
        t[i] = std::thread(launcher, i+1);
    
    for (int i = 0; i < num_vlc; i++) 
        t[i].join();
}
```