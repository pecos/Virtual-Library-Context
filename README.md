# Introduction

**Virtual Library Contexts (VLCs)** is a library-level virtualization framework that encapsulate sets of libraries and provide performance isolation between them.

What VLCs **DO** for applications:
- partition compute resources between libraries to avoid contention
- compose potentially incompatible libraries
- safely parallelize thread-unsafe code
- enable nested parallelism that is not supported natively

What VLCs **DO NOT** REQUIRE:
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

- A modified version of [Implib.so](https://github.com/yinengy/Implib.so) shim generator, need to be initialized as a submodule.
- libseccomp-dev

# Installation

VLCs is a header only library and no installation is required. To use it, simply include the header file "VLC/runtime.h" and "VLC/loader.h" in the application.

To run examples of VLCs

```
apt install libseccomp-dev
git clone --recurse-submodules -j8 git@github.com:pecos/Virtual-Library-Context.git
cd Virtual-Library-Context
./scripts/run_openmp.sh launcher_vlc
./examples/openmp/run_transparency.sh
```

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

(NOTE: This is an experimental feature and not all libraries work on this mode (e.g. Galois).)

### Build shim
To use transparent mode, a shim is required to be generated for libraries in VLCs. We provide a toolchain to do that and there is a script  `run_transparency.sh` to compile shim and launch VLC in transparency mode for each of the applications in the `example` folder. 

To make your own shim, please reference the script [run_transparency.sh](https://github.com/pecos/Virtual-Library-Context/blob/master/examples/openBLAS/run_transparency.sh). All you need to do is update `LIB_FILE` which is the absolute path to the library in the script, and also make a `symbols.txt` file which contains a list of symbols of the library to be used in the applications. 

To ease the burden of finding symbol names in complicated libraries, we provide a (script)[https://github.com/pecos/Virtual-Library-Context/blob/master/scripts/gen_shim_symbols.py] to automatically print all function symbols that the applications would use. Its usage is
```
> python3 gen_shim_symbols.py <application_binary> <abs_path_of_library>
```

### Config VLCs

First, the `VLC::Runtime` should be declared and initialized by calling `vlc.initialize()` at the beginning of main().

Then we make threads so each VLC could run in parallel.

To assign resources like available CPU cores to each VLC, we need to define a 
`VLC::Context` object with VLC ID (start from 1, since 0 is preserved for default namespace) and tid of current thread. `avaliable_cpu()` is called to set the number of CPUs and `register_vlc()` is called to tell runtime how the context should be.

To load libraries into VLC, simply initialize a `VLC::Loader` object with VLC ID and an **absolute path** to the library and the last parameter is `true` to indicate we are using transparent mode. The default number of VLCs is 2 in the script, please remeber to update the number `NUM_VLC` if you need more VLCs.

The computation kernel code could then be executed directly. VLC Runtime will resolve the symbols automatically in transparent mode.

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

This mode requires manually loading all function symbols of the libraries used in the application code. This provides better compatibility than transparent mode. And there is no need to generate a shim.

User need to define the function pointer types of libraries they want to use at the top of the application code. And then put a mapping of the function name to its mangled symbols names in `register_functions()`. To find the mangled symbols in your object, try
```
> objdump -T <hello.so> | grep "hello"
``` 

When loading libraries into VLC, user needs to get the function pointers manually from VLC by calling `loader.load_func<function_type>("function_name");`, and all libraries API used in the application need to be replaced with these functions pointers. If an API in the libraries loaded into VLC is not used, there is no need to load that unused function pointer.

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

# GPU Support
(Experimental features, please report issues if you see errors when using GPUs with VLCs)
To support libraries using GPU (e.g. Kokkos), VLCs provide *VLC Service* feature which is a dedicated VLC that encapsulates libraries that are dlmopen-incompatible (e.g. CUDA runtime).

Just like the Transparent Mode, VLC Service relies on the generated shim of the target libraries so the application can link to shim and the shim will be responsible for dispatching API requests to the VLC Service. Without the use of VLC Service, applications will get an error when trying to load CUDA runtime into a VLC.

How to generate a shim for VLC Service could be found by reference [gen_cudart_shim.sh](https://github.com/pecos/Virtual-Library-Context/blob/master/scripts/gen_cudart_shim.sh). And the Kokkos examples provide an example of VLC Service for CUDA.

## Known Libraries that required VLC Service support
Those libraries are dlmopen-incompatible and require the use of VLC Service as a workaround. There may exist more libraries and the list will be updated once we find them.
- CUDA
- Pthread (only for glibc < 2.34)
- OpenCL

## Known Issues

Issues to be fixed.

- Transparent Mode does not work on Galois
- Transparent Mode may not work with VLC Service
- Dynamic data symbols are not supported in Transparent Mode yet
- The max number of VLCs is limited by the number of linker namespaces (less than 16)
- Compiling with compiler optimization enabled may break transparent mode for certain libraries (e.g. oneDNN)
