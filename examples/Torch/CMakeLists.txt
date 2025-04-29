cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(benchmark)

list(APPEND CMAKE_PREFIX_PATH "/home/yyan/miniconda3/envs/dev/lib/python3.11/site-packages/torch/lib")

find_package(Torch REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")  O2 flag lead to issues in PyTorch + VLCs, avoid using it!
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")

add_subdirectory(benchmarks/DNN)
add_subdirectory(benchmarks/WideDNN)
add_subdirectory(benchmarks/GPTlite)
add_subdirectory(benchmarks/Transformer)
add_subdirectory(src/initialize_time)

add_executable(parallel_vlc benchmarks/parallel_vlc.cpp)
target_include_directories(parallel_vlc PRIVATE ../../include)

add_executable(parallel benchmarks/parallel.cpp)
target_include_directories(parallel PRIVATE ../../include)

add_executable(vlc src/initialize_time/vlc.cpp)
target_include_directories(vlc PRIVATE ../../include)
