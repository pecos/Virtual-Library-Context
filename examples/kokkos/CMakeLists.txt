cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(benchmark LANGUAGES CXX CUDA)

list(APPEND CMAKE_PREFIX_PATH "/home/yyan/vlc/examples/kokkos/lib/kokkos/install/lib")
set(CMAKE_CXX_COMPILER "/home/yyan/vlc/examples/kokkos/lib/kokkos/install/bin/nvcc_wrapper")

find_package(Kokkos 4.6 REQUIRED CONFIG)

find_package(MPI REQUIRED)
if (MPI_FOUND)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
else (MPI_FOUND)
    message(SEND_ERROR "This application cannot compile without MPI")
endif (MPI_FOUND)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DNDEBUG")

set(CMAKE_CUDA_ARCHITECTURES 60)
set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared")

add_executable(heat3d_mpi src/heat3d/heat3d_mpi.cpp)
target_link_libraries(heat3d_mpi PRIVATE Kokkos::kokkos ${MPI_LIBRARIES})

add_library(heat3d SHARED src/heat3d/heat3d_kernel.cpp)
target_link_libraries(heat3d PRIVATE Kokkos::kokkos)

add_executable(heat3d_vlc src/heat3d/heat3d_vlc.cu)
set_target_properties(heat3d_vlc PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(heat3d_vlc PRIVATE CUDA::cudart)
target_link_libraries(heat3d_vlc PRIVATE CUDA::cuda_driver)
target_include_directories(heat3d_vlc PRIVATE ../../include)


# add_executable(parallel benchmarks/parallel.cpp)
# target_include_directories(parallel PRIVATE ../../include)

# add_executable(vlc src/initialize_time/vlc.cpp)
# target_include_directories(vlc PRIVATE ../../include)
