cmake_minimum_required(VERSION 3.11.0)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(Virtual-Library-Contexts)

add_library(virtual-library-context INTERFACE)
target_include_directories(virtual-library-context
  INTERFACE $<BUILD_INTERFACE:/include/VLC>
)
