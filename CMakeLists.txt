cmake_minimum_required(VERSION 3.11.0)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(Virtual-Library-Contexts)

add_library(virtual-library-context INTERFACE)
target_include_directories(virtual-library-context
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

add_executable(test-dlmopen src/test_dlmopen.cpp)
set_property(TARGET test-dlmopen PROPERTY CXX_STANDARD 23)
target_include_directories(test-dlmopen PUBLIC)
target_link_libraries(test-dlmopen PRIVATE virtual-library-context)
target_compile_options(test-dlmopen PRIVATE -fPIC -ldl -g)
target_compile_features(test-dlmopen PUBLIC cxx_std_23)
