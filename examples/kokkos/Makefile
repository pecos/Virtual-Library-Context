CXX_COMPILER=g++
NO_DEBUG=-DNDEBUG
CFLAGS=-I./include -I../../include -std=c++17 $(NO_DEBUG)

# kokkos settings
KOKKOS_DEVICES=Cuda
KOKKOS_CUDA_OPTIONS=enable_lambda
KOKKOS_ARCH="HSW,Pascal60"  #haswell CPU and P100 GPU

MAKEFILE_PATH := $(subst Makefile,,$(abspath $(lastword $(MAKEFILE_LIST))))

ifndef KOKKOS_PATH
  KOKKOS_PATH = $(MAKEFILE_PATH)lib/kokkos
endif

# NVCC settings
CXX=${KOKKOS_PATH}/bin/nvcc_wrapper
CXXFLAGS=-Wall $(NO_DEBUG) -I../../include -I./include -fPIC

override CXXFLAGS += -I$(MAKEFILE_PATH)

DEPFLAGS = -M
LINK = ${CXX}
LINKFLAGS =

MPICC=~/openmpi-5.0.3/build/bin/mpicc

MPI_CFLAGS=$(shell $(MPICC) -showme:compile)
MPI_CLDFLAGS=$(shell $(MPICC) -showme:link)

include $(KOKKOS_PATH)/Makefile.kokkos

libhelloworld.so: src/helloworld.cu
	nvcc -arch=sm_60 --cudart=shared --compiler-options '-fPIC' $< -shared -o $@

libkokkos_compute.so: src/kokkos_compute.cpp $(KOKKOS_LINK_DEPENDS) $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(KOKKOS_CXXLDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $< $(KOKKOS_LIBS) $(LIB) -shared -o $@

launcher_novlc: src/launcher_novlc.cpp src/kokkos_compute.cpp $(KOKKOS_LINK_DEPENDS) $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(KOKKOS_CXXLDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $< src/kokkos_compute.cpp $(KOKKOS_LIBS) $(LIB) -o $@

launcher: src/launcher.cpp libkokkos_compute.so ../../include/VLC/runtime.h
	$(CXX_COMPILER) $(CFLAGS) -o $@ $< -pthread -ldl

test_cuda: src/test_cuda.cu libhelloworld.so ../../include/VLC/runtime.h
	nvcc -arch=sm_60 --cudart=shared --compiler-options '$(CXXFLAGS)' $< -o $@

test_rocm: src/test_rocm.cpp librocmhelloworld.so ../../include/VLC/runtime.h
	$(CXX_COMPILER) $(CFLAGS) -o $@ $< -pthread -ldl

heat3d_mpi: src/mpi/heat3d.cpp $(KOKKOS_LINK_DEPENDS) $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(MPI_CFLAGS) $(MPI_CLDFLAGS) $(KOKKOS_CXXLDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $< $(KOKKOS_LIBS) $(LIB) -lmpi -o $@

libheat3d.so: src/vlc/heat3d_vlc.cpp
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(KOKKOS_CXXLDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $< $(KOKKOS_LIBS) $(LIB) -shared -o $@

heat3d_vlc: src/vlc/heat3d_launcher.cu libheat3d.so ../../include/VLC/runtime.h
	nvcc -arch=sm_60 --cudart=shared --compiler-options '$(CXXFLAGS)' $< -o $@ -ldl

initialize_time_vlc: src/initialize_time/vlc.cpp  libkokkos_compute.so ../../include/VLC/runtime.h
	nvcc -arch=sm_60 --cudart=shared --compiler-options '$(CXXFLAGS)' $< -o $@ -ldl

initialize_time_base: src/initialize_time/base.cpp  libkokkos_compute.so
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(KOKKOS_CXXLDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $< -L./ -lkokkos_compute $(KOKKOS_LIBS) $(LIB) -o $@

clean:
	rm -f *.o *.so launcher_novlc