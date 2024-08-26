#!/bin/bash

srcdir=$(pwd)/lib/kokkos
builddir=$(pwd)/lib/kokkos/build
kokkos_install_prefix=${srcdir}/install
compiler_used_to_build_kokkos=${srcdir}/bin/nvcc_wrapper

rm -rf ${kokkos_install_prefix}
mkdir ${kokkos_install_prefix}
mkdir ${builddir}

cmake -S ${srcdir} -B ${builddir} \
  -DCMAKE_INSTALL_PREFIX=${kokkos_install_prefix} \
  -DCMAKE_CXX_COMPILER=${compiler_used_to_build_kokkos} \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ENABLE_CUDA_LAMBDA=ON \
  -DKokkos_ARCH_PASCAL60=ON

# ../generate_makefile.bash --cxxflags="-fPIC" --ldflags="-fPIC" --arch=Pascal60 --prefix=../lib --with-cuda --with-cuda-options="enable_lambda"

# make -j32 kokkoslib
# make -j32 install

# cd ..

# rm -rf cpu_build
# mkdir cpu_build
# cd cpu_build

# ../generate_makefile.bash --cxxflags="-fPIC" --ldflags="-fPIC" --arch=HSW --prefix=../lib

# make -j32 kokkoslib
# make -j32 install

# cd ..

# ncpu=1
# ngpu=4

# for i in $(seq 0 $ncpu);
# do
# 	cp -R cpu_build "cpu_build_${i}"
# done

# for i in $(seq 0 $ngpu);
# do
# 	cp -R gpu_build "gpu_build_${i}"
# done