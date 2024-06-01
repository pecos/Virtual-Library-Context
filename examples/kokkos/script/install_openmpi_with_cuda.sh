wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.3.tar.bz2
tar xf openmpi-5.0.3.tar.bz2
cd openmpi-5.0.3
mkdir build
# sudo ln -s /usr/lib/pkgconfig/cuda-11.7.pc /usr/lib/x86_64-linux-gnu/pkgconfig/cuda.pc
./configure --with-cuda=/usr/local/cuda  --with-cuda-libdir=/usr/lib/x86_64-linux-gnu --enable-mca-dso=btl-smcuda,rcache-rgpusm,rcache-gpusm,accelerator-cuda --prefix=$(pwd)/build 2>&1 | tee config.out
make -j 8 all 2>&1 | tee make.out
make install 2>&1 | tee install.out