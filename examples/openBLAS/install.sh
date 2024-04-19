# compile arpack-ng
cd lib/arpack-ng
mkdir build
cd build
cmake -D EXAMPLES=ON -D MPI=ON -D BUILD_SHARED_LIBS=ON ..
make

# install globally
sudo make install
# verify
make test

# ezarpack
cd ../lib/arpack-ng
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/home/yyan/vlc/examples/arpack/lib/ezARPACK/install -DExamples=ON -DTests=ON
# verify
make test