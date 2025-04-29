rm -rf build
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/home/yyan/vlc/examples/Torch/libtorch ..
cmake --build . --config Release -j 24