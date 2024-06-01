EXAMPLE_FOLDER=/home/yyan/vlc/examples/kokkos
# use when glibc < 2.34
# PATCH_PATH=/home/yyan/vlc/libcudarts.so

PATCH_PATH=""
# LIB_PATH=$CUDA_PATH:$EXAMPLE_FOLDER/
LIB_PATH=/home/yyan/vlc/cudart_shim:$EXAMPLE_FOLDER


cd $EXAMPLE_FOLDER
make $1 -j32

if [ "$1" == "heat3d_mpi" ]; then
/home/yyan/openmpi-5.0.3/build/bin/mpirun -np 2 $EXAMPLE_FOLDER/${1}
else
rm address.tmp
# valgrind --trace-children=yes --leak-check=full --show-leak-kinds=all env LD_PRELOAD=$PATCH_PATH LD_LIBRARY_PATH=$LIB_PATH $EXAMPLE_FOLDER/${1}
# perf stat -B -e task-clock,context-switches,cpu-migrations,page-faults,branch-misses,cache-references,cache-misses,cycles,instructions,branches,faults env LD_PRELOAD=$PATCH_PATH LD_LIBRARY_PATH=$LIB_PATH $EXAMPLE_FOLDER/${1}
time LD_LIBRARY_PATH=$LIB_PATH $EXAMPLE_FOLDER/${1}
fi
