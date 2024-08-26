EXAMPLE_FOLDER=examples/microbenchmark
# use when glibc < 2.34
# PATCH_PATH=/home/yyan/vlc/libcudarts.so

PATCH_PATH=""
# LIB_PATH=$CUDA_PATH:$EXAMPLE_FOLDER/
if [ "$1" == "cuda_overhead_vlc" ]; then
rm address.tmp
LIB_PATH=cudart_shim:$EXAMPLE_FOLDER
elif [ "$1" == "helloworld_overhead_vlc" ]; then
rm address.tmp
LIB_PATH=examples/microbenchmark/helloworld_stub_overhead/hello_stub/:$EXAMPLE_FOLDER
else
LIB_PATH=$EXAMPLE_FOLDER
fi


# make -C $EXAMPLE_FOLDER $1 -j32

# valgrind --trace-children=yes --leak-check=full --show-leak-kinds=all env LD_PRELOAD=$PATCH_PATH LD_LIBRARY_PATH=$LIB_PATH $EXAMPLE_FOLDER/${1}
# perf stat -B -e task-clock,context-switches,cpu-migrations,page-faults,branch-misses,cache-references,cache-misses,cycles,instructions,branches,faults env LD_PRELOAD=$PATCH_PATH LD_LIBRARY_PATH=$LIB_PATH $EXAMPLE_FOLDER/${1}
time LD_LIBRARY_PATH=$LIB_PATH $EXAMPLE_FOLDER/${1}
