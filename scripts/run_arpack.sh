EXAMPLE_FOLDER=/home/yyan/vlc/examples/arpack
# use when glibc < 2.34
# PATCH_PATH=/home/yyan/vlc/pthread_patch.so 
# LIB_PATH=/home/yyan/vlc/:$EXAMPLE_FOLDER/
PATCH_PATH=""
LIB_PATH=$EXAMPLE_FOLDER/

cd $EXAMPLE_FOLDER
make $1
# valgrind --trace-children=yes --leak-check=full --show-leak-kinds=all env LD_PRELOAD=$PATCH_PATH LD_LIBRARY_PATH=$LIB_PATH $EXAMPLE_FOLDER/${1}
# perf stat -B -e task-clock,context-switches,cpu-migrations,page-faults,branch-misses,cache-references,cache-misses,cycles,instructions,branches,faults env LD_PRELOAD=$PATCH_PATH LD_LIBRARY_PATH=$LIB_PATH $EXAMPLE_FOLDER/${1}
time LD_PRELOAD=$PATCH_PATH LD_LIBRARY_PATH=$LIB_PATH $EXAMPLE_FOLDER/${1}
# LD_LIBRARY_PATH=$LIB_PATH strace $EXAMPLE_FOLDER/${1}
# rm address.tmp