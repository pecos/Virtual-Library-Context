# C compiler
CC = g++
ICC = icc
CC_FLAGS = -g -fopenmp -O2

all: bfs bfs_offload

bfs: bfs.cpp 
	$(CC) $(CC_FLAGS) bfs.cpp -o bfs

bfs.so: bfs.cpp 
	$(CC) $(CC_FLAGS) bfs.cpp -shared -o bfs.so -fPIC

bfs_offload: bfs.cpp
	$(ICC) $(CC_FLAGS) -DOMP_OFFLOAD bfs.cpp -o bfs_offload

clean:
	rm -f bfs bfs.so bfs_offload result.txt
