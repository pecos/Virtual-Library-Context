CXX=g++
NO_DEBUG=-DNDEBUG # -g
CFLAGS=-I../../include -std=c++17 -Wall $(NO_DEBUG)

# benchmark
cascade_train: benchmarks/cascade_train.c
	$(CXX) $(CFLAGS) -o $@ $< -lfann

steepness_train: benchmarks/steepness_train.c
	$(CXX) $(CFLAGS) -o $@ $< -lfann

xor_train: benchmarks/xor_train.c
	$(CXX) $(CFLAGS) -o $@ $< -lfann
	
cascade_train_transparent: benchmarks/cascade_train_transparent.cpp ../../include/VLC/runtime.h
	$(CXX) $(CFLAGS) -o $@ $< -lfann

steepness_train_transparent: benchmarks/steepness_train_transparent.cpp ../../include/VLC/runtime.h
	$(CXX) $(CFLAGS) -o $@ $< -lfann

xor_train_transparent: benchmarks/xor_train_transparent.cpp ../../include/VLC/runtime.h
	$(CXX) $(CFLAGS) -o $@ $< -lfann

clean:
	rm -f xor_fixed.data xor_train_transparent steepness_train_transparent xor_train steepness_train cascade_train_transparent cascade_train *.net *.so *.so.* *.S *.ver *.init.c vlc_callback.c vlc_hashmap.c vlc_hashmap.h