CXX=g++
# NO_DEBUG=-DNDEBUG
CFLAGS=-I./include -g $(NO_DEBUG)

pthread_patch.so: pthread_shim/pthread_patch.c
	gcc -shared -o $@ $< -fPIC

test_forge_getaffinity: tests/test_forge_getaffinity.cpp include/VLC.h
	$(CXX) $(CFLAGS) -o $@ $< -lseccomp

test_forge_cpu_file: tests/test_forge_cpu_file.cpp include/VLC.h
	$(CXX) $(CFLAGS) -o $@ $< -lseccomp

test_pthread: tests/test_pthread.cpp include/VLC.h
	$(CXX) $(CFLAGS) -o $@ $< -pthread -ldl

clean:
	rm -f test_forge_getaffinity test_pthread pthread_patch.so