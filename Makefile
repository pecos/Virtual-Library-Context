CXX=g++
CFLAGS=-I./include -DNDEBUG -O3

pthread_patch.so: pthread_shim/pthread_patch.c
	gcc -shared -o $@ $< -fPIC

test_syscall_intercept: tests/test_syscall_intercept.cpp include/VLC.h
	$(CXX) $(CFLAGS) -o $@ $< -lseccomp

test_pthread: tests/test_pthread.cpp
	$(CXX) -o $@ $< -pthread -ldl -g

clean:
	rm -f test_syscall_intercept test_pthread