CXX=g++
NO_DEBUG=-DNDEBUG
CFLAGS=-I./include -g $(NO_DEBUG)

pthread_patch.so: pthread_shim/pthread_patch.c
	gcc -shared -o $@ $< -fPIC

test_syscall_intercept: tests/test_syscall_intercept.cpp include/VLC.h
	$(CXX) $(CFLAGS) -o $@ $< -lseccomp

test_pthread: tests/test_pthread.cpp include/VLC.h
	$(CXX) $(CFLAGS) -o $@ $< -pthread -ldl

clean:
	rm -f test_syscall_intercept test_pthread pthread_patch.so