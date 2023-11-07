CXX=g++
CFLAGS=-I./include -DNDEBUG -O3

test_syscall_intercept: test/test_syscall_intercept.cpp include/VLC.h
	$(CXX) $(CFLAGS) -o $@ $< -lseccomp 

clean:
	rm -f test_syscall_intercept