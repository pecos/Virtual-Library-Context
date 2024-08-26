CXX=g++
NO_DEBUG=-DNDEBUG # -g

CFLAGS=-I. -I../../include -std=c++17 -L./build -Wall $(NO_DEBUG)

build/libfoobar.so: foobar.cpp foobar.h
	$(CXX) $(CFLAGS) -shared -o $@ $^ -fPIC

test_transparency: test_transparency.cpp build/libfoobar.so
	$(CXX) $(CFLAGS) -o $@ $< -ldl -lfoobar

clean:
	rm -f build/libfoobar.so test_transparency