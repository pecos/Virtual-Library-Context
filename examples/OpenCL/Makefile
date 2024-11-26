CXX=g++
NO_DEBUG=-DNDEBUG -O2# -g
CFLAGS=-I./include -I../../include -std=c++17 -Wall $(NO_DEBUG)

hello.so: src/hello.cpp
	$(CXX) -shared -o $@ $< -lOpenCL -fPIC

parallel_vlc: src/parallel_vlc.cpp hello.so
	$(CXX) $(CFLAGS) -o $@ $<

clean:
	rm -f hello.so parallel_vlc