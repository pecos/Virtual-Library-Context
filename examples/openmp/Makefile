CXX=g++
NO_DEBUG=-DNDEBUG
CFLAGS=-I./include -I../../include -g -std=c++11 $(NO_DEBUG)

libaddmp.so: src/addmp.cpp
	$(CXX) $(CFLAGS) -c -o src/addmp.o $< -pthread -fopenmp -fPIC
	$(CXX) $(CFLAGS) -shared -o $@ src/addmp.o -pthread -fopenmp -fPIC

libsummp.so: src/summp.cpp
	$(CXX) $(CFLAGS) -c -o src/summp.o $< -pthread -fopenmp -fPIC
	$(CXX) $(CFLAGS) -shared -o $@ src/summp.o -pthread -fopenmp -fPIC

launcher: src/launcher.cpp libaddmp.so libsummp.so ../../include/VLC.h
	$(CXX) $(CFLAGS) -o $@ $< -Wl,--undefined=__pthread_create -pthread -fopenmp -ldl

launcher_novlc: src/launcher_novlc.cpp src/addmp.cpp src/summp.cpp
	$(CXX) $(CFLAGS) -o $@ $< src/addmp.cpp src/summp.cpp -pthread -fopenmp


clean:
	rm -f libaddmp.so libsummp.so launcher