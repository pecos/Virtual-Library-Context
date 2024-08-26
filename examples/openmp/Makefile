CXX=g++
NO_DEBUG=-DNDEBUG
CFLAGS=-I./include -I../../include -g -std=c++17 -Wall $(NO_DEBUG)

libaddmp.so: src/addmp.cpp
	$(CXX) $(CFLAGS) -c -o addmp.o $< -pthread -fopenmp -fPIC
	$(CXX) $(CFLAGS) -shared -o $@ addmp.o -pthread -fopenmp -fPIC

libsummp.so: src/summp.cpp
	$(CXX) $(CFLAGS) -c -o summp.o $< -pthread -fopenmp -fPIC
	$(CXX) $(CFLAGS) -shared -o $@ summp.o -pthread -fopenmp -fPIC

libpowermp.so: src/powermp.cpp
	$(CXX) $(CFLAGS) -c -o powermp.o $< -pthread -fopenmp -fPIC
	$(CXX) $(CFLAGS) -shared -o $@ powermp.o -pthread -fopenmp -fPIC

launcher_vlc: src/launcher_vlc.cpp libaddmp.so libsummp.so ../../include/VLC/runtime.h
	$(CXX) $(CFLAGS) -o $@ $< -Wl,--undefined=__pthread_create -pthread -fopenmp -ldl

launcher_multiple: src/launcher_multiple.cpp libaddmp.so libsummp.so libpowermp.so ../../include/VLC/runtime.h
	$(CXX) $(CFLAGS) -o $@ $< -pthread -fopenmp -ldl

launcher_novlc: src/launcher_novlc.cpp src/addmp.cpp src/summp.cpp src/powermp.cpp
	$(CXX) $(CFLAGS) -o $@ $^ -pthread -fopenmp

launcher_vlc_transparency: src/launcher_vlc_transparency.cpp ../../include/VLC/loader.h
	$(CXX) $(CFLAGS) -o $@ $^ -fopenmp

initialize_time_vlc: src/initialize_time/vlc.cpp  libaddmp.so ../../include/VLC/runtime.h
	$(CXX) $(CFLAGS) -o $@ $<

initialize_time_base: src/initialize_time/base.cpp  libaddmp.so
	$(CXX) $(CFLAGS) -o $@ $< -L./ -laddmp

clean:
	rm -f *.o *.so launcher_vlc launcher_vlc_transparency launcher_novlc launcher_multiple *.so initialize_time_vlc