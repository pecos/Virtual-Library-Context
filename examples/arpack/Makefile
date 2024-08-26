CXX=g++
NO_DEBUG=-DNDEBUG
CFLAGS=-I./include -I../../include -I/home/yyan/vlc/examples/arpack/lib/ezARPACK/install/include -g -std=c++17 -Wall $(NO_DEBUG)

libeign.so: src/eign.cpp
	$(CXX) $(CFLAGS) -shared -o $@ $< -larpack -fPIC

eign_launcher_vlc_transparency: src/eign_launcher_vlc_transparency.cpp src/eign.cpp include/eign.h ../../include/VLC/runtime.h
	$(CXX) $(CFLAGS) -o $@ $^ -larpack

eign_launcher_vlc: src/eign_launcher_vlc.cpp libeign.so ../../include/VLC/runtime.h
	$(CXX) $(CFLAGS) -o $@ $<

eign_launcher_seq: src/eign_launcher_seq.cpp src/eign.cpp include/eign.h
	$(CXX) $(CFLAGS) -o $@ $^ -larpack

initialize_time_vlc: src/initialize_time/vlc.cpp  libeign.so ../../include/VLC/runtime.h
	$(CXX) $(CFLAGS) -o $@ $<

initialize_time_base: src/initialize_time/base.cpp  libeign.so
	$(CXX) $(CFLAGS) -o $@ $< -L./ -larpack -leign

clean:
	rm -f *.so *.so.2 eign_launcher_vlc eign_launcher_vlc_transparency eign_launcher_seq initialize_time_vlc initialize_time_base

