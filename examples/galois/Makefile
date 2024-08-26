CXX=g++
NO_DEBUG=-DNDEBUG
HEADER_GALOIS_DIR=lib/Galois/install/include
LIBGALOIS_DIR=lib/Galois/build/libgalois
CFLAGS=-I./include -I../../include -I$(HEADER_GALOIS_DIR) -L$(LIBGALOIS_DIR) -std=c++17 -Wall $(NO_DEBUG)

libtestgalois.so: src/testgalois.cpp
	$(CXX) $(CFLAGS) -shared -o $@ $< -lgalois_shmem -pthread -lnuma -fPIC

libbfsgalois.so: src/bfs.cpp
	$(CXX) $(CFLAGS) -shared -o $@ $< -lgalois_shmem -pthread -lnuma -fPIC

launcher_vlc: src/launcher_vlc.cpp libtestgalois.so ../../include/VLC/runtime.h
	$(CXX) $(CFLAGS) -o $@ $< -pthread -ldl

launcher_novlc: src/launcher_novlc.cpp src/testgalois.cpp
	$(CXX) $(CFLAGS) -o $@ $< src/testgalois.cpp -pthread -lgalois_shmem -lnuma

launcher_single: src/launcher_single.cpp libtestgalois.so
	$(CXX) $(CFLAGS) -o $@ $< -ldl

bfs_launcher_vlc: src/bfs_launcher_vlc.cpp libbfsgalois.so ../../include/VLC/runtime.h
	$(CXX) $(CFLAGS) -o $@ $< -lgalois_shmem -pthread -ldl

bfs_launcher_novlc: src/bfs_launcher_novlc.cpp libbfsgalois.so
	$(CXX) $(CFLAGS) -o $@ $< -lgalois_shmem -pthread -ldl

bfs_launcher_vlc_transparency: src/bfs_launcher_vlc_transparency.cpp src/bfs.cpp ../../include/VLC/runtime.h
	$(CXX) $(CFLAGS) -o $@ $^ -lgalois_shmem

initialize_time_vlc: src/initialize_time/vlc.cpp  libbfsgalois.so ../../include/VLC/runtime.h
	$(CXX) $(CFLAGS) -o $@ $<

initialize_time_base: src/initialize_time/base.cpp  libbfsgalois.so
	$(CXX) $(CFLAGS) -o $@ $< -L./ -lgalois_shmem -lbfsgalois

clean:
	rm -rf bfs_launcher_vlc_transparency initialize_time_base initialize_time_vlc bfs_launcher_novlc bfs_launcher_vlc launcher_single launcher_novlc launcher_vlc *.so
