CXX=g++
NO_DEBUG=-DNDEBUG -O2# -g
CFLAGS=-I./include -I../../include -std=c++17 -Wall $(NO_DEBUG)

libaddmp.so: src/addmp.cpp
	$(CXX) $(CFLAGS) -c -o addmp.o $< -pthread -fopenmp -fPIC
	$(CXX) $(CFLAGS) -shared -o $@ addmp.o -pthread -fopenmp

libsummp.so: src/summp.cpp
	$(CXX) $(CFLAGS) -c -o summp.o $< -pthread -fopenmp -fPIC
	$(CXX) $(CFLAGS) -shared -o $@ summp.o -pthread -fopenmp

libpowermp.so: src/powermp.cpp
	$(CXX) $(CFLAGS) -c -o powermp.o $< -pthread -fopenmp -fPIC
	$(CXX) $(CFLAGS) -shared -o $@ powermp.o -pthread -fopenmp

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

# benchmarks
is:
	cd benchmarks/IS; make is
	mv benchmarks/IS/is .

is.so:
	cd benchmarks/IS; make is.so
	mv benchmarks/IS/is.so .

is_transparent:    
	cd benchmarks/IS; make is_transparent

dc:
	cd benchmarks/DC; make dc
	mv benchmarks/DC/dc .

dc.so:
	cd benchmarks/DC; make dc.so
	mv benchmarks/DC/dc.so .

bfs.so:
	cd benchmarks/bfs; make bfs.so
	mv benchmarks/bfs/bfs.so .

kmeans.so:
	cd benchmarks/kmeans; make kmeans.so
	mv benchmarks/kmeans/kmeans.so .

hotspot3D.so:
	cd benchmarks/hotspot3D; make hotspot3D.so
	mv benchmarks/hotspot3D/hotspot3D.so .

cfd.so:
	cd benchmarks/cfd; make cfd.so
	mv benchmarks/cfd/cfd.so .

dc_transparent:    
	cd benchmarks/DC; make dc_transparent

parallel: benchmarks/parallel.cpp kmeans.so hotspot3D.so cfd.so
	$(CXX) $(CFLAGS) -o $@ $<

parallel_vlc: benchmarks/parallel_vlc.cpp kmeans.so hotspot3D.so cfd.so ../../include/VLC/tuning.h
	$(CXX) $(CFLAGS) -o $@ $<

arraybench:
	cd benchmarks/EPCC; make IDA=59049 arraybench
	mv benchmarks/EPCC/arraybench .

arraybench_transparent:
	cd benchmarks/EPCC; make IDA=59049 arraybench_transparent
	mv benchmarks/EPCC/arraybench_transparent .

arraybench.so:
	cd benchmarks/EPCC; make IDA=59049 arraybench.so
	mv benchmarks/EPCC/arraybench.so .

clean:
	rm -f address.* *.init.c *.tramp.S *.ver vlc_callback.c vlc_hashmap.c vlc_hashmap.h *.o *.so ADC.view.* ADC.logf.* ADC.dat.*
	rm -f arraybench arraybench_transparent is dc parallel launcher_vlc launcher_vlc_transparency launcher_novlc launcher_multiple *.so initialize_time_vlc
	rm -f parallel_vlc parallel *.out
	cd benchmarks/IS; make clean
	cd benchmarks/DC; make clean
	cd benchmarks/EPCC; make clean
	cd benchmarks/kmeans; make clean
	cd benchmarks/bfs; make clean
	cd benchmarks/hotspot3D; make clean
	cd benchmarks/cfd; make clean