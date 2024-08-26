CXX=g++
NO_DEBUG=-DNDEBUG # -g
OPEN_BLAS_VERSION=openblas64-pthread#openblas64-openmp

CFLAGS=-I./include -I../../include -I/usr/include/x86_64-linux-gnu/$(OPEN_BLAS_VERSION)/ -L/usr/lib/x86_64-linux-gnu/$(OPEN_BLAS_VERSION)/ -std=c++17 -Wall $(NO_DEBUG)

libmatmul.so: src/matmul.cpp
	$(CXX) $(CFLAGS) -shared -o $@ $< -lopenblas64 -fPIC

matmul_launcher_vlc_transparency: src/matmul_launcher_vlc_transparency.cpp ../../include/VLC/runtime.h
	$(CXX) $(CFLAGS) -o $@ $< -lopenblas64

matmul_launcher_vlc: src/matmul_launcher_vlc.cpp libmatmul.so ../../include/VLC/runtime.h
	$(CXX) $(CFLAGS) -o $@ $<

matmul_launcher_seq: src/matmul_launcher_seq.cpp src/matmul.cpp include/matmul.h
	$(CXX) $(CFLAGS) -o $@ $^ -lopenblas64

matmul_launcher_multithread: src/matmul_launcher_multithread.cpp src/matmul.cpp include/matmul.h
	$(CXX) $(CFLAGS) -o $@ $^ -lopenblas64

with_openmp_seq: src/openmp_contention/with_openmp_seq.cpp
	$(CXX) $(CFLAGS) -o $@ $^ -lopenblas64 -fopenmp

with_openmp_parallel: src/openmp_contention/with_openmp_parallel.cpp
	$(CXX) $(CFLAGS) -o $@ $^ -lopenblas64 -fopenmp

openmp_alone: src/openmp_contention/openmp_alone.cpp
	$(CXX) $(CFLAGS) -o $@ $^ -fopenmp

openblas_alone: src/openmp_contention/openblas_alone.cpp
	$(CXX) $(CFLAGS) -o $@ $^ -lopenblas64

libreduction.so: src/openmp_contention/reduction.cpp
	$(CXX) $(CFLAGS) -shared -fPIC -o $@ $^ -fopenmp

libdgemm.so: src/openmp_contention/dgemm.cpp
	$(CXX) $(CFLAGS) -shared -fPIC -o $@ $^ -lopenblas64

with_openmp_vlc: src/openmp_contention/with_openmp_vlc.cpp  libreduction.so libdgemm.so ../../include/VLC/runtime.h
	$(CXX) $(CFLAGS) -o $@ $<

initialize_time_vlc: src/initialize_time/vlc.cpp  libmatmul.so ../../include/VLC/runtime.h
	$(CXX) $(CFLAGS) -o $@ $<

initialize_time_base: src/initialize_time/base.cpp  libmatmul.so
	$(CXX) $(CFLAGS) -o $@ $< -L./ -lmatmul

clean:
	rm -f *.so initialize_time_base initialize_time_vlc with_openmp_vlc openblas_alone openmp_alone with_openmp_seq with_openmp_parallel matmul_launcher_seq matmul_launcher_vlc matmul_launcher_multithread matmul_launcher_vlc_transparency *.so.0