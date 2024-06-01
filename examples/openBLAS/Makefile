CXX=g++
NO_DEBUG=-DNDEBUG # -g
OPEN_BLAS_VERSION=openblas64-openmp#openblas64-pthread

CFLAGS=-I./include -I../../include -I/usr/include/x86_64-linux-gnu/$(OPEN_BLAS_VERSION)/ -L/usr/lib/x86_64-linux-gnu/$(OPEN_BLAS_VERSION)/ -std=c++17 -Wall $(NO_DEBUG)

libmatmul.so: src/matmul.cpp
	$(CXX) $(CFLAGS) -shared -o $@ $< -lopenblas64 -fPIC

matmul_launcher: src/matmul_launcher.cpp libmatmul.so ../../include/VLC/runtime.h
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
	rm *.so matmul_launcher