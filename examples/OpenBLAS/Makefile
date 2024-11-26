OPENBLAS_VERSION=openblas-pthread#openblas-openmp
OPENBLAS64_VERSION=openblas64-pthread#openblas64-openmp
OPENBLAS64_INCDIR=-I/usr/include/x86_64-linux-gnu/$(OPENBLAS64_VERSION)/
OPENBLAS64_LIBDIR=-L/usr/lib/x86_64-linux-gnu/$(OPENBLAS64_VERSION)/
OPENBLAS_INCDIR=-I/usr/include/x86_64-linux-gnu/$(OPENBLAS_VERSION)/
OPENBLAS_LIBDIR=-L/usr/lib/x86_64-linux-gnu/$(OPENBLAS_VERSION)/

CXX=g++
NO_DEBUG=-DNDEBUG -O2 # -g
CFLAGS=-I./include -I../../include $(OPENBLAS_INCDIR) $(OPENBLAS_LIBDIR) $(OPENBLAS64_INCDIR) $(OPENBLAS64_LIBDIR) -std=c++17 -Wall $(NO_DEBUG)

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

# micro benchmark
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

# benchmark
cholesky: benchmarks/cholesky.cpp
	$(CXX) $(CFLAGS) -o $@ $^ -lopenblas
	
gemm: benchmarks/gemm.cpp
	$(CXX) $(CFLAGS) -o $@ $^ -lopenblas

linear_equation: benchmarks/linear_equation.cpp
	$(CXX) $(CFLAGS) -o $@ $^ -lopenblas

cholesky.so: benchmarks/cholesky.cpp
	$(CXX) $(CFLAGS) -shared -o $@ $^ -lopenblas -fPIC
	
gemm.so: benchmarks/gemm.cpp
	$(CXX) $(CFLAGS) -shared -o $@ $^ -lopenblas -fPIC

linear_equation.so: benchmarks/linear_equation.cpp
	$(CXX) $(CFLAGS) -shared -o $@ $^ -lopenblas -fPIC

parallel: benchmarks/parallel.cpp cholesky.so gemm.so linear_equation.so
	$(CXX) $(CFLAGS) -o $@ $<

gemm_vlc_transparent: benchmarks/gemm_vlc_transparent.cpp  ../../include/VLC/runtime.h
	$(CXX) $(CFLAGS) -o $@ $< -lopenblas

cholesky_vlc_transparent: benchmarks/cholesky_vlc_transparent.cpp ../../include/VLC/runtime.h
	$(CXX) $(CFLAGS) -o $@ $< -lopenblas

linear_equation_vlc_transparent: benchmarks/linear_equation_vlc_transparent.cpp  ../../include/VLC/runtime.h
	$(CXX) $(CFLAGS) -o $@ $< -lopenblas

parallel_vlc: benchmarks/parallel_vlc.cpp  ../../include/VLC/runtime.h cholesky.so gemm.so linear_equation.so
	$(CXX) $(CFLAGS) -o $@ $<

clean:
	rm -f *.so parallel parallel_vlc dgemm cholesky gemm linear_equation gemm_vlc_transparent linear_equation_vlc_transparent cholesky_vlc_transparent initialize_time_base initialize_time_vlc with_openmp_vlc openblas_alone openmp_alone with_openmp_seq with_openmp_parallel matmul_launcher_seq matmul_launcher_vlc matmul_launcher_multithread matmul_launcher_vlc_transparency *.so.0 *.S *.ver *.init.c vlc_callback.c vlc_hashmap.c vlc_hashmap.h