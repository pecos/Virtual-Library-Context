CXX=g++
NO_DEBUG=-DNDEBUG # -g

CFLAGS=-I../../include -std=c++17 -Wall $(NO_DEBUG)

overhead: overhead.cpp  ../../include/VLC/runtime.h
	$(CXX) $(CFLAGS) -o $@ $<

libcudaoverhead.so: cuda_shim_overhead/cuda_overhead.cu
	nvcc -arch=sm_60 --cudart=shared --compiler-options '-fPIC' $< -shared -o $@

cuda_overhead_vlc: cuda_shim_overhead/vlc.cu libcudaoverhead.so ../../include/VLC/runtime.h
	nvcc -arch=sm_60 --cudart=shared --compiler-options '$(CFLAGS)' $< -o $@

cuda_overhead_base: cuda_shim_overhead/base.cu cuda_shim_overhead/cuda_overhead.h libcudaoverhead.so
	nvcc -arch=sm_60 --compiler-options '$(CFLAGS)' $< -I./cuda_shim_overhead -L./ -lcudaoverhead -o $@

libhello.so: helloworld_stub_overhead/hello.cpp
	$(CXX) $(CFLAGS) -I./helloworld_stub_overhead -shared -fPIC  $< -o $@

helloworld_overhead_vlc: helloworld_stub_overhead/vlc.cpp libhello.so
	$(CXX) $(CFLAGS) -I./helloworld_stub_overhead -L./ $< -lhello -o $@ 

helloworld_overhead_base: helloworld_stub_overhead/base.cpp libhello.so
	$(CXX) $(CFLAGS) -I./helloworld_stub_overhead -L./ $< -lhello -o $@ 

clean:
	rm *.so matmul_launcher