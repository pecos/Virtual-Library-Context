ONEDNN_INCDIR=-I/usr/local/include/oneapi/dnnl/
ONEDNN_LIBDIR=-L/usr/local/lib/

CXX=g++
# -O1 and other optimization flag will break oneDNN + VLCs transparent mode
NO_DEBUG=-DNDEBUG # -g
CFLAGS=-I../../include $(ONEDNN_INCDIR) $(ONEDNN_LIBDIR) -std=c++17 -Wall $(NO_DEBUG)

# benchmark
cnn_inference_f32: benchmarks/cnn_inference_f32.c benchmarks/example_utils.h
	$(CXX) $(CFLAGS) -o $@ $< -ldnnl

cpu_cnn_training_f32: benchmarks/cpu_cnn_training_f32.c benchmarks/example_utils.h
	$(CXX) $(CFLAGS) -o $@ $< -ldnnl

cpu_matmul_csr: benchmarks/cpu_matmul_csr.c benchmarks/example_utils.h
	$(CXX) $(CFLAGS) -o $@ $< -ldnnl
	
cnn_inference_f32_transparent: benchmarks/cnn_inference_f32_transparent.cpp benchmarks/example_utils.h ../../include/VLC/runtime.h
	$(CXX) $(CFLAGS) -o $@ $< -ldnnl

cpu_cnn_training_f32_transparent: benchmarks/cpu_cnn_training_f32_transparent.cpp benchmarks/example_utils.h ../../include/VLC/runtime.h
	$(CXX) $(CFLAGS) -o $@ $< -ldnnl

cpu_matmul_csr_transparent: benchmarks/cpu_matmul_csr_transparent.cpp benchmarks/example_utils.h ../../include/VLC/runtime.h
	$(CXX) $(CFLAGS) -o $@ $< -ldnnl

clean:
	rm -f cpu_matmul_csr_transparent cpu_matmul_csr cpu_cnn_training_f32 cnn_inference_f32_transparent cpu_cnn_training_f32_transparent cnn_inference_f32 cnn_inference_f32_transparent *.so *.so.* *.S *.ver *.init.c vlc_callback.c vlc_hashmap.c vlc_hashmap.h