./scripts/gen_shim.sh /usr/local/cuda-11.7/lib64/libcudart.so.11.0 cudart_shim/symbols.txt cudart_shim/vlc_callback.c
mv libcudart.so.11.0 cudart_shim/