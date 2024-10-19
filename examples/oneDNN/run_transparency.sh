#!/bin/bash
cd $(dirname "$0")

LIB_FILE="/usr/local/lib/libdnnl.so.3"
SYMBOL_FILE="symbols.txt"
NUM_VLC=1
LIB_NAME=$(basename $LIB_FILE)        

# make clean
rm -f address.*

python3 ../../lib/Implib.so/implib-gen.py --no-lazy-load --verbose --symbol-list=$SYMBOL_FILE $LIB_FILE --dlopen-callback=dlopen_callback --vlc=${NUM_VLC}
gcc -DIMPLIB_EXPORT_SHIMS -fPIC -shared -Wl,--version-script=${LIB_NAME}.ver ${LIB_NAME}.tramp.S ${LIB_NAME}.init.c vlc_hashmap.c vlc_callback.c -o ${LIB_NAME} -ldl

make $1
time LD_LIBRARY_PATH=. ./$1 ${@:2}