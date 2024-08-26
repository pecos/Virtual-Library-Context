#!/bin/bash
cd $(dirname "$0")

LIB_FILE="/usr/local/lib/libarpack.so.2"
SYMBOL_FILE="symbols.txt"
NUM_VLC=2
LIB_NAME=$(basename $LIB_FILE)        

rm ${LIB_NAME}
python3 ../../lib/Implib.so/implib-gen.py --no-lazy-load --verbose --symbol-list=$SYMBOL_FILE $LIB_FILE --dlopen-callback=dlopen_callback --vlc=${NUM_VLC}
gcc -DIMPLIB_EXPORT_SHIMS -fPIC -shared -Wl,--version-script=${LIB_NAME}.ver ${LIB_NAME}.tramp.S ${LIB_NAME}.init.c vlc_hashmap.c vlc_callback.c -o ${LIB_NAME} -ldl

rm -f address.* eign_launcher_vlc_transparency *.init.c *.tramp.S *.ver vlc_callback.c vlc_hashmap.c vlc_hashmap.h

make eign_launcher_vlc_transparency
time LD_LIBRARY_PATH=. ./eign_launcher_vlc_transparency

rm -f address.*