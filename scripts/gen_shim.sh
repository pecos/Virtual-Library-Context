#!/bin/bash
LIB_FILE=$1
SYMBOL_FILE=$2
VLC_CALLBACK_FILE=$3
LIB_NAME=$(basename $LIB_FILE)

rm ${LIB_NAME}
python3 lib/Implib.so/implib-gen.py --no-lazy-load --verbose --symbol-list=$SYMBOL_FILE $LIB_FILE --dlopen-callback=dlopen_callback
gcc -DIMPLIB_EXPORT_SHIMS -fPIC -shared -Wl,--version-script=${LIB_NAME}.ver ${LIB_NAME}.tramp.S ${LIB_NAME}.init.c $VLC_CALLBACK_FILE -o ${LIB_NAME} -ldl