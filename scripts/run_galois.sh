#!/bin/bash
cd "$(dirname $(dirname "$0"))"

EXAMPLE_FOLDER=examples/galois
GALOIS_INSTALL_FOLDER=examples/galois/lib/Galois/build/libgalois
# use when glibc < 2.34
# PATCH_PATH=/home/yyan/vlc/pthread_patch.so 
# LIB_PATH=/home/yyan/vlc/:$EXAMPLE_FOLDER/
PATCH_PATH=""
LIB_PATH=$EXAMPLE_FOLDER/:$GALOIS_INSTALL_FOLDER/

make -C $EXAMPLE_FOLDER $1
time LD_PRELOAD=$PATCH_PATH LD_LIBRARY_PATH=$LIB_PATH $EXAMPLE_FOLDER/${1}
