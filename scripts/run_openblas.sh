#!/bin/bash
cd "$(dirname $(dirname "$0"))"

EXAMPLE_FOLDER=examples/openBLAS
# use when glibc < 2.34
# PATCH_PATH=location_of_vlc/pthread_patch.so 
# LIB_PATH=location_of_vlc/:$EXAMPLE_FOLDER/
PATCH_PATH=""
LIB_PATH=$EXAMPLE_FOLDER/

cd $EXAMPLE_FOLDER
make $1
time LD_PRELOAD=$PATCH_PATH LD_LIBRARY_PATH=./ ./${1}
