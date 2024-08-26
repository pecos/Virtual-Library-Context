#!/bin/bash
cd "$(dirname $(dirname "$0"))"

EXAMPLE_FOLDER=/home/yyan/vlc/examples/arpack
# use when glibc < 2.34
# PATCH_PATH=/home/yyan/vlc/pthread_patch.so 
# LIB_PATH=/home/yyan/vlc/:$EXAMPLE_FOLDER/
PATCH_PATH=""
LIB_PATH=$EXAMPLE_FOLDER/

cd $EXAMPLE_FOLDER
make $1
time LD_PRELOAD=$PATCH_PATH LD_LIBRARY_PATH=./ ./${1}