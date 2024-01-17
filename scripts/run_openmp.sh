cd /home/yyan/vlc/examples/openmp/
make launcher
LD_PRELOAD=/home/yyan/vlc/pthread_patch.so LD_LIBRARY_PATH=/home/yyan/vlc/:/home/yyan/vlc/examples/openmp/ /home/yyan/vlc/examples/openmp/launcher
rm address.tmp