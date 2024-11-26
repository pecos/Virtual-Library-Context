OMP_NUM_THREADS = 24

all: euler3d_cpu

euler3d_cpu: euler3d_cpu.cpp
	g++ -O3 -Dblock_length=$(OMP_NUM_THREADS) -fopenmp euler3d_cpu.cpp -o euler3d_cpu

cfd.so: euler3d_cpu.cpp
	g++ -O3 -Dblock_length=$(OMP_NUM_THREADS) -fopenmp euler3d_cpu.cpp -shared -o cfd.so -fPIC


clean:
	rm -f cfd.so euler3d_cpu
