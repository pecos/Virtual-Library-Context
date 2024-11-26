CC = gcc
CCFLAGS = -g -fopenmp -O3
OUTPUT = *.out

3D:
	$(CC) -o 3D $(CCFLAGS) 3D.c -lm

hotspot3D.so: 3D.c
	$(CC) -shared -o hotspot3D.so $(CCFLAGS) 3D.c -lm -fPIC

clean:
	rm -f 3D hotspot3D.so $(OUTPUT)
