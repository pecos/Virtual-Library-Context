# C compiler
CC = gcc
CC_FLAGS = -g -fopenmp -O2 

kmeans: cluster.o getopt.o kmeans.o kmeans_clustering.o 
	$(CC) $(CC_FLAGS) cluster.o getopt.o kmeans.o kmeans_clustering.o  -o kmeans

kmeans.so: cluster.o getopt.o kmeans.o kmeans_clustering.o 
	$(CC) $(CC_FLAGS) cluster.o getopt.o kmeans.o kmeans_clustering.o -shared -o kmeans.so

%.o: %.[ch]
	$(CC) $(CC_FLAGS) $< -c -fPIC

cluster.o: cluster.c 
	$(CC) $(CC_FLAGS) cluster.c -c -fPIC
	
getopt.o: getopt.c 
	$(CC) $(CC_FLAGS) getopt.c -c -fPIC
	
kmeans.o: kmeans.c 
	$(CC) $(CC_FLAGS) kmeans.c -c -fPIC

kmeans_clustering.o: kmeans_clustering.c kmeans.h
	$(CC) $(CC_FLAGS) kmeans_clustering.c -c -fPIC

clean:
	rm -f *.o *~ kmeans 
