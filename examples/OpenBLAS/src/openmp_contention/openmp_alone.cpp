#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <chrono>


double f2(double *X, double *Y, int size){
    double v = 0.0;

    #pragma omp parallel for reduction(+:v)
    for(int i = 0; i < size; i++){
        v += X[i];
        v += Y[i];
    }
    return v;
}


int main(int argc, char **argv){
    int size = 10000,
        num_itr = 100;

    double *A = (double*) malloc(size * size * sizeof(double)),
           *B = (double*) malloc(size * size * sizeof(double));

    for(int i=0; i<size*size; i++){
        A[i] = 0.1;
    }
    for(int i=0; i<size*size; i++){
        B[i] = 1.2;
    }

    double v = 0.0;

    // run openMP
    auto t1 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<num_itr; i++){
        // Followed by parallel loop
        v += f2(A, B, size * size);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "finish in " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;

    std::cout << "result = " << v << std::endl;

    free(A), free(B);

    return 0;
}