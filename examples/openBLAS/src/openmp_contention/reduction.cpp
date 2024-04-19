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

void reduction(int num_itr, int size, double *A, double *B, double *v) {
    double ret = 0.0;
    for(int i=0; i<num_itr; i++){
        ret += f2(A, B, size * size);
    }
    *v = ret;
}