#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <cblas.h>
#include <iostream>
#include <chrono>


double f2(double *X, double *Y, int size){
    double v = 0.0;

    #pragma omp parallel for reduction(+:v)
    for(int i=0; i<size; i++){
        v += X[i];
        v += Y[i];
    }

    return v;
}

void launch_openBLAS(int num_itr, int size, double *A, double *B, double *C) {
    for(int i=0; i<num_itr; i++){
        // BLAS call
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, size, size, size, 1.0, A, size, B, size, 0.0, C, size);    // C <- A @ B.T
    }

}

void launch_openMP(int num_itr, int size, double *A, double *B, double *v) {
    double ret = 0.0;
    for(int i=0; i<num_itr; i++){
        ret += f2(A, B, size * size);
    }
    *v = ret;
}


int main(int argc, char **argv){
    int size_mp = 10000,
        num_itr_mp = 100;

    double *A_mp = (double*) malloc(size_mp * size_mp * sizeof(double)),
           *B_mp = (double*) malloc(size_mp * size_mp * sizeof(double));

    for(int i=0; i<size_mp*size_mp; i++){
        A_mp[i] = 0.1;
    }
    for(int i=0; i<size_mp*size_mp; i++){
        B_mp[i] = 1.2;
    }


    int size_blas = 2000,
        num_itr_blas = 100;

    double *A_blas = (double*) malloc(size_blas * size_blas * sizeof(double)),
           *B_blas = (double*) malloc(size_blas * size_blas * sizeof(double)),
           *C_blas = (double*) malloc(size_blas * size_blas * sizeof(double));

    for(int i=0; i<size_blas*size_blas; i++){
        A_blas[i] = 0.1;
    }
    for(int i=0; i<size_blas*size_blas; i++){
        B_blas[i] = 1.2;
    }

    double v = 0.0;

    // run openMP and openBLAS in sequential
    
    auto t1 = std::chrono::high_resolution_clock::now();

    launch_openBLAS(num_itr_blas, size_blas, A_blas, B_blas, C_blas);
    launch_openMP(num_itr_mp, size_mp, A_mp, B_mp,  &v);

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "finish in " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" << std::endl;

    std::cout << "result = " << v << std::endl;

    free(A_blas), free(B_blas), free(C_blas), free(A_mp), free(B_mp);

    return 0;
}