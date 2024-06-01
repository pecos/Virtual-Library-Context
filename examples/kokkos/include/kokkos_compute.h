#ifndef KOKKOS_COMPUTE_H
#define KOKKOS_COMPUTE_H

void kokkos_init(int dev);

void kokkos_finalize();

double kokkos_reduction_copy_to_device(double* array, const int N, const int dev_id);

#endif