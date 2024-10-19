#ifndef HEAT3D_H
#define HEAT3D_H

void kokkos_initialize(int dev);
void kokkos_finalize();
void *initialize_system(int me, int nranks, int argc, char* argv[]);
void finalize_system(void *sys);
int get_N(void *sys);
int get_I(void *sys);
void heat3d_phase1(void *sys, int t);
void heat3d_exchange_data(void *this_sys, void *other_sys);
double heat3d_phase2(void *sys);

#endif