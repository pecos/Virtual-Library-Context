#include <stdio.h>

#include "kokkos_compute.h"

int main() {
    printf("%p\n", kokkos_init);
    return 0;
}