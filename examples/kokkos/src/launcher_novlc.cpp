#include <iostream>
#include <vector>
#include <numeric>

#include "kokkos_compute.h"

int main() {
    int dev_id = 0;

    kokkos_init(dev_id);

    int n = 100000;

    std::vector<double> v(n);
    std::iota(v.begin(), v.end(), 1.0);

    double result = kokkos_reduction_copy_to_device(v.data(), n, dev_id);
    std::cout << "kokkos: result=" << result << std::endl;

    kokkos_finalize();
}
