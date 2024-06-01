#include <Kokkos_Core.hpp>
#include <stdio.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

#include "kokkos_compute.h"

typedef Kokkos::DefaultExecutionSpace DeviceSpace;

void kokkos_init(int dev) {
	Kokkos::initialize(Kokkos::InitializationSettings()  // (since 3.7)
                       .set_disable_warnings(false)
					   .set_device_id(dev));
}

void kokkos_finalize() {
	Kokkos::finalize();
}

double kokkos_reduction_copy_to_device(double* array, const int N, const int dev_id) {

	using h_view = typename Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

	//Wrap raw pointer in Kokkos View for easy management. 	     
	h_view host_array(array, N);

	//Setting range policy and doing explicit copies isn't necessary. The above should work, but this is a safety
	cudaSetDevice(dev_id);
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	Kokkos::Cuda cuda1(stream);
	auto range_policy = Kokkos::RangePolicy<Kokkos::Cuda>(cuda1, 0, N);

	//Explicit copies because the default execution space isn't being set correctly
	using d_view = typename Kokkos::View<double*, Kokkos::CudaSpace>;
	d_view device_array("device_array", N);
	Kokkos::deep_copy(device_array, host_array);

	double sum = 0.0;
	{
		Kokkos::parallel_reduce("Reduction", N, KOKKOS_LAMBDA(const int i, double& lsum){
			lsum += device_array(i);
		}, sum);

		Kokkos::fence();
	}
	
	cudaStreamDestroy(stream);

	return sum;
};
