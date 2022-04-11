#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include "helper_cuda.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "CudaAllocator.h"

template <typename Func>
__global__ void ParallelFor(int n, Func func) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    func(i);
  }
}

int main(int argc, char const* argv[]) {
  int n = 2560;
  if (argc == 2) {
    n = std::atoi(argv[1]);
  }

  thrust::host_vector<float> x_host(n);
  thrust::host_vector<float> y_host(n);

  auto genf = [] { return std::rand() * (1.f / RAND_MAX); };
  thrust::generate(x_host.begin(), x_host.end(), genf);
  thrust::generate(y_host.begin(), y_host.end(), genf);

  for (int i = 0; i < n; ++i) {
    std::printf("%f %f\n", x_host[i], y_host[i]);
  }

  float pi = 3.1415f;
  // copy and auto call cudaMemcpy
  thrust::device_vector<float> x_dev = x_host;
  thrust::device_vector<float> y_dev = y_host;

  for (int i = 0; i < n; ++i) {
    std::printf("x: %f, y: %f\n", x_dev[i], y_dev[i]);
  }

  ParallelFor<<<20, 128>>>(n, [pi, x_dev = x_dev.data(), y_dev = y_dev.data()] __device__(int i) {
    x_dev[i] = pi * x_dev[i] + y_dev[i];
  });
  checkCudaErrors(cudaDeviceSynchronize());

  x_host = x_dev; // copy back

  for (int i = 0; i < n; ++i) {
    std::printf("x_dev[%d] = %f\n", i, x_host[i]);
  }

  // thrust template functions
  thrust::for_each(x_dev.begin(), x_dev.end(), [] __device__(float& x) { x += 10.f; });

  x_host = x_dev;
  for (int i = 0; i < n; ++i) {
    std::printf("x_dev[%d] = %f\n", i, x_host[i]);
  }

  thrust::for_each(
      y_dev.cbegin(), y_dev.cend(), [] __device__(float const& x) { printf("%f\n", x + 1.f); });
  // thrust::reduce, sort, find_if, count_if, reverse, inclusive_scan, etc
  thrust::for_each(
      thrust::make_counting_iterator(3),
      thrust::make_counting_iterator(8),
      [] __device__(int i) { std::printf("%d\n", i); }
      // 3 4 5 6 7
  );

  thrust::for_each(
      thrust::make_zip_iterator(x_dev.begin(), x_dev.end()),
      thrust::make_zip_iterator(y_dev.begin(), y_dev.end()),
      // thrust::make_zip_iterator(x_dev.cbegin(), x_dev.cend()),
      [pi] __device__(auto const& tup) {
        auto& x = thrust::get<0>(tup);
        auto& y = thrust::get<1>(tup);
        // auto& z = thrust::get<2>(tup);
        y = pi * y + x;
      });
  // c++17 structural binding

  return 0;
}
