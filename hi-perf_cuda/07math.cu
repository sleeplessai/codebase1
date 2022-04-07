#include "CudaAllocator.h"
#include "helper_cuda.h"
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

// template <typename Func>
template <typename Func> __global__ void parallel_for(int n, Func func) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    func(i);
  }
}

int main(int argc, char const *argv[]) {
  int n = 3840;
  if (argc == 2) {
    n = std::atoi(argv[1]);
  }

  int n_threads = 128, n_blocks = (n + n_threads - 1) / n;
  std::vector<float, CudaAllocator<float>> vals(n);

  parallel_for<<<n_blocks, n_threads>>>(n,
                                        [vals = vals.data()] __device__(int i) {
                                          // vals[i] = sinf(i);
                                          vals[i] = __sinf(i);
                                        });
  // gpu_buildin __sinf: faster but lower precision
  // other faster gpu__buildins: __expf, __logf, __cost, __powf
  // enable --use_fast_math, replace all

  checkCudaErrors(cudaDeviceSynchronize());

  for (int i = 0; i < n; ++i) {
    std::printf("diff %d = %f\n", i, vals[i] - std::sin(i));
  }

  return 0;
}
