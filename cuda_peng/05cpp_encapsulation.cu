#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "helper_cuda.h"

#include "CudaAllocator.h"

// gpu main
template <typename T>
__global__ void kernel(T* arr, size_t n) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    arr[i] = i;
  }
}

int main(int argc, char const* argv[]) {
  size_t n = 4096;
  std::vector<int, CudaAllocator<int>> v(n);
  printf("%u\n", v.capacity());
  v.reserve(n >> 1);
  printf("%u\n", v.capacity());
  int n_threads = 128, n_blocks = ((n >> 1) + n_threads - 1) / n_threads;

  kernel<int><<<n_blocks, n_threads>>>(v.data(), n);

  return 0;
}
