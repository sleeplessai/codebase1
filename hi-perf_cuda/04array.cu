#include "helper_cuda.h"
#include <cstdio>
#include <cuda_runtime.h>

#define Leftovers_problem_solving
#define Grid_stride_loop

__global__ void kernel(int *arr, int n) {
#ifdef Leftovers_problem_solving

#ifdef Grid_stride_loop
  // Grid stride loop
  // for (int i = threadIdx.x; i < n; i += blockDim.x) {
  //     arr[i] = i;
  // }
  // Grid tride loop with gridDim
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    arr[i] = i;
    // blockDim: the number of threads
    // gridDim: the number of grids (thread pool)
  }
#else
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n)
    return;
  arr[i] = i;
#endif

#else
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  arr[i] = i;
#endif
}

int main(int argc, char const *argv[]) {

  int n = 1020;
  int *arr;
  checkCudaErrors(cudaMallocManaged(&arr, sizeof(int) * n));

  int n_threads = 256;
  int n_blocks = (n + n_threads - 1) / n_threads;
  // istead of n / n_threads

  kernel<<<n_blocks, n_threads>>>(arr, n);
  cudaDeviceSynchronize();

  for (int i = 0; i < n; ++i) {
    std::printf("%d ", arr[i]);
  }
  std::putchar('\n');

  return 0;
}
// 59: 18
