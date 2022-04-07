#include <cstdio>
#include <cuda_runtime.h>

__host__ __device__ void greet(const char *info) {
  // __cuda_arch__ gtx1080 sm61; rtx2080 sm75; ga100 sm80

#ifdef __CUDA_ARCH__
  printf(info);
  printf("__cuda_arch__: %d\n", __CUDA_ARCH__);
  printf("Run from GPU.\n");
#else
  printf(info);
  printf("Run from CPU\n");
#endif
}

__global__ void kernel() {
  // main() on GPU
  printf("hello, world\n");
  greet("hello, again!\n");
}

// __host__ can call __global__
// __global__ can call __device__
// __device__ can call __device__

int main() {
  kernel<<<1, 3>>>();

  cudaError_t cuda_error = cudaDeviceSynchronize();
  if (cuda_error != cudaSuccess) {
    std::printf(cudaGetErrorString(cuda_error));
  }

  greet("cpu_main_info\n");

  return 0;
}

