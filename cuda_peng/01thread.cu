#include <cuda_runtime.h>
#include <cstdio>

__global__ void kernel() {
  unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int tnb = blockDim.x * gridDim.x;

  printf("Flattened thread %d of %d\n", tid, tnb);

  printf("Block %d of %d, Thread %d of %d\n", blockIdx.x, gridDim.x, threadIdx.x, blockDim.x);
}

int main(int argc, char const* argv[]) {
  // kernel<<<2, 4>>>();
  kernel<<<dim3(2, 1, 1), dim3(2, 3, 1)>>>();
  // <<<BlockNum, ThreadPerBlcok>>>
  // <<<gridDim, blockDim>>>
  cudaDeviceSynchronize();
  return 0;
}
