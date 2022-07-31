#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "CudaAllocator.h"
#include "helper_cuda.h"

template <typename T>
__global__ void matrix_transpose(T* out, const T* in, int nx, int ny) {
  int linearized = blockIdx.x * blockDim.x + threadIdx.x;
  int y = linearized / nx;
  int x = linearized % nx;
  if (x >= nx || y >= ny)
    return;
  out[y * nx + x] = in[x * nx + y];
}

int main() {
  int nx = 1 << 14, ny = 1 << 14;
  std::vector<int, CudaAllocator<int>> in(nx * ny);
  std::vector<int, CudaAllocator<int>> out(nx * ny);

  for (int i = 0; i < nx * ny; i++) {
    in[i] = i;
  }

  // TICK(parallel_transpose);
  matrix_transpose<int><<<nx * ny / 1024, 1024>>>(out.data(), in.data(), nx, ny);
  checkCudaErrors(cudaDeviceSynchronize());
  // TOCK(parallel_transpose);

  for (int y = 0; y < ny; y++) {
    for (int x = 0; x < nx; x++) {
      if (out[y * nx + x] != in[x * nx + y]) {
        std::printf("Wrong At x=%d,y=%d: %d != %d\n", x, y, out[y * nx + x], in[x * nx + y]);
        return -1;
      }
    }
  }

  std::printf("All Correct!\n");
  return 0;
}
