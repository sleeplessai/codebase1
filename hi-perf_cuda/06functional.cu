#include "helper_cuda.h"
#include <cstdio>
#include <cuda_runtime.h>

#include "CudaAllocator.h"
#include <vector>

template <class Func> __global__ void grid_stride_loop(size_t n, Func func) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    func(i);
  }
}

struct MyFunctor {
  __device__ void operator()(int i) const { printf("number %d\n", i); }
};

int main(int argc, char const *argv[]) {

  int n = 256;
  // grid_stride_loop<<<32, 64>>>(n, MyFunctor{});

  // checkCudaErrors(cudaDeviceSynchronize());

  // add --extended-lambda to nvcc
  // target_compile_options(06functional PUBLIC
  // $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
  grid_stride_loop<<<16, 16>>>(
      n, [] __device__(int i) { printf("lambda0 info: number %d\n", i); });

  checkCudaErrors(cudaDeviceSynchronize());

  std::vector<int, CudaAllocator<int>> v(n);
  // int* v_data = v.data();
  // for shallow copying
  grid_stride_loop<<<16, 16>>>(n, [v = v.data()] __device__(int i) {
    v[i] = i;
    printf("lambda1 info: v_data[%d] = %d\n", i, i);
  });

  checkCudaErrors(cudaDeviceSynchronize());

  return 0;
}

