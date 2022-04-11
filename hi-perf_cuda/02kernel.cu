#include <cuda_runtime.h>
#include <cstdio>
#include <memory>

#include "helper_cuda.h"

#define Accesss_gpu_addr

__global__ void kernel(int* pret) {
  *pret = 0x2028;
}

int main(int argc, char const* argv[]) {
#ifndef Accesss_gpu_addr
  int ret = 0;
  int* hret = static_cast<int*>(malloc(sizeof(int)));

  kernel<<<1, 1>>>(&ret);
  kernel<<<1, 1>>>(hret);
  cudaError_t err = cudaDeviceSynchronize();
  printf("%d\n", ret);
  printf("%s : %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
  // an illegal memory access was encountered

  checkCudaErrors(cudaDeviceSynchronize());
  free(hret);
#else
  int* pret;
  checkCudaErrors(cudaMalloc(&pret, sizeof(int)));
  kernel<<<1, 1>>>(pret);
  checkCudaErrors(cudaDeviceSynchronize());
  // printf("ret_on_gpu: %d\n", *pret);

  int pdst;
  checkCudaErrors(cudaMemcpy(&pdst, pret, sizeof(int), cudaMemcpyDeviceToHost));
  // has once cudaDeviceSynchronize()

  // cudaMemcpyKind
  printf("dst_on_cpu: %d\n", pdst);

  cudaFree(pret);
#endif

  return 0;
}
