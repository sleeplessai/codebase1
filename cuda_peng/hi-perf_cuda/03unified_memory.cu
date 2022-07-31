#include <cuda_runtime.h>
#include <cstdio>
#include "helper_cuda.h"

// #define SEGMENT_FAULT_DEMO

__global__ void kernel(int* pret) {
  *pret = 0x978;
}

int main(int argc, char const* argv[]) {
  int* pret;
  checkCudaErrors(cudaMallocManaged(&pret, sizeof(int)));
  // no need to copy manually
  kernel<<<1, 1>>>(pret);
  checkCudaErrors(cudaDeviceSynchronize());
  printf("result %d @ %x\n", *pret, pret);

  cudaFree(pret);

#ifdef SEGMENT_FAULT_DEMO
  int cpu_val = *pret;
  printf("result %d @ %x\n", cpu_val, &cpu_val);
#endif

  return 0;
}
