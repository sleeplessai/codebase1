#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include "CudaAllocator.h"
#include "helper_cuda.h"

const int n = 1 << 20, n_threads = 1 << 10;

__global__ void parallel_sum_shared(int64_t* arr, int64_t* sum, int n) {
  // Block-local storage (BLS) compiler optimization

  __shared__ volatile int64_t local_sum[n_threads];
  // __shared__ memory depends on cuda_arch. set volatile to prevent compiler
  // optimization.
  // https://stackoverflow.com/questions/4437527/why-do-we-use-volatile-keyword

  int j = threadIdx.x, i = blockIdx.x;
  // local_sum[j] = arr[i * n_threads + j];

  int64_t temp_sum = 0;
  for (int t = i * n_threads + j; t < n; t += n_threads * gridDim.x) {
    temp_sum += arr[t];
  }
  local_sum[j] = temp_sum;
  __syncthreads();
  if (j < n_threads / 2) {
    local_sum[j] += local_sum[j + n_threads / 2];
  }
  __syncthreads();
  if (j < n_threads / 4) {
    local_sum[j] += local_sum[j + n_threads / 4];
  }
  __syncthreads();
  if (j < n_threads / 8) {
    local_sum[j] += local_sum[j + n_threads / 8];
  }
  __syncthreads();
  if (j < n_threads / 16) {
    local_sum[j] += local_sum[j + n_threads / 16];
  }
  __syncthreads();
  // Warp (32 threads as a group) SM dispatches a warp once
  // Let warp in a single If block as possible (warp divergence)
  // So, we merge when j < 32 if branch

  if (j < n_threads / 32) {
    local_sum[j] += local_sum[j + n_threads / 32];
    local_sum[j] += local_sum[j + n_threads / 64];
    local_sum[j] += local_sum[j + n_threads / 128];
    local_sum[j] += local_sum[j + n_threads / 256];
    local_sum[j] += local_sum[j + n_threads / 512];
    if (j == 0) {
      sum[i] = local_sum[0] + local_sum[1];
    }
  }
}

int main(int argc, char const* argv[]) {
  // Streaming Multiprocessors (SM) process threads
  // on multiple blocks with shared memory space.

  std::vector<int64_t, CudaAllocator<int64_t>> arr(n);
  std::vector<int64_t, CudaAllocator<int64_t>> sum(n / n_threads);

  const int64_t max_int = 592ll;
  int64_t sum_cpu = 0;
  for (int i = 0; i < n; ++i) {
    arr[i] = std::rand() % max_int;
    sum_cpu += arr[i];
  }

  // blockDim.x should be an integer multiple of 3
  // parallel_sum_shared<<<n / n_threads, n_threads>>>(arr.data(), sum.data(),
  // n);
  parallel_sum_shared<<<n / (4 * n_threads), n_threads>>>(arr.data(), sum.data(), n);
  checkCudaErrors(cudaDeviceSynchronize());

  // int64_t sum_res = std::accumulate(sum.begin(), sum.end(), 0);
  int64_t sum_res = 0;
  std::for_each(sum.begin(), sum.end(), [&sum_res](const int& x) { sum_res += x; });

  std::cout << "sum_on_cpu = " << sum_cpu << std::endl;
  std::cout << "sum_on_gpu = " << sum_res << std::endl;

  return 0;
}
