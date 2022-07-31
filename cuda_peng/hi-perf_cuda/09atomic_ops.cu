#include <cuda_runtime.h>
#include <cstdio>
#include "helper_cuda.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/random/uniform_int_distribution.h>

#include <iostream>
#include <numeric>
#include "CudaAllocator.h"

#define __Impl_with_thrust false
#define __AtomicCas_mul_impl true

#if __Impl_with_thrust

namespace thrust {
namespace random {
template <class T>
__host__ __device__ T __random(size_t seed, T a = 0.f, T b = 1.f) {
  thrust::default_random_engine rng;
  thrust::uniform_real_distribution<T> dist(a, b);
  rng.discard(seed);
  return dist(rng);
}
} // namespace random
} // namespace thrust

template <class T>
__global__ void parallel_for(thrust::device_ptr<T> p, size_t n, T a = 0.f, T b = 1.f) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    p[i] = thrust::random::__random<T>(0x1c * (n - i), a, b);
    p[n] = p[n] + p[i];
  }
}

int main(int argc, char const* argv[]) {
  int n = 2048;
  if (argc == 2) {
    n = std::atoi(argv[1]);
    // no ensure to be qualified
  }

  thrust::device_vector<float> v_dev(n + 1);
  v_dev.back() = static_cast<float>(0.f);
  parallel_for<<<32, 256>>>(v_dev.data(), n, -10.f, 10.f);
  /* __asm {
      mov ax, p[n]
      mov bx, p[i]
      add ax, bx
      mov p[n], ax
  }; */
  v_dev.back() = thrust::reduce(thrust::device, v_dev.begin(), v_dev.end() - 1);

  checkCudaErrors(cudaDeviceSynchronize());

  thrust::host_vector<float> v_host(n + 1);
  v_host = v_dev;
  float sum = 0.f;
  thrust::for_each(v_host.begin(), v_host.end() - 1, [&sum](float& x) {
    // std::printf("%f\n", x);
    sum += x;
  });
  std::printf("\nsum_on_gpu = %f\nsum_on_cpu = %f\n", v_host.back(), sum);

  return 0;
}

#else

template <typename T>
__global__ void parallel_sum(T* v, T* sum, const size_t n) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    v[i] = static_cast<T>(i / 2); // casually init
    atomicAdd(sum, v[i]);
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-functions
  }
}

#if __AtomicCas_mul_impl

constexpr float epsilon = 1e-8f;

__device__ float atomicCAS_f32(float* p, float cmp, float val) {
  // https://gist.github.com/PolarNick239/5e370892989800fe42154145911b141f
  return __int_as_float(atomicCAS((int*)p, __float_as_int(cmp), __float_as_int(val)));
}

__device__ __inline__ float atomic_cas_mul(float* dst, float src) {
  float old = *dst, expect;
  do {
    expect = old;
    old = atomicCAS_f32(dst, expect, src * expect);
    // atomicCAS(dst, cmp, src) { old = *dst; if (old == cmp) *dst = src; }
  } while (fabsf(expect - old) >= epsilon);
  return old;
}

template <typename T>
__global__ void parallel_mul(T* v, T* mul, const size_t n) {
  // optimize: accumulated multiplication
  T _mul = static_cast<T>(1.f);
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    // v[i] = 1.f / sqrtf(static_cast<T>(i + epsilon)); // casually init
    v[i] = sqrtf(i + epsilon); // casually init
    _mul *= v[i];
  }
  atomic_cas_mul(mul, _mul);
  // only one atomic op on one thread
}

#endif // __AtomicCas_mul_impl

int main(int argc, char const* argv[]) {
  int n = 2048;
  if (argc == 2) {
    n = std::atoi(argv[1]);
  }

  std::vector<float, CudaAllocator<float>> v(n);
  std::vector<float, CudaAllocator<float>> sum_gpu(1);
  sum_gpu = {0.f};
  parallel_sum<float><<<32, 256>>>(v.data(), sum_gpu.data(), n);

  checkCudaErrors(cudaDeviceSynchronize());

  float sum_cpu = 0.f;
  for (int i = 0; i < n; ++i) {
    sum_cpu += v[i];
    // std::printf("curr sum: %f\n", sum_cpu);
  }

  /* std::printf("sum_on_gpu = %f sum_on_cpu = %f \n", sum_cpu, v.at(n));
  // Buffer memory gpu or cpu-only once, so who comes first can be printed
  correctly.
  // Lesson: gpu only for computation */

  std::cout << "sum_on_gpu = " << sum_gpu[0] << " sum_on_cpu = " << sum_cpu << std::endl;

#if __AtomicCas_mul_impl

  std::vector<float, CudaAllocator<float>> mul_gpu(1);
  mul_gpu = {1.f};

  parallel_mul<float><<<32, 256>>>(v.data(), mul_gpu.data(), n);
  checkCudaErrors(cudaDeviceSynchronize());

  float mul_cpu = 1.f;
  std::for_each(v.rbegin(), v.rend(), [&mul_cpu](const float& x) { mul_cpu *= x; });
  std::cout << "mul_on_gpu = " << mul_gpu[0] << " mul_on_cpu = " << mul_cpu << std::endl;

#endif // __AtomicCas_mul_impl

  return 0;
}

#endif // __Impl_with_thrust
