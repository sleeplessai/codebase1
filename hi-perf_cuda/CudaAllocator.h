#pragma once
#include "helper_cuda.h"

template <typename T>
struct CudaAllocator {
  using value_type = T;

  T* allocate(size_t size) const {
    T* ptr = nullptr;
    checkCudaErrors(cudaMallocManaged(&ptr, size * sizeof(T)));
    return ptr;
  }

  void deallocate(T* ptr, size_t) const noexcept {
    checkCudaErrors(cudaFree(ptr));
  }

  // Disable automatic zero-value constructor
  template <class... Args>
  void construct(T* p, Args&&... args) {
    if constexpr (!(sizeof...(Args) == 0 && std::is_pod_v<T>)) {
      ::new (static_cast<void*>(p)) T(std::forward<Args>(args)...);
    }
    // https://en.cppreference.com/w/cpp/language/parameter_pack
  }
};
// https://liam.page/2018/03/16/keywords-typename-and-class-in-Cxx/
