cmake_minimum_required(VERSION 3.18)

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

project(HiPerformanceCuda LANGUAGES CXX CUDA)

add_executable(00greet 00greet.cu)
add_executable(01thread 01thread.cu)
add_executable(02kernel 02kernel.cu helper_cuda.h)
add_executable(03unified_memory 03unified_memory.cu helper_cuda.h)
add_executable(04array 04array.cu helper_cuda.h)
add_executable(05cpp_encapsulation 05cpp_encapsulation.cu helper_cuda.h CudaAllocator.h)

add_executable(06functional 06functional.cu helper_cuda.h CudaAllocator.h)
target_compile_options(06functional PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

add_executable(07math 07math.cu helper_cuda.h CudaAllocator.h)
target_compile_options(07math PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

add_executable(08thrust 08thrust.cu helper_cuda.h CudaAllocator.h)
target_compile_options(08thrust PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

add_executable(09atomic_ops 09atomic_ops.cu helper_cuda.h CudaAllocator.h)
target_compile_options(09atomic_ops PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

add_executable(10block_shared_memory 10block_shared_memory.cu helper_cuda.h CudaAllocator.h)
target_compile_options(10block_shared_memory PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

add_executable(11matrix_transpose 11matrix_transpose.cu helper_cuda.h CudaAllocator.h)
target_compile_options(11matrix_transpose PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
