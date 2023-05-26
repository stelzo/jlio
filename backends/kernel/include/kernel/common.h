#pragma once

#ifdef USE_CUDA
#include <cuda_runtime.h>
#define CHECK_LAST_CUDA_ERROR() checkLastCudaError(__FILE__, __LINE__)
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

void check(cudaError_t err, const char *const func, const char *const file, const int line);
void checkLastCudaError(const char *const file, const int line);
#endif

// debugging
// #define USE_CUDA

#ifndef THREADING
#ifdef USE_THREADING
#define THREADING std::execution::par_unseq,
#else
#define THREADING
#endif
#endif

#ifdef USE_CUDA
#define JLIO_FUNCTION __device__
#define JLIO_KERNEL __global__
#define JLIO_INLINE_FUNCTION __forceinline__ JLIO_FUNCTION
#define JLIO_INLINE_DEVICE_HOST __forceinline__ __host__ __device__
#else
#define JLIO_FUNCTION
#define JLIO_KERNEL
#define JLIO_INLINE_FUNCTION __inline__
#define JLIO_INLINE_DEVICE_HOST __inline__
#include <execution> // for parallel algorithms in std
#include <algorithm>
#endif
