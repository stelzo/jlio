#pragma once

#ifdef USE_CUDA
#define CHECK_LAST_CUDA_ERROR() checkLastCudaError(__FILE__, __LINE__)
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

#include <iostream>

template <typename T>
void check(T err, const char *const func, const char *const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        std::exit(EXIT_FAILURE);
    }
}

void checkLastCudaError(const char *const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        std::exit(EXIT_FAILURE);
    }
}
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
#include <cuda_runtime.h>

#define JLIO_FUNCTION __device__
#define JLIO_KERNEL __global__
#define JLIO_INLINE_FUNCTION __forceinline__ JLIO_FUNCTION
#else
#define JLIO_FUNCTION
#define JLIO_KERNEL
#define JLIO_INLINE_FUNCTION __inline__
#include <execution> // for parallel algorithms in std
#include <algorithm>
#endif
