#pragma once

#ifdef USE_CUDA
#include <cuda/common.h>
#else
#include <cstring>
#include <cstdlib>
#include <vector>
#endif

namespace jlio
{
    template <typename T>
    void malloc(T **ptr, size_t size);

    void memset(void *ptr, int value, size_t count);

#ifndef USE_CUDA
    std::vector<size_t> indexIota(size_t size);
#endif

    void memcpy(void *dst, const void *src, size_t count, int kind = 0);

    void free(void *ptr);

    enum cudaMemcpyKind
    {
        cudaMemcpyHostToHost = 0,
        cudaMemcpyHostToDevice = 1,
        cudaMemcpyDeviceToHost = 2,
        cudaMemcpyDeviceToDevice = 3,
        cudaMemcpyDefault = 4
    };

#ifdef USE_CUDA
    ::cudaMemcpyKind cudaMemcpyKindMap[] = {::cudaMemcpyHostToHost,
                                            ::cudaMemcpyHostToDevice,
                                            ::cudaMemcpyDeviceToHost,
                                            ::cudaMemcpyDeviceToDevice,
                                            ::cudaMemcpyDefault};
#endif
} // namespace jlio