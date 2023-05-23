#include <kernel/memory.h>
#include <iostream>

namespace jlio
{
    template <typename T>
    void malloc(T **ptr, size_t size)
    {
#ifdef USE_CUDA
        CHECK_CUDA_ERROR(cudaMallocManaged(ptr, size));
#else
        *ptr = (T *)std::malloc(size);
        if (ptr == NULL)
        {
            std::cerr << "Error allocating memory" << std::endl;
            std::exit(EXIT_FAILURE);
        }
#endif
    }

    void memset(void *ptr, int value, size_t count)
    {
#ifdef USE_CUDA
        CHECK_CUDA_ERROR(cudaMemset(ptr, value, count));
#else
        std::memset(ptr, value, count);
#endif
    }

#ifndef USE_CUDA
    std::vector<size_t> indexIota(size_t size)
    {
        std::vector<size_t> indices(size);
        for (size_t i = 0; i < size; ++i)
        {
            indices[i] = i;
        }
        return std::move(indices);
    }
#endif

    void memcpy(void *dst, const void *src, size_t count, int kind)
    {
#ifdef USE_CUDA
        CHECK_CUDA_ERROR(cudaMemcpy(dst, src, count, jlio::cudaMemcpyKindMap[kind]));
        //dst = src; // TODO only on jetson: no need to copy to and from device when using unified memory; but just assigning the pointer would make free fail
#else
        std::memcpy(dst, src, count);
#endif
    }

    void free(void *ptr)
    {
#ifdef USE_CUDA
        CHECK_CUDA_ERROR(cudaFree(ptr));
#else
        std::free(ptr);
#endif
    }

} // namespace jlio