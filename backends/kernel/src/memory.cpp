#include <kernel/memory.h>
#include <kernel/point.h>

namespace jlio
{
    void malloc(void **ptr, size_t size)
    {
#ifdef USE_CUDA
        CHECK_CUDA_ERROR(cudaMallocManaged(ptr, size));
#else
        *ptr = std::malloc(size);
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
        ::cudaMemcpyKind _kind;
        switch (kind)
        {
        case 0:
            _kind = ::cudaMemcpyHostToHost;
            break;
        case 1:
            _kind = ::cudaMemcpyHostToDevice;
            break;
        case 2:
            _kind = ::cudaMemcpyDeviceToHost;
            break;
        case 3:
            _kind = ::cudaMemcpyDeviceToDevice;
            break;
        case 4:
            _kind = ::cudaMemcpyDefault;
            break;
        default:
            _kind = ::cudaMemcpyDefault;
        }
        CHECK_CUDA_ERROR(cudaMemcpy(dst, src, count, _kind));
        // dst = src; // TODO only on jetson: no need to copy to and from device when using unified memory; but just assigning the pointer would make free fail
#else
        std::memcpy(dst, src, count);
#endif
    }

    void free(void *ptr)
    {
        if (ptr == NULL)
        {
            return;
        }

#ifdef USE_CUDA
        CHECK_CUDA_ERROR(cudaFree(ptr));
#else
        std::free(ptr);
#endif
    }

} // namespace jlio
