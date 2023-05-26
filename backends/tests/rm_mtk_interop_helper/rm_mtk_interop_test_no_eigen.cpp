#include <kernel/common.h>
#include <kernel/memory.h>
#include <kernel/math/math.h>

JLIO_KERNEL
void krnl_jacobian_test(double *data, double *mat, int rows, int cols, int i)
{
    for (size_t j = 0; j < 12; j++)
    {
        mat[i + j * rows] = data[j];
    }
}

// return on cpu
rmagine::MatrixXd jacobian_test_gpu_internal(double *data, int rows, int cols, int i)
{
    rmagine::MatrixXd mdata_host(rows, cols);

    double *data_gpu = nullptr;
    jlio::malloc((void **)&data_gpu, sizeof(double) * 12);
    jlio::memcpy((void **)data_gpu, data, sizeof(double) * 12, jlio::cudaMemcpyHostToDevice);

#ifdef USE_CUDA
    krnl_jacobian_test<<<1, 1>>>(data_gpu, mdata_host.m_data, rows, cols, i);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
#else
    krnl_jacobian_test(data_gpu, mdata_host.m_data, rows, cols, i);
#endif

    jlio::free((void **)data_gpu);

    return mdata_host;
}