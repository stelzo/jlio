#include <kernel/common.h>
#include <kernel/preprocess.h>
#include <kernel/point.h>
#include <kernel/memory.h>

#ifndef USE_CUDA
#include <mutex>
#endif

JLIO_INLINE_FUNCTION
float squared_distance(const jlio::PointXYZINormal &p)
{
    return p.x * p.x + p.y * p.y + p.z * p.z;
}

JLIO_KERNEL
void krnl_filter_points_xyziring(unsigned char* raw_data, uint32_t raw_data_size, uint32_t point_step, jlio::PointXYZINormal* output_buffer, uint32_t* out_size, float near_dist, float far_dist
    #ifndef USE_CUDA
    , int32_t id, std::mutex &mtx
    #endif
    )
{
    #ifdef USE_CUDA
    int32_t id = blockDim.x * blockIdx.x + threadIdx.x;
    #endif

    // too many threads
    if (id >= raw_data_size / sizeof(jlio::OusterPoint))
    {
        return;
    }
    
    // downsampling
    if (id % 3 != 0)
    {
        return;
    }

    jlio::OusterPoint* res_ptr = (jlio::OusterPoint*) (raw_data + (point_step * id));
    jlio::PointXYZINormal result;
    result.x = res_ptr->x;
    result.y = res_ptr->y;
    result.z = res_ptr->z;

    float dist = squared_distance(result);
    if (dist < near_dist * near_dist || dist > far_dist * far_dist) // saving the sqrt, so we need to square the input distance
    {
        return;
    }

    result.intensity = res_ptr->intensity;
    result.normal_x = 0;
    result.normal_y = 0;
    result.normal_z = 0;
    result.curvature = res_ptr->t * 1.e-6f; // nanosecond with ousterpoint

    #ifndef USE_CUDA
    mtx.lock();
    int size_before_add = *out_size;
    #else
    int size_before_add = atomicAdd(out_size, 1);
    #endif
    output_buffer[size_before_add] = result;
    #ifndef USE_CUDA
    *out_size = size_before_add + 1;
    mtx.unlock();
    #endif
}

void filter_map_ouster(const u_int8_t* source, uint32_t source_size, uint32_t point_step, jlio::PointXYZINormal* out, uint32_t* out_size, float near_dist, float far_dist)
{
    assert(point_step == sizeof(jlio::OusterPoint));

    if (source_size == 0 || source == nullptr)
    {
        return;
    }

    uint32_t input_points_len = source_size / point_step;

    // working buffer on GPU where the input point cloud lives
    u_int8_t* input_raw_buffer = nullptr;
    jlio::malloc(&input_raw_buffer, source_size);
    jlio::memcpy(input_raw_buffer, reinterpret_cast<const void*>(source), static_cast<size_t>(source_size), jlio::cudaMemcpyHostToDevice);

    // output buffer with reduced, filtered size
    jlio::PointXYZINormal* output_pt_buffer = nullptr;
    jlio::malloc(&output_pt_buffer, input_points_len * sizeof(jlio::OusterPoint)); // allocate enough, because we know the upper size beforehand

    // size for the result as return but needed in the kernel
    uint32_t* out_size_device = nullptr;
    jlio::malloc(&out_size_device, sizeof(uint32_t));
    jlio::memset(out_size_device, 0, sizeof(uint32_t));
    
#ifdef USE_CUDA
    constexpr size_t THREADS_PER_BLOCK = 1024;
    size_t BLOCKS = std::ceil((float)input_points_len / THREADS_PER_BLOCK);
    krnl_filter_points_xyziring<<<BLOCKS, THREADS_PER_BLOCK>>>(input_raw_buffer, source_size, point_step, output_pt_buffer, out_size_device, near_dist, far_dist);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
#else
    auto idx = jlio::indexIota(input_points_len);
    std::mutex mtx;
    std::for_each(THREADING idx.begin(), idx.end(), [&](int32_t id) {
        krnl_filter_points_xyziring(input_raw_buffer, source_size, point_step, output_pt_buffer, out_size_device, near_dist, far_dist, id, mtx);
    });
#endif

    jlio::free(input_raw_buffer);

    jlio::memcpy(out_size, out_size_device, sizeof(uint32_t), jlio::cudaMemcpyDeviceToHost);
    jlio::free(out_size_device);

    jlio::malloc(&out, sizeof(jlio::PointXYZINormal) * (*out_size));
    jlio::memcpy(out, output_pt_buffer, sizeof(jlio::PointXYZINormal) * (*out_size), jlio::cudaMemcpyDeviceToDevice);
    jlio::free(output_pt_buffer);
}