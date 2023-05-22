/**
 * @brief Kernels for per point jacobian matrix generation state steps in the kalman filter.
 * @author Christopher "stelzo" Sieh
 * @date 2022-05-21
 */

#include <cuda/point_jacobian.h>
#include <cuda/point.h>

#include <cuda/common.h>
#include <cuda/memory.h>

JLIO_KERNEL
void krnl_jacobian(PointXYZINormal_CUDA *laser_cloud_ori, size_t laser_cloud_ori_size,
                   PointXYZINormal_CUDA *corr_normvect, size_t corr_normvect_size,
                   rmagine::Quaterniond rot,
                   rmagine::Vector3d offset_T_L_I,
                   rmagine::Quaterniond offset_R_L_I,
                   double *h_x_raw, int h_x_rows, int h_x_cols,
                   double *h_raw, int h_rows, int h_cols,
                   bool extrinsic_est_en
#ifndef USE_CUDA
                   ,
                   size_t i
#endif
)
{
// ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); // 23
// ekfom_data.h.resize(effct_feat_num);
#ifdef USE_CUDA
    int i = blockIdx.x * blockDim.x + threadIdx.x;
#endif
    if (i >= laser_cloud_ori_size)
        return;

    rmagine::Vector3d v(laser_cloud_ori[i].x, laser_cloud_ori[i].y, laser_cloud_ori[i].z);
    rmagine::Matrix3x3d S;
    S(0, 0) = 0.0;
    S(0, 1) = -v.z;
    S(0, 2) = v.y;
    S(1, 0) = v.z;
    S(1, 1) = 0.0;
    S(1, 2) = -v.x;
    S(2, 0) = -v.y;
    S(2, 1) = v.x;
    S(2, 2) = 0.0;
    rmagine::Vector3d point_this = offset_R_L_I * v + offset_T_L_I;

    /*** get the normal vector of closest surface/corner ***/
    const PointXYZINormal_CUDA &norm_p = corr_normvect[i];
    rmagine::Vector3d norm_vec(norm_p.x, norm_p.y, norm_p.z);

    /*** calculate the Measuremnt Jacobian matrix H ***/
    rmagine::Vector3d C = rot.conjugate() * norm_vec;
    rmagine::Vector3d A = S * C;
    if (extrinsic_est_en) // false with given extrinsic
    {
        rmagine::Vector3d B = S * (offset_R_L_I.conjugate() * C); // s.rot.conjugate()*norm_vec);
        h_x_raw[i + 0 * h_x_cols] = norm_p.x;
        h_x_raw[i + 1 * h_x_cols] = norm_p.y;
        h_x_raw[i + 2 * h_x_cols] = norm_p.z;

        h_x_raw[i + 3 * h_x_cols] = A.x;
        h_x_raw[i + 4 * h_x_cols] = A.y;
        h_x_raw[i + 5 * h_x_cols] = A.z;

        h_x_raw[i + 6 * h_x_cols] = B.x;
        h_x_raw[i + 7 * h_x_cols] = B.y;
        h_x_raw[i + 8 * h_x_cols] = B.z;

        h_x_raw[i + 9 * h_x_cols] = C.x;
        h_x_raw[i + 10 * h_x_cols] = C.y;
        h_x_raw[i + 11 * h_x_cols] = C.z;
    }
    else
    {
        h_x_raw[i + 0 * h_x_cols] = norm_p.x;
        h_x_raw[i + 1 * h_x_cols] = norm_p.y;
        h_x_raw[i + 2 * h_x_cols] = norm_p.z;

        h_x_raw[i + 3 * h_x_cols] = A.x;
        h_x_raw[i + 4 * h_x_cols] = A.y;
        h_x_raw[i + 5 * h_x_cols] = A.z;

        h_x_raw[i + 6 * h_x_cols] = 0.0;
        h_x_raw[i + 7 * h_x_cols] = 0.0;
        h_x_raw[i + 8 * h_x_cols] = 0.0;

        h_x_raw[i + 9 * h_x_cols] = 0.0;
        h_x_raw[i + 10 * h_x_cols] = 0.0;
        h_x_raw[i + 11 * h_x_cols] = 0.0;
    }

    /*** Measuremnt: distance to the closest surface/corner ***/
    h_raw[i] = -norm_p.intensity;
}

void kf_jacobian(
    rmagine::Quaterniond rot,
    rmagine::Vector3d offset_T_L_I,
    rmagine::Quaterniond offset_R_L_I,
    bool *point_selected_surf,
    void *_laser_cloud_ori, size_t laser_cloud_ori_size,
    void *_corr_normvect, size_t corr_normvect_size,
    rmagine::MatrixXd &h_x,
    rmagine::MatrixXd &h,
    size_t effct_feat_num)
{
    PointXYZINormal_CUDA *laser_cloud_ori = (PointXYZINormal_CUDA *)_laser_cloud_ori;
    PointXYZINormal_CUDA *corr_normvect = (PointXYZINormal_CUDA *)_corr_normvect;
    constexpr size_t THREADS_PER_BLOCK = 1024;

#ifndef USE_CUDA
    auto idx = jlio::indexIota(effct_feat_num);
    std::for_each(THREADING idx.begin(), idx.end(), [&](size_t i)
                  { krnl_jacobian(laser_cloud_ori, laser_cloud_ori_size,
                                  corr_normvect, corr_normvect_size,
                                  rot,
                                  offset_T_L_I,
                                  offset_R_L_I,
                                  h_x.m_data, h_x.m_numRows, h_x.m_numCols,
                                  h.m_data, h.m_numRows, h.m_numCols,
                                  false,
                                  i); });
#else
    krnl_jacobian<<<std::ceil((float)effct_feat_num / THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(
        laser_cloud_ori, laser_cloud_ori_size,
        corr_normvect, corr_normvect_size,
        rot,
        offset_T_L_I,
        offset_R_L_I,
        h_x.m_data, h_x.m_numRows, h_x.m_numCols,
        h.m_data, h.m_numRows, h.m_numCols,
        false);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
#endif
}

JLIO_KERNEL
void krnl_jacobian_test(double *data, double *mat, int rows, int cols, int i)
{
    for (size_t j = 0; j < 12; j++)
    {
        mat[i + j * rows] = data[j];
    }
}
