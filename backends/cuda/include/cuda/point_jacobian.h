#pragma once

#include <cuda/math/math.h>

void kf_jacobian(
    rmagine::Quaterniond rot,
    rmagine::Vector3d offset_T_L_I,
    rmagine::Quaterniond offset_R_L_I,
    bool *point_selected_surf,
    void *_laser_cloud_ori, size_t laser_cloud_ori_size,
    void *_corr_normvect, size_t corr_normvect_size,
    rmagine::MatrixXd &h_x,
    rmagine::MatrixXd &h,
    size_t effct_feat_num);