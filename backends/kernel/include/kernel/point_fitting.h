#pragma once

#include <kernel/math/math.h>

int kf_point_state_step(
    void *_body_cloud, size_t body_cloud_size,
    void *_world_cloud, size_t world_cloud_size,
    void *_nearest_points, size_t nearest_points_size, size_t *nearest_points_sizes,
    void *_normvec, size_t normvec_size,
    rmagine::Quaterniond rot,
    rmagine::Vector3d offset_T_L_I,
    rmagine::Quaterniond offset_R_L_I,
    rmagine::Vector3d pos,
    void *kd_tree,
    bool *point_selected_surf,
    void *_laser_cloud_ori, size_t laser_cloud_ori_size,
    void *_corr_normvect, size_t corr_normvect_size,
    bool converge);