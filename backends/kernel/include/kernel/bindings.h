#pragma once

#include <kalman/kalman.h>

int step_kalman(
    void *_body_cloud, size_t body_cloud_size,
    void *_world_cloud, size_t world_cloud_size,
    void *_nearest_points, size_t nearest_points_size, size_t *nearest_points_sizes,
    void *_normvec, size_t normvec_size,
    state_ikfom *s,
    void *kd_tree,
    bool *point_selected_surf,
    void *_laser_cloud_ori, size_t laser_cloud_ori_size,
    void *_corr_normvect, size_t corr_normvect_size,
    bool converge);

void step_jacobian(
    bool *point_selected_surf,
    state_ikfom *s,
    void *_laser_cloud_ori, size_t laser_cloud_ori_size,
    void *_corr_normvect, size_t corr_normvect_size,
    size_t effct_feat_num);