#include <kernel/bindings.h>
#include <kernel/point.h>

#include <kernel/point_fitting.h>
#include <kernel/point_jacobian.h>

#include <kernel/eigen_mapping.h>

#include <kernel/math/math.h>

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
    bool converge)
{
    return kf_point_state_step(
        _body_cloud, body_cloud_size,
        _world_cloud, world_cloud_size,
        _nearest_points, nearest_points_size, nearest_points_sizes,
        _normvec, normvec_size,
        rmagine::Quaterniond(s->rot.w(), s->rot.x(), s->rot.y(), s->rot.z()),
        rmagine::Vector3d(s->offset_T_L_I.x(), s->offset_T_L_I.y(), s->offset_T_L_I.z()),
        rmagine::Quaterniond(s->offset_R_L_I.w(), s->offset_R_L_I.x(), s->offset_R_L_I.y(), s->offset_R_L_I.z()),
        rmagine::Vector3d(s->pos.x(), s->pos.y(), s->pos.z()),
        kd_tree,
        point_selected_surf,
        _laser_cloud_ori, laser_cloud_ori_size,
        _corr_normvect, corr_normvect_size,
        converge);
}

void step_jacobian(
    bool *point_selected_surf,
    esekfom::dyn_share_datastruct<double> &ekfom_data,
    state_ikfom &s,
    void *_laser_cloud_ori, size_t laser_cloud_ori_size,
    void *_corr_normvect, size_t corr_normvect_size,
    size_t effct_feat_num)
{
    // allocate cuda matrixes on GPU
    rmagine::MatrixXd h_x(ekfom_data.h_x.rows(), ekfom_data.h_x.cols());
    rmagine::MatrixXd h(ekfom_data.h.rows(), ekfom_data.h.cols());

    kf_jacobian(
        rmagine::Quaterniond(s.rot.w(), s.rot.x(), s.rot.y(), s.rot.z()),
        rmagine::Vector3d(s.offset_T_L_I.x(), s.offset_T_L_I.y(), s.offset_T_L_I.z()),
        rmagine::Quaterniond(s.offset_R_L_I.w(), s.offset_R_L_I.x(), s.offset_R_L_I.y(), s.offset_R_L_I.z()),
        point_selected_surf,
        _laser_cloud_ori, laser_cloud_ori_size,
        _corr_normvect, corr_normvect_size,
        h_x,
        h,
        effct_feat_num);

    // copy back to CPU
    ekfom_data.h_x = toEigen(h_x);
    ekfom_data.h = toEigen(h);
}