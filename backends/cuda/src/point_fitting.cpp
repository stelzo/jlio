/**
 * @brief Kernels for per point nearest neighbor search and filtering for state steps in the kalman filter.
 * @author Christopher "stelzo" Sieh
 * @date 2022-05-21
 */

#include <cuda/point_fitting.h>
#include <cuda/point.h>
#include <cuda/pca.h>
#include <cuda/common.h>
#include <cuda/kdtree.h>
#include <cuda/memory.h>

#ifndef USE_CUDA
#include <mutex>
#endif

#define FLT_MAX 3.402823466e+38F

namespace jlio
{
    template <typename T>
    JLIO_INLINE_FUNCTION const T &min(const T &a, const T &b)
    {
        return (a < b) ? a : b;
    }

    template <typename T>
    JLIO_INLINE_FUNCTION const T &max(const T &a, const T &b)
    {
        return (a > b) ? a : b;
    }

}

JLIO_INLINE_FUNCTION
float calc_box_dist_gpu(KD_TREE_NODE<PointXYZINormal_CUDA> *node, PointXYZINormal_CUDA point)
{
    if (node == nullptr)
        return INFINITY;
    float min_dist = 0.0;
    if (point.x < node->node_range_x[0])
    {
        min_dist += (point.x - node->node_range_x[0]) * (point.x - node->node_range_x[0]);
    }
    if (point.x > node->node_range_x[1])
    {
        min_dist += (point.x - node->node_range_x[1]) * (point.x - node->node_range_x[1]);
    }
    if (point.y < node->node_range_y[0])
    {
        min_dist += (point.y - node->node_range_y[0]) * (point.y - node->node_range_y[0]);
    }
    if (point.y > node->node_range_y[1])
    {
        min_dist += (point.y - node->node_range_y[1]) * (point.y - node->node_range_y[1]);
    }
    if (point.z < node->node_range_z[0])
    {
        min_dist += (point.z - node->node_range_z[0]) * (point.z - node->node_range_z[0]);
    }
    if (point.z > node->node_range_z[1])
    {
        min_dist += (point.z - node->node_range_z[1]) * (point.z - node->node_range_z[1]);
    }

    /*if (min_dist < 0.001)
    {
        printf("min_dist < %f, node_range_x %f. %f. %f, node_range_y %f. %f. %f, node_range_z %f. %f. %f actual p: %f, %f, %f, PPDIST=%f\n",
        min_dist, node->node_range_x[0], node->node_range_x[1], point.x,
        node->node_range_y[0], node->node_range_y[1], point.y,
         node->node_range_z[0], node->node_range_z[1], point.z,
         node->point.x, node->point.y, node->point.z,
         calc_dist_gpu(node->point, point));
    }*/
    return min_dist;
}

JLIO_INLINE_FUNCTION
float calc_dist_gpu(PointXYZINormal_CUDA a, PointXYZINormal_CUDA b)
{
    return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z);
}

JLIO_FUNCTION
void Push_Down(KD_TREE_NODE<PointXYZINormal_CUDA> *root)
{
    if (root == nullptr)
        return;

    if (root->need_push_down_to_left && root->left_son_ptr != nullptr)
    {
        root->left_son_ptr->tree_downsample_deleted |= root->tree_downsample_deleted;
        root->left_son_ptr->point_downsample_deleted |= root->tree_downsample_deleted;
        root->left_son_ptr->tree_deleted =
            root->tree_deleted || root->left_son_ptr->tree_downsample_deleted;
        root->left_son_ptr->point_deleted =
            root->left_son_ptr->tree_deleted || root->left_son_ptr->point_downsample_deleted;
        if (root->tree_downsample_deleted)
            root->left_son_ptr->down_del_num = root->left_son_ptr->TreeSize;
        if (root->tree_deleted)
            root->left_son_ptr->invalid_point_num = root->left_son_ptr->TreeSize;
        else
            root->left_son_ptr->invalid_point_num = root->left_son_ptr->down_del_num;
        root->left_son_ptr->need_push_down_to_left = true;
        root->left_son_ptr->need_push_down_to_right = true;
        root->need_push_down_to_left = false;
    }
    if (root->need_push_down_to_right && root->right_son_ptr != nullptr)
    {
        root->right_son_ptr->tree_downsample_deleted |= root->tree_downsample_deleted;
        root->right_son_ptr->point_downsample_deleted |= root->tree_downsample_deleted;
        root->right_son_ptr->tree_deleted =
            root->tree_deleted || root->right_son_ptr->tree_downsample_deleted;
        root->right_son_ptr->point_deleted =
            root->right_son_ptr->tree_deleted || root->right_son_ptr->point_downsample_deleted;
        if (root->tree_downsample_deleted)
            root->right_son_ptr->down_del_num = root->right_son_ptr->TreeSize;
        if (root->tree_deleted)
            root->right_son_ptr->invalid_point_num = root->right_son_ptr->TreeSize;
        else
            root->right_son_ptr->invalid_point_num = root->right_son_ptr->down_del_num;
        root->right_son_ptr->need_push_down_to_left = true;
        root->right_son_ptr->need_push_down_to_right = true;
        root->need_push_down_to_right = false;
    }
    return;
}

JLIO_FUNCTION
void Search(KD_TREE_NODE<PointXYZINormal_CUDA> *root, int k_nearest, PointXYZINormal_CUDA point, MANUAL_HEAP_GPU *q, float max_dist)
{
    if (root == nullptr || root->tree_deleted)
    {
        return;
    }

    float max_dist_sqr = max_dist * max_dist;
    float dist_to_kth_neighbor = FLT_MAX;

    bool bt = false;
    KD_TREE_NODE<PointXYZINormal_CUDA> *prev_node = nullptr;
    KD_TREE_NODE<PointXYZINormal_CUDA> *current_node = root;

    do
    {
        if (root->tree_deleted)
        {
            break;
        }

        float cur_dist = calc_box_dist_gpu(current_node, point);

        if (current_node->need_push_down_to_left || current_node->need_push_down_to_right)
        {
            Push_Down(current_node);
        }

        if (!bt && !current_node->point_deleted)
        {
            float dist = calc_dist_gpu(point, current_node->point);
            // printf("dist: %f %f %f %f\n", current_node->point.x, current_node->point.y, current_node->point.z, dist);
            if (dist <= max_dist_sqr && (q->size() < k_nearest || dist < q->top().dist))
            {
                if (q->size() >= k_nearest)
                    q->pop();
                PointType_CMP_GPU current_point(current_node->point, dist);
                q->push(current_point);
                // printf("push point: %f %f %f %f\n", current_node->point.x, current_node->point.y, current_node->point.z, dist);
                if (q->size() == k_nearest)
                {
                    dist_to_kth_neighbor = q->top().dist;
                }
            }
        }

        auto *child_left = current_node->left_son_ptr;
        auto *child_right = current_node->right_son_ptr;

        float dist_left_node = calc_box_dist_gpu(child_left, point);
        float dist_right_node = calc_box_dist_gpu(child_right, point);

        bool traverse_left = child_left != nullptr && dist_left_node <= jlio::min(dist_to_kth_neighbor, max_dist_sqr);
        bool traverse_right = child_right != nullptr && dist_right_node <= jlio::min(dist_to_kth_neighbor, max_dist_sqr);

        auto *best_child = (dist_left_node <= dist_right_node) ? child_left : child_right;
        auto *other_child = (dist_left_node <= dist_right_node) ? child_right : child_left;

        if (!bt)
        {
            if (!traverse_left && !traverse_right)
            {
                bt = true;
                auto parent = current_node->father_ptr;
                prev_node = current_node;
                current_node = parent;
            }
            else
            {
                prev_node = current_node;
                current_node = (traverse_left) ? child_left : child_right;
                if (traverse_left && traverse_right)
                {
                    current_node = best_child;
                }
            }
        }
        else
        {
            float mind(INFINITY);

            if (other_child != nullptr)
            {
                mind = jlio::max(dist_left_node, dist_right_node);
            }

            if (other_child != nullptr && prev_node == best_child && mind <= dist_to_kth_neighbor)
            {
                prev_node = current_node;
                current_node = other_child;
                bt = false;
            }
            else
            {
                auto parent = current_node->father_ptr;
                prev_node = current_node;
                current_node = parent;
            }
        }
    } while (current_node != nullptr);
}

JLIO_FUNCTION
void Nearest_Search(void *root, PointXYZINormal_CUDA point, size_t k_nearest,
                    PointXYZINormal_CUDA *Nearest_Points, int *Nearest_Points_Size,
                    float *Point_Distance, size_t *Point_Distance_Size,
                    float max_dist)
{
    MANUAL_HEAP_GPU q;
    q.init();

    KD_TREE_NODE<PointXYZINormal_CUDA> *tree = (KD_TREE_NODE<PointXYZINormal_CUDA> *)root;
    Search(tree, k_nearest, point, &q, max_dist);

    int k_found = jlio::min((int)k_nearest, int(q.size()));
    *Point_Distance_Size = 0;
    *Nearest_Points_Size = 0;
    for (int i = 0; i < k_found; i++)
    {
        Nearest_Points[*Nearest_Points_Size] = q.top().point;
        Point_Distance[*Point_Distance_Size] = q.top().dist;
        q.pop();

        (*Nearest_Points_Size)++;
        (*Point_Distance_Size)++;
    }

    //(*Nearest_Points_Size)--;
    //(*Point_Distance_Size)--;

    /*printf("search done. k-found: %d, 1st nearest p %f, %f, %f; dist %f\n",
    k_found, Nearest_Points[*Nearest_Points_Size-1].x, Nearest_Points[*Nearest_Points_Size-1].y, Nearest_Points[*Nearest_Points_Size-1].z,
    Point_Distance[*Point_Distance_Size-1]);*/

    // printf("search done. k-found: %d\n", k_found);
    return;
}

JLIO_KERNEL
void krnl_raw_nearest_search(void *root, PointXYZINormal_CUDA *point, size_t k_nearest,
                             PointXYZINormal_CUDA *Nearest_Points, int *Nearest_Points_Size,
                             float *Point_Distance, size_t *Point_Distance_Size,
                             float max_dist)
{
    Nearest_Search(root, *point, k_nearest, Nearest_Points, Nearest_Points_Size, Point_Distance, Point_Distance_Size, max_dist);
}

void Raw_Nearest_Search(void *root, PointXYZINormal_CUDA *point, size_t k_nearest,
                        PointXYZINormal_CUDA *Nearest_Points, int *Nearest_Points_Size,
                        float *Point_Distance, size_t *Point_Distance_Size,
                        float max_dist)
{
#ifdef USE_CUDA
    krnl_raw_nearest_search<<<1, 1>>>(root, point, k_nearest, Nearest_Points, Nearest_Points_Size, Point_Distance, Point_Distance_Size, max_dist);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
#else
    krnl_raw_nearest_search(root, point, k_nearest, Nearest_Points, Nearest_Points_Size, Point_Distance, Point_Distance_Size, max_dist);
#endif
}

/*
void jacobian_test_cpu(double* data, Eigen::MatrixXd* target, int i)
{
    assert(target != nullptr);
    assert(target->data() != nullptr);

    int h_x_rows = target->rows();
    int h_x_cols = target->cols();

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> h_x(target->data(), h_x_rows, h_x_cols);
    h_x.block<1, 12>(i, 0) << data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11];
}

void jacobian_test_gpu(double* data, int rows, int cols, Eigen::MatrixXd& target, int i)
{
    assert(data != nullptr);

    rmagine::MatrixXd mdata_host(rows, cols);

    double* data_gpu = nullptr;
    cudaMalloc(&data_gpu, sizeof(double) * 12);
    cudaMemcpy(data_gpu, data, sizeof(double) * 12, cudaMemcpyHostToDevice);

    krnl_jacobian_test<<<1, 1>>>(data_gpu, mdata_host.m_data, rows, cols, i);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();

    assert(mdata_host(i, 2) > 1.1);
    assert(mdata_host(i, 2) < 1.3);
    assert(mdata_host(i, 3) > 1.2);

    mdata_host.toEigenInpl(target);

    cudaFree(data_gpu);
}
*/

JLIO_KERNEL
void krnl_point_kf_state(void *_body_cloud, size_t body_cloud_size,
                         void *_world_cloud, size_t world_cloud_size,
                         void *_nearest_points, size_t nearest_points_size, size_t *nearest_points_sizes,
                         void *_normvec, size_t normvec_size,
                         void *kd_tree,
                         bool *point_selected_surf,
                         bool ekfom_data_converged,
                         rmagine::Quaterniond rot,
                         rmagine::Vector3d offset_T_L_I,
                         rmagine::Quaterniond offset_R_L_I,
                         rmagine::Vector3d pos
#ifndef USE_CUDA
                         ,
                         size_t i
#endif
)
{
#ifdef USE_CUDA
    int i = blockIdx.x * blockDim.x + threadIdx.x;
#endif
    if (i >= body_cloud_size)
        return;

    // conversion to cuda types with same memory layout
    PointXYZINormal_CUDA *body_cloud = (PointXYZINormal_CUDA *)_body_cloud;
    PointXYZINormal_CUDA *world_cloud = (PointXYZINormal_CUDA *)_world_cloud;
    PointXYZINormal_CUDA *nearest_points = (PointXYZINormal_CUDA *)_nearest_points;
    PointXYZINormal_CUDA *normvec = (PointXYZINormal_CUDA *)_normvec;

    constexpr size_t NUM_MATCH_POINTS = 5; // hardcoded everywhere but defines are not allowed in device code

    point_selected_surf[i] = false; // initialize if point is relevant for surface feature

    // map frame conversion with last pose
    rmagine::Vector3d p_body(body_cloud[i].x, body_cloud[i].y, body_cloud[i].z);
    rmagine::Vector3d p_global(rot * (offset_R_L_I * p_body + offset_T_L_I) + pos);
    world_cloud[i].x = p_global.x;
    world_cloud[i].y = p_global.y;
    world_cloud[i].z = p_global.z;
    world_cloud[i].intensity = body_cloud[i].intensity;

    // closest distances to map points
    float pointSearchSqDis[NUM_MATCH_POINTS];
    size_t dist_size = NUM_MATCH_POINTS;

    if (ekfom_data_converged)
    {
        // find closest points in map
        int nearest_size = 0;
        Nearest_Search(kd_tree, world_cloud[i], NUM_MATCH_POINTS, &nearest_points[i * NUM_MATCH_POINTS], &nearest_size, pointSearchSqDis, &dist_size, INFINITY);
        nearest_points_sizes[i] = (size_t)nearest_size;

        point_selected_surf[i] = nearest_points_sizes[i] < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false
                                                                                                                                 : true;
    }

    // disregard points that are not relevant for surface feature because they are too far away from every map point
    if (!point_selected_surf[i])
    {
        return;
    }

    point_selected_surf[i] = false;

    // build array of Eigen vectors for plane estimation
    rmagine::Vector3d nearest_points_custom[NUM_MATCH_POINTS];
    for (int j = 0; j < nearest_points_sizes[i]; j++)
    {
        nearest_points_custom[j] = rmagine::Vector3d(nearest_points[i * NUM_MATCH_POINTS + j].x,
                                                     nearest_points[i * NUM_MATCH_POINTS + j].y,
                                                     nearest_points[i * NUM_MATCH_POINTS + j].z);
    }

    rmagine::Vector3d normal;
    rmagine::VectorN<4, double> plane_coeffs;

    bool found_normal = pca_constant(nearest_points_custom, nearest_points_sizes[i], (double)0.1f, &normal, &plane_coeffs);
    if (!found_normal)
    {
        return;
    }

    normvec[i].x = (float)plane_coeffs(0);
    normvec[i].y = (float)plane_coeffs(1);
    normvec[i].z = (float)plane_coeffs(2);
    normvec[i].curvature = (float)plane_coeffs(3);

    rmagine::Vector3d delta = p_global - pos;
    if (normvec[i].x * delta.x + normvec[i].y * delta.y + normvec[i].z * delta.z > 0.0f)
    {
        normvec[i].x = -normvec[i].x;
        normvec[i].y = -normvec[i].y;
        normvec[i].z = -normvec[i].z;
    }

    float pd2 = normvec[i].x * world_cloud[i].x + normvec[i].y * world_cloud[i].y + normvec[i].z * world_cloud[i].z + normvec[i].curvature;
    double p_body_norm = sqrtf(p_body.x * p_body.x + p_body.y * p_body.y + p_body.z * p_body.z);
    double s = 1.0f - 0.9f * fabs(pd2) / p_body_norm;

    point_selected_surf[i] = fabs(s) > 0.9f;
}

JLIO_KERNEL
void krnl_filter_selected_surf(PointXYZINormal_CUDA *body_cloud, size_t body_cloud_size,
                               PointXYZINormal_CUDA *normvec, size_t normvec_size,
                               bool *point_selected_surf,
                               PointXYZINormal_CUDA *laser_cloud_ori, size_t laser_cloud_ori_size,
                               PointXYZINormal_CUDA *corr_normvect, size_t corr_normvect_size,
                               int *effct_feat_num
#ifndef USE_CUDA
                               ,
                               size_t i, std::mutex *mtx
#endif
)
{
#ifdef USE_CUDA
    int i = blockIdx.x * blockDim.x + threadIdx.x;
#endif
    if (i >= body_cloud_size || i >= normvec_size)
    {
        return;
    }

    if (!point_selected_surf[i])
    {
        // printf("point selected surf false\n");
        return;
    }

#ifdef USE_CUDA
    int old_effct_feat_num = atomicAdd(effct_feat_num, 1);
// printf("effct num %d\n", old_effct_feat_num);
#else
    mtx->lock();
    int old_effct_feat_num = *effct_feat_num;
    (*effct_feat_num)++;
    mtx->unlock();
#endif
    laser_cloud_ori[old_effct_feat_num] = body_cloud[i];
    corr_normvect[old_effct_feat_num] = normvec[i];
}

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
    bool converge)
{
    PointXYZINormal_CUDA *body_cloud = (PointXYZINormal_CUDA *)_body_cloud;
    PointXYZINormal_CUDA *normvec = (PointXYZINormal_CUDA *)_normvec;
    PointXYZINormal_CUDA *laser_cloud_ori = (PointXYZINormal_CUDA *)_laser_cloud_ori;
    PointXYZINormal_CUDA *corr_normvect = (PointXYZINormal_CUDA *)_corr_normvect;

    constexpr size_t THREADS_PER_BLOCK = 1024;

    // std::cout << "body_cloud_size " << body_cloud_size << ", normvec_size " << normvec_size << std::endl;
    size_t kf_state_grid_dim = static_cast<size_t>(std::ceil((float)body_cloud_size / THREADS_PER_BLOCK));

#ifdef USE_CUDA
    krnl_point_kf_state<<<kf_state_grid_dim, THREADS_PER_BLOCK>>>(
        _body_cloud, body_cloud_size,
        _world_cloud, world_cloud_size,
        _nearest_points, nearest_points_size, nearest_points_sizes,
        _normvec, normvec_size,
        kd_tree,
        point_selected_surf,
        converge,
        rot,
        offset_T_L_I,
        offset_R_L_I,
        pos);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
#else
    {
        auto idx = jlio::indexIota(body_cloud_size);
        std::for_each(THREADING idx.begin(), idx.end(), [&](size_t i)
                      { krnl_point_kf_state(
                            _body_cloud, body_cloud_size,
                            _world_cloud, world_cloud_size,
                            _nearest_points, nearest_points_size, nearest_points_sizes,
                            _normvec, normvec_size,
                            kd_tree,
                            point_selected_surf,
                            converge,
                            rot,
                            offset_T_L_I,
                            offset_R_L_I,
                            pos,
                            i); });
    }
#endif
    // std::cout << "point_kf_state run successfull" << std::endl;

    size_t selected_surf_grid_dim = static_cast<size_t>(std::ceil((float)body_cloud_size / THREADS_PER_BLOCK));
    // std::cout << "Dim " << selected_surf_grid_dim << ", " << THREADS_PER_BLOCK << std::endl;

    CHECK_LAST_CUDA_ERROR();
    int *effct_feat_num = nullptr;

    jlio::malloc(&effct_feat_num, sizeof(int));
    jlio::memset(effct_feat_num, 0, sizeof(int));

    int *effct_feat_num_host = nullptr;
    effct_feat_num_host = new int;

#ifdef USE_CUDA
    krnl_filter_selected_surf<<<selected_surf_grid_dim, THREADS_PER_BLOCK>>>(
        body_cloud, body_cloud_size,
        normvec, normvec_size,
        point_selected_surf,
        laser_cloud_ori, laser_cloud_ori_size,
        corr_normvect, corr_normvect_size,
        effct_feat_num);
    cudaDeviceSynchronize();
    CHECK_LAST_CUDA_ERROR();
#else
    {
        std::mutex mtx;
        auto idx = jlio::indexIota(body_cloud_size);
        std::for_each(THREADING idx.begin(), idx.end(), [&](size_t i)
                      { krnl_filter_selected_surf(
                            body_cloud, body_cloud_size,
                            normvec, normvec_size,
                            point_selected_surf,
                            laser_cloud_ori, laser_cloud_ori_size,
                            corr_normvect, corr_normvect_size,
                            effct_feat_num,
                            i, &mtx); });
    }
#endif

    jlio::memcpy(effct_feat_num_host, effct_feat_num, sizeof(int), jlio::cudaMemcpyDeviceToHost);

    if (effct_feat_num)
    {
        jlio::free(effct_feat_num);
        effct_feat_num = nullptr;
    }

    int res = *effct_feat_num_host;
    if (effct_feat_num_host)
    {
        delete effct_feat_num_host;
    }

    return res;
}