#include <kernel/math/math.h>
#include <kalman/kalman.h>
#include <kernel/eigen_mapping.h>
#include <kernel/memory.h>
#include <Eigen/Dense>
#include <kernel/common.h>

#include <random>
#include <iostream>
#include <gtest/gtest.h>

#include <rm_mtk_interop_test_no_eigen.h>

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> distrib(-1.0, 1.0);

constexpr double epsilon = 1e-6;

Eigen::Quaterniond Random_Quaternion()
{
    double x = distrib(gen);
    double y = distrib(gen);
    double z = distrib(gen);
    double w = distrib(gen);
    return Eigen::Quaterniond(w, x, y, z).normalized();
}

/**
 * Testing interop with MTK structs. Rmagine structs work reliable on the GPU, while Eigen does not.
 */
TEST(rm_mtk_interop, quat)
{
    auto rquat = Random_Quaternion();

    SO3 rot(rquat.w(), rquat.x(), rquat.y(), rquat.z());
    rmagine::Quaterniond rm_rot(rquat.w(), rquat.x(), rquat.y(), rquat.z());

    // conjugation
    EXPECT_NEAR(rot.conjugate().w(), rm_rot.conjugate().w, epsilon);
    EXPECT_NEAR(rot.conjugate().x(), rm_rot.conjugate().x, epsilon);
    EXPECT_NEAR(rot.conjugate().y(), rm_rot.conjugate().y, epsilon);
    EXPECT_NEAR(rot.conjugate().z(), rm_rot.conjugate().z, epsilon);

    // rotation matrix
    EXPECT_NEAR(rot.toRotationMatrix()(0, 0), rm_rot.toRotationMatrix()(0, 0), epsilon);
    EXPECT_NEAR(rot.toRotationMatrix()(0, 1), rm_rot.toRotationMatrix()(0, 1), epsilon);
    EXPECT_NEAR(rot.toRotationMatrix()(0, 2), rm_rot.toRotationMatrix()(0, 2), epsilon);

    EXPECT_NEAR(rot.toRotationMatrix()(1, 0), rm_rot.toRotationMatrix()(1, 0), epsilon);
    EXPECT_NEAR(rot.toRotationMatrix()(1, 1), rm_rot.toRotationMatrix()(1, 1), epsilon);
    EXPECT_NEAR(rot.toRotationMatrix()(1, 2), rm_rot.toRotationMatrix()(1, 2), epsilon);

    EXPECT_NEAR(rot.toRotationMatrix()(2, 0), rm_rot.toRotationMatrix()(2, 0), epsilon);
    EXPECT_NEAR(rot.toRotationMatrix()(2, 1), rm_rot.toRotationMatrix()(2, 1), epsilon);
    EXPECT_NEAR(rot.toRotationMatrix()(2, 2), rm_rot.toRotationMatrix()(2, 2), epsilon);

    // multiplication
    Eigen::Vector3d norm_vec(Random_Quaternion().x(), Random_Quaternion().y(), Random_Quaternion().z());
    Eigen::Vector3d C = rot.conjugate() * norm_vec;

    rmagine::Vector3d rm_norm_vec(norm_vec.x(), norm_vec.y(), norm_vec.z());
    EXPECT_NEAR(norm_vec.x(), rm_norm_vec.x, epsilon);
    EXPECT_NEAR(norm_vec.y(), rm_norm_vec.y, epsilon);
    EXPECT_NEAR(norm_vec.z(), rm_norm_vec.z, epsilon);

    rmagine::Vector3d rm_C = rm_rot.conjugate() * rm_norm_vec;

    EXPECT_NEAR(C.x(), rm_C.x, epsilon);
    EXPECT_NEAR(C.y(), rm_C.y, epsilon);
    EXPECT_NEAR(C.z(), rm_C.z, epsilon);

    // normalize
    Eigen::Quaterniond norm_quat = Random_Quaternion();
    norm_quat.w() += distrib(gen);
    norm_quat.x() += distrib(gen);
    norm_quat.y() += distrib(gen);
    norm_quat.z() += distrib(gen);
    rmagine::Quaterniond rm_norm_quat(norm_quat.w(), norm_quat.x(), norm_quat.y(), norm_quat.z());

    EXPECT_NEAR(norm_quat.normalized().w(), rm_norm_quat.normalized().w, epsilon);
    EXPECT_NEAR(norm_quat.normalized().x(), rm_norm_quat.normalized().x, epsilon);
    EXPECT_NEAR(norm_quat.normalized().y(), rm_norm_quat.normalized().y, epsilon);
    EXPECT_NEAR(norm_quat.normalized().z(), rm_norm_quat.normalized().z, epsilon);

    // reinterpret cast
    /*norm_quat = Random_Quaternion();
    rmagine::Quaterniond rm_casted = *(reinterpret_cast<rmagine::Quaterniond*>(&norm_quat));

    EXPECT_NEAR(norm_quat.w(), rm_casted.w, epsilon);
    EXPECT_NEAR(norm_quat.x(), rm_casted.x, epsilon);
    EXPECT_NEAR(norm_quat.y(), rm_casted.y, epsilon);
    EXPECT_NEAR(norm_quat.z(), rm_casted.z, epsilon);*/
}

TEST(rm_mtk_interop, vector3)
{
    // reinterpret cast
    auto norm_quat = Random_Quaternion();
    vect3 t(Eigen::Vector3d(norm_quat.x(), norm_quat.y(), norm_quat.z()));
    rmagine::Vector3d rm_casted = *(reinterpret_cast<rmagine::Vector3d *>(&t));

    EXPECT_NEAR(t.x(), rm_casted.x, epsilon);
    EXPECT_NEAR(t.y(), rm_casted.y, epsilon);
    EXPECT_NEAR(t.z(), rm_casted.z, epsilon);
}

void jacobian_test_cpu(double *data, Eigen::MatrixXd *target, int i)
{
    assert(target != nullptr);
    assert(target->data() != nullptr);

    int h_x_rows = target->rows();
    int h_x_cols = target->cols();

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> h_x(target->data(), h_x_rows, h_x_cols);
    h_x.block<1, 12>(i, 0) << data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11];
}

void jacobian_test_gpu(double *data, int rows, int cols, Eigen::MatrixXd &target, int i)
{
    assert(data != nullptr);

    rmagine::MatrixXd mdata_host = jacobian_test_gpu_internal(data, rows, cols, i);

    assert(mdata_host(i, 2) > 1.1);
    assert(mdata_host(i, 2) < 1.3);
    assert(mdata_host(i, 3) > 1.2);

    target = toEigen(mdata_host);
}

TEST(rm_mtk_interop, eigen_block_map_cpu)
{
    Eigen::MatrixXd *h_x_p = new Eigen::MatrixXd(1234, 12);
    double data[] = {1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11};

    int idx = 0;
    jacobian_test_cpu(data, h_x_p, idx);

    Eigen::MatrixXd h_x = *h_x_p;
    delete h_x_p;

    EXPECT_NEAR(data[0], h_x(idx, 0), epsilon);
    EXPECT_NEAR(data[1], h_x(idx, 1), epsilon);
    EXPECT_NEAR(data[2], h_x(idx, 2), epsilon);
    EXPECT_NEAR(data[3], h_x(idx, 3), epsilon);
    EXPECT_NEAR(data[4], h_x(idx, 4), epsilon);
    EXPECT_NEAR(data[5], h_x(idx, 5), epsilon);
    EXPECT_NEAR(data[6], h_x(idx, 6), epsilon);
    EXPECT_NEAR(data[7], h_x(idx, 7), epsilon);
    EXPECT_NEAR(data[8], h_x(idx, 8), epsilon);
    EXPECT_NEAR(data[9], h_x(idx, 9), epsilon);
    EXPECT_NEAR(data[10], h_x(idx, 10), epsilon);
    EXPECT_NEAR(data[11], h_x(idx, 11), epsilon);
}

TEST(rm_mtk_interop, eigen_rm_matrixXd_interop)
{
    Eigen::MatrixXd h_x_p;
    double data[] = {1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11};

    int idx = 0;

    rmagine::MatrixXd rm_h_x(1234, 12);
    rm_h_x(idx, 0) = data[0];
    rm_h_x(idx, 1) = data[1];
    rm_h_x(idx, 2) = data[2];
    rm_h_x(idx, 3) = data[3];
    rm_h_x(idx, 4) = data[4];
    rm_h_x(idx, 5) = data[5];
    rm_h_x(idx, 6) = data[6];
    rm_h_x(idx, 7) = data[7];
    rm_h_x(idx, 8) = data[8];
    rm_h_x(idx, 9) = data[9];
    rm_h_x(idx, 10) = data[10];
    rm_h_x(idx, 11) = data[11];

    // h_x_p = *(reinterpret_cast<Eigen::MatrixXd*>(&rm_h_x));
    h_x_p = toEigen(rm_h_x);

    EXPECT_NEAR(data[0], h_x_p(idx, 0), epsilon);
    EXPECT_NEAR(data[1], h_x_p(idx, 1), epsilon);
    EXPECT_NEAR(data[2], h_x_p(idx, 2), epsilon);
    EXPECT_NEAR(data[3], h_x_p(idx, 3), epsilon);
    EXPECT_NEAR(data[4], h_x_p(idx, 4), epsilon);
    EXPECT_NEAR(data[5], h_x_p(idx, 5), epsilon);
    EXPECT_NEAR(data[6], h_x_p(idx, 6), epsilon);
    EXPECT_NEAR(data[7], h_x_p(idx, 7), epsilon);
    EXPECT_NEAR(data[8], h_x_p(idx, 8), epsilon);
    EXPECT_NEAR(data[9], h_x_p(idx, 9), epsilon);
    EXPECT_NEAR(data[10], h_x_p(idx, 10), epsilon);
    EXPECT_NEAR(data[11], h_x_p(idx, 11), epsilon);

    EXPECT_NEAR(0.0, h_x_p(idx + 1, 11), epsilon);
}

TEST(rm_mtk_interop, eigen_mat_change)
{
    Eigen::MatrixXd h_x(1234, 12);

    h_x(1, 2) = 20.0;
    EXPECT_NEAR(h_x(1, 2), 20, epsilon);

    rmagine::MatrixXd mat(1234, 12);
    mat(1, 2) = 20.0;

    EXPECT_NEAR(toEigen(mat)(1, 2), 20.0, epsilon);
}

TEST(rm_mtk_interop, eigen_block_gpu)
{
    Eigen::MatrixXd h_x(1234, 12);
    double data[] = {1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11};

    int idx = 0;
    jacobian_test_gpu(data, 1234, 12, h_x, idx);

    EXPECT_NEAR(data[0], h_x(idx, 0), epsilon);
    EXPECT_NEAR(data[1], h_x(idx, 1), epsilon);
    EXPECT_NEAR(data[2], h_x(idx, 2), epsilon);
    EXPECT_NEAR(data[3], h_x(idx, 3), epsilon);
    EXPECT_NEAR(data[4], h_x(idx, 4), epsilon);
    EXPECT_NEAR(data[5], h_x(idx, 5), epsilon);
    EXPECT_NEAR(data[6], h_x(idx, 6), epsilon);
    EXPECT_NEAR(data[7], h_x(idx, 7), epsilon);
    EXPECT_NEAR(data[8], h_x(idx, 8), epsilon);
    EXPECT_NEAR(data[9], h_x(idx, 9), epsilon);
    EXPECT_NEAR(data[10], h_x(idx, 10), epsilon);
    EXPECT_NEAR(data[11], h_x(idx, 11), epsilon);
}
