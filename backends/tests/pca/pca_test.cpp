#include <gtest/gtest.h>
#include <kernel/math/math.h>
#include <kernel/pca.h>
#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <iostream>

rmagine::Vector3d z1_points[] = {{0, 0, 0}, {-2, 0, 0}, {0, 1, 0}, {3, 1, 0}, {2, -2, 0}};

rmagine::Vector3d y1_points[] = {{0, 0, 2}, {-2, 0, 0}, {0, 0, 0}, {3, 0, -1}, {2, 0, 1}};

rmagine::Vector3d hard_plane[] = {{2, 1, 1}, {1, 1, 0}, {0, 1, 0}, {-2, 1.05, 0}, {-3, 1, 0}};

rmagine::Vector3d hard_plane_1off[] = {{2, 1, 1}, {1, 1, 0}, {0, 1, 0}, {-2, 5, 0}, {-3, 1, 0}};

rmagine::Vector3d hard_plane_1off_marginal[] = {{2, 1, 1}, {1, 1, 0}, {0, 1, 0}, {-2, 1.105, 0}, {-3, 1, 0}};

float threshold = 0.1; // same as in fast_lio implementation

// Original version from FAST_LIO2
#define NUM_MATCH_POINTS 5

typedef std::vector<pcl::PointXYZI> PointVector;

template<typename T>
bool esti_plane(Eigen::Matrix<T, 4, 1> &pca_result, const PointVector &point, const T &threshold)
{
    Eigen::Matrix<T, NUM_MATCH_POINTS, 3> A;
    Eigen::Matrix<T, NUM_MATCH_POINTS, 1> b;
    A.setZero();
    b.setOnes();
    b *= -1.0f;

    for (int j = 0; j < NUM_MATCH_POINTS; j++)
    {
        A(j,0) = point[j].x;
        A(j,1) = point[j].y;
        A(j,2) = point[j].z;
    }

    Eigen::Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);

    T n = normvec.norm();
    pca_result(0) = normvec(0) / n;
    pca_result(1) = normvec(1) / n;
    pca_result(2) = normvec(2) / n;
    pca_result(3) = 1.0 / n;

    for (int j = 0; j < NUM_MATCH_POINTS; j++)
    {
        float to_check = fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z + pca_result(3));
        if (to_check > threshold)
        {
            return false;
        }
    }
    return true;
}

TEST(pca, ilikebigbits_hard_plane) {
    rmagine::Vector3d normal;
    rmagine::VectorN<4, double> coeffs;

    bool result = pca_constant(hard_plane, sizeof(hard_plane) / sizeof(hard_plane[0]),
                                       threshold, &normal, &coeffs);
    EXPECT_TRUE(result);
}

TEST(pca, ilikebigbits_hard_plane_1off) {
    rmagine::Vector3d normal;
    rmagine::VectorN<4, double> coeffs;

    bool result = pca_constant(hard_plane_1off, sizeof(hard_plane_1off) / sizeof(hard_plane_1off[0]),
                                       threshold, &normal, &coeffs);
    EXPECT_FALSE(result);
}

TEST(pca, ilikebigbits_hard_plane_1off_marginal) {
    rmagine::Vector3d normal;
    rmagine::VectorN<4, double> coeffs;

    bool result = pca_constant(hard_plane_1off, sizeof(hard_plane_1off) / sizeof(hard_plane_1off[0]),
                                       threshold, &normal, &coeffs);
    EXPECT_FALSE(result);
}

TEST(pca, custom_iterative_z1) {
    rmagine::Vector3d normal;
    rmagine::VectorN<4, double> coeffs;

    bool result = pca_custom_iterative(z1_points, sizeof(z1_points) / sizeof(z1_points[0]),
                                       threshold, &normal, &coeffs);
    EXPECT_TRUE(result);

    EXPECT_NEAR(normal.x, 0.0, 0.01);
    EXPECT_NEAR(normal.y, 0.0, 0.01);
    EXPECT_NEAR(normal.z, 1.0, 0.01);
}

TEST(pca, custom_iterative_y1) {
    rmagine::Vector3d normal;
    rmagine::VectorN<4, double> coeffs;

    bool result = pca_custom_iterative(y1_points, sizeof(y1_points) / sizeof(y1_points[0]),
                                       threshold, &normal, &coeffs);
    EXPECT_TRUE(result);

    EXPECT_NEAR(normal.x, 0.0, 0.01);
    EXPECT_NEAR(normal.y, 1.0, 0.01);
    EXPECT_NEAR(normal.z, 0.0, 0.01);
}

TEST(pca, ilikebigbits_z1) {
    rmagine::Vector3d normal;
    rmagine::VectorN<4, double> coeffs;

    bool result =
        pca_constant(z1_points, sizeof(z1_points) / sizeof(z1_points[0]), threshold, &normal, &coeffs);
    EXPECT_TRUE(result);

    EXPECT_NEAR(normal.x, 0.0, 0.01);
    EXPECT_NEAR(normal.y, 0.0, 0.01);
    EXPECT_NEAR(normal.z, 1.0, 0.01);
}

TEST(pca, ilikebigbits_y1) {
    rmagine::Vector3d normal;
    rmagine::VectorN<4, double> coeffs;

    bool result =
        pca_constant(y1_points, sizeof(y1_points) / sizeof(y1_points[0]), threshold, &normal, &coeffs);
    EXPECT_TRUE(result);

    EXPECT_NEAR(normal.x, 0.0, 0.01);
    EXPECT_NEAR(normal.y, 1.0, 0.01);
    EXPECT_NEAR(normal.z, 0.0, 0.01);
}

TEST(pca, fast_lio_hard_plane) {
    PointVector z1_points_eigen;
    z1_points_eigen.resize(sizeof(hard_plane) / sizeof(hard_plane[0]));
    for (size_t i = 0; i < sizeof(hard_plane) / sizeof(hard_plane[0]); i++) {
        z1_points_eigen[i].x = hard_plane[i].x;
        z1_points_eigen[i].y = hard_plane[i].y;
        z1_points_eigen[i].z = hard_plane[i].z;
    }

    Eigen::Matrix<float, 4, 1> normal = Eigen::Matrix<float, 4, 1>::Zero();

    bool result = esti_plane(normal, z1_points_eigen, threshold);
    EXPECT_TRUE(result);
}

TEST(pca, fast_lio_hard_plane_1off) {
    PointVector z1_points_eigen;
    z1_points_eigen.resize(sizeof(hard_plane_1off) / sizeof(hard_plane_1off[0]));
    for (size_t i = 0; i < sizeof(hard_plane_1off) / sizeof(hard_plane_1off[0]); i++) {
        z1_points_eigen[i].x = hard_plane_1off[i].x;
        z1_points_eigen[i].y = hard_plane_1off[i].y;
        z1_points_eigen[i].z = hard_plane_1off[i].z;
    }

    Eigen::Matrix<float, 4, 1> normal = Eigen::Matrix<float, 4, 1>::Zero();

    bool result = esti_plane(normal, z1_points_eigen, threshold);
    EXPECT_FALSE(result);
}


/*
TEST(pca, fast_lio_z1) {
    PointVector z1_points_eigen;
    z1_points_eigen.resize(sizeof(z1_points) / sizeof(z1_points[0]));
    for (size_t i = 0; i < sizeof(z1_points) / sizeof(z1_points[0]); i++) {
        z1_points_eigen[i].x = z1_points[i].x;
        z1_points_eigen[i].y = z1_points[i].y;
        z1_points_eigen[i].z = z1_points[i].z;
    }

    Eigen::Matrix<float, 4, 1> normal = Eigen::Matrix<float, 4, 1>::Zero();

    bool result = esti_plane(normal, z1_points_eigen, threshold);
    EXPECT_TRUE(result);

    EXPECT_NEAR(normal.x(), 0.0, 0.01);
    EXPECT_NEAR(normal.y(), 0.0, 0.01);
    EXPECT_NEAR(normal.z(), 1.0, 0.01);
    EXPECT_NEAR(normal.w(), 0.0, 0.01);
}

TEST(pca, fast_lio_y1) {
    PointVector z1_points_eigen;
    z1_points_eigen.resize(sizeof(y1_points) / sizeof(y1_points[0]));
    for (int i = 0; i < sizeof(y1_points) / sizeof(y1_points[0]); i++) {
        z1_points_eigen[i].x = y1_points[i].x;
        z1_points_eigen[i].y = y1_points[i].y;
        z1_points_eigen[i].z = y1_points[i].z;
    }

    Eigen::Matrix<float, 4, 1> normal = Eigen::Matrix<float, 4, 1>::Zero();

    bool result = esti_plane(normal, z1_points_eigen, threshold);
    EXPECT_TRUE(result) << "result = " << std::boolalpha << result << std::endl;

    EXPECT_NEAR(normal.x(), 0.0, 0.01) << "normal.x() = " << normal.x() << std::endl;
    EXPECT_NEAR(normal.y(), 1.0, 0.01) << "normal.y() = " << normal.y() << std::endl;
    EXPECT_NEAR(normal.z(), 0.0, 0.01) << "normal.z() = " << normal.z() << std::endl;
    EXPECT_NEAR(normal.w(), 0.0, 0.01) << "normal.w() = " << normal.w() << std::endl;
}
*/

/*
TEST(pca, fast_lio_z1_stress)
{
    PointVector z1_points_eigen;
    z1_points_eigen.resize(sizeof(y1_points));
    for (int i = 0; i < sizeof(y1_points); i++)
    {
        z1_points_eigen[i].x = y1_points[i].x;
        z1_points_eigen[i].y = y1_points[i].y;
        z1_points_eigen[i].z = y1_points[i].z;
    }

    Eigen::Matrix<float, 4, 1> normal;

    size_t size = 1000000;

    for (size_t i = 0; i < size; i++) {
        bool result = esti_plane(normal, z1_points_eigen, threshold);
    }

    EXPECT_TRUE(result);
    EXPECT_NEAR(normal.x(), 0.0, 0.01);
    EXPECT_NEAR(normal.y(), 0.0, 0.01);
    EXPECT_NEAR(normal.z(), 1.0, 0.01);
    EXPECT_NEAR(normal.w(), 0.0, 0.01);
}

TEST(pca, ilikebigbits_z1_stress)
{
    rmagine::Vector3d normal;

    size_t size = 1000000;

    for (size_t i = 0; i < size; i++) {
        bool result = pca_constant(z1_points, sizeof(z1_points), threshold, &normal);
    }

    EXPECT_TRUE(result);
    EXPECT_NEAR(normal.x, 0.0, 0.01);
    EXPECT_NEAR(normal.y, 0.0, 0.01);
    EXPECT_NEAR(normal.z, 1.0, 0.01);
}

TEST(pca, custom_iterative_z1_stress)
{
    rmagine::Vector3d normal;

    size_t size = 1000000;

    for (size_t i = 0; i < size; i++) {
        bool result = pca_custom_iterative(z1_points, sizeof(z1_points), threshold, &normal);

    }

    EXPECT_TRUE(result);
    EXPECT_NEAR(normal.x, 0.0, 0.01);
    EXPECT_NEAR(normal.y, 0.0, 0.01);
    EXPECT_NEAR(normal.z, 1.0, 0.01);
}
*/