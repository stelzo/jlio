#pragma once

#include <cuda/math/math.h>

#ifdef USE_CUDA
#include <cusolverDn.h>
#endif

RMAGINE_INLINE_FUNCTION
rmagine::Vector3d calc_centroid(const rmagine::Vector3d *cluster, size_t cluster_size)
{
    rmagine::Vector3d sum(0, 0, 0);
    for (size_t i = 0; i < cluster_size; i++)
    {
        sum.addInplace(cluster[i]);
    }

    return sum * (1.0 / static_cast<double>(cluster_size));
}

RMAGINE_INLINE_FUNCTION
rmagine::Matrix3x3d calc_cov(const rmagine::Vector3d *cluster, size_t cluster_size, const rmagine::Vector3d &centroid)
{
    double xx = 0.0;
    double xy = 0.0;
    double xz = 0.0;
    double yy = 0.0;
    double yz = 0.0;
    double zz = 0.0;

    for (size_t i = 0; i < cluster_size; i++)
    {
        rmagine::Vector3d r = cluster[i] - centroid;
        xx += r.x * r.x;
        xy += r.x * r.y;
        xz += r.x * r.z;
        yy += r.y * r.y;
        yz += r.y * r.z;
        zz += r.z * r.z;
    }

    xx /= static_cast<double>(cluster_size);
    xy /= static_cast<double>(cluster_size);
    xz /= static_cast<double>(cluster_size);
    yy /= static_cast<double>(cluster_size);
    yz /= static_cast<double>(cluster_size);
    zz /= static_cast<double>(cluster_size);

    rmagine::Matrix3x3d cov;
    cov(0, 0) = xx;
    cov(0, 1) = xy;
    cov(0, 2) = xz;
    cov(1, 0) = xy;
    cov(1, 1) = yy;
    cov(1, 2) = yz;
    cov(2, 0) = xz;
    cov(2, 1) = yz;
    cov(2, 2) = zz;

    return std::move(cov);
}

// https://www.ilikebigbits.com/2017_09_25_plane_from_points_2.html
RMAGINE_INLINE_FUNCTION
bool pca_constant(const rmagine::Vector3d *cluster, size_t n, float normal_threshold, rmagine::Vector3d *normal, rmagine::VectorN<4, double> *plane_equation)
{
    if (n < 3)
    {
        return false;
    }

    rmagine::Vector3d centroid = calc_centroid(cluster, n);
    rmagine::Matrix3x3d cov = calc_cov(cluster, n, centroid);

    double xx = cov(0, 0);
    double xy = cov(0, 1);
    double xz = cov(0, 2);
    double yy = cov(1, 1);
    double yz = cov(1, 2);
    double zz = cov(2, 2);

    rmagine::Vector3d weighted_dir(0, 0, 0);

    {
        double det_x = yy * zz - yz * yz;
        rmagine::Vector3d axis_dir(det_x, xz * yz - xy * zz, xy * yz - xz * yy);
        double weight = det_x * det_x;
        if (weighted_dir.dot(axis_dir) < 0.0)
        {
            weight = -weight;
        }

        weighted_dir += axis_dir * weight;
    }

    {
        double det_y = xx * zz - xz * xz;
        rmagine::Vector3d axis_dir(xz * yz - xy * zz, det_y, xy * xz - yz * xx);
        double weight = det_y * det_y;
        if (weighted_dir.dot(axis_dir) < 0.0)
        {
            weight = -weight;
        }

        weighted_dir += axis_dir * weight;
    }

    {
        double det_z = xx * yy - xy * xy;
        rmagine::Vector3d axis_dir(xy * yz - xz * yy, xy * xz - yz * xx, det_z);
        double weight = det_z * det_z;
        if (weighted_dir.dot(axis_dir) < 0.0)
        {
            weight = -weight;
        }

        weighted_dir += axis_dir * weight;
    }

    *normal = weighted_dir.normalized();

    double d = -(normal->x * centroid.x + normal->y * centroid.y + normal->z * centroid.z);

    (*plane_equation)(0) = (*normal).x;
    (*plane_equation)(1) = (*normal).y;
    (*plane_equation)(2) = (*normal).z;
    (*plane_equation)(3) = d;

    // if any of the points is too far from the plane, return false
    for (size_t i = 0; i < n; i++)
    {
        if (fabs((*plane_equation)(0) * cluster[i].x + (*plane_equation)(1) * cluster[i].y + (*plane_equation)(2) * cluster[i].z + (*plane_equation)(3)) > normal_threshold)
        {
            return false;
        }
    }

    return true;
}

/**
 * Plane Normal vector estimation using PCA
 *
 */
RMAGINE_INLINE_FUNCTION
bool pca_custom_iterative(const rmagine::Vector3d *cluster, size_t cluster_size, float normal_threshold, rmagine::Vector3d *normal, rmagine::VectorN<4, double> *plane_equation)
{
    if (cluster_size < 3)
    {
        return false;
    }

    rmagine::Vector3d cog = calc_centroid(cluster, cluster_size);
    rmagine::Matrix3x3d K = calc_cov(cluster, cluster_size, cog);

    double A[3][3] = {{K(0, 0), K(0, 1), K(0, 2)}, {K(1, 0), K(1, 1), K(1, 2)}, {K(2, 0), K(2, 1), K(2, 2)}};

    // jacobi eigen solver
    // init eigen vector matrix as identity
    double eigenvectors[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    double eigenvalues[3] = {0, 0, 0};

    int n = 3;

    // Compute the off-diagonal norm of A
    double offdiag_norm = 0.0;
    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            offdiag_norm += A[i][j] * A[i][j];
        }
    }

    // Compute the eigenvalues and eigenvectors
    int max_iter = 1000;
    for (int iter = 0; iter < max_iter; iter++)
    {
        // Find the largest off-diagonal element
        int p = 0;
        int q = 1;
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                if (fabs(A[i][j]) > fabs(A[p][q]))
                {
                    p = i;
                    q = j;
                }
            }
        }

        // Check for convergence
        double threshold = 1e-8;
        if (offdiag_norm < threshold)
        {
            break;
        }

        // Compute the Jacobi rotation angle
        double theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        double t = 1.0 / (fabs(theta) + sqrt(theta * theta + 1.0));
        if (theta < 0)
        {
            t = -t;
        }

        // Compute the Jacobi rotation matrix
        double c = 1.0 / sqrt(t * t + 1.0);
        double s = t * c;

        // Apply the Jacobi rotation
        double A_pq = A[p][q];
        A[p][q] = 0.0;
        A[q][p] = 0.0;
        A[p][p] -= t * A_pq;
        A[q][q] += t * A_pq;
        for (int r = 0; r < n; r++)
        {
            if (r != p && r != q)
            {
                double A_pr = A[p][r];
                double A_qr = A[q][r];
                A[p][r] = c * A_pr - s * A_qr;
                A[r][p] = A[p][r];
                A[q][r] = c * A_qr + s * A_pr;
                A[r][q] = A[q][r];
            }

            // Update the eigenvectors
            double eigenvector_pr = eigenvectors[p][r];
            double eigenvector_qr = eigenvectors[q][r];
            eigenvectors[p][r] = c * eigenvector_pr - s * eigenvector_qr;
            eigenvectors[q][r] = c * eigenvector_qr + s * eigenvector_pr;
        }

        // Update the off-diagonal norm
        offdiag_norm -= A_pq * A_pq;
    }

    // Sort the eigenvalues and eigenvectors
    for (int i = 0; i < n; i++)
    {
        eigenvalues[i] = A[i][i];
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            if (eigenvalues[i] > eigenvalues[j])
            {
                double tmp = eigenvalues[i];
                eigenvalues[i] = eigenvalues[j];
                eigenvalues[j] = tmp;
                for (int k = 0; k < n; k++)
                {
                    double tmp = eigenvectors[k][i];
                    eigenvectors[k][i] = eigenvectors[k][j];
                    eigenvectors[k][j] = tmp;
                }
            }
        }
    }

    // printf("eigenvalues: %lf %lf %lf \n", eigenvalues[0], eigenvalues[1], eigenvalues[2]);

    // eigen vector with smallest eigen value is the normal
    double minEigenV[3] = {eigenvectors[0][0], eigenvectors[1][0], eigenvectors[2][0]};

    // normalize
    double norm = sqrt(minEigenV[0] * minEigenV[0] + minEigenV[1] * minEigenV[1] + minEigenV[2] * minEigenV[2]);
    minEigenV[0] /= norm;
    minEigenV[1] /= norm;
    minEigenV[2] /= norm;
    double d = -(minEigenV[0] * cog.x + minEigenV[1] * cog.y + minEigenV[2] * cog.z);

    (*plane_equation)(0) = minEigenV[0];
    (*plane_equation)(1) = minEigenV[1];
    (*plane_equation)(2) = minEigenV[2];
    (*plane_equation)(3) = d;

    normal->x = (float)minEigenV[0];
    normal->y = (float)minEigenV[1];
    normal->z = (float)minEigenV[2];

    for (size_t i = 0; i < cluster_size; i++)
    {
        double x = minEigenV[0] * cluster[i].x;
        double y = minEigenV[1] * cluster[i].y;
        double z = minEigenV[2] * cluster[i].z;

        double absolute = fabs(x + y + z);
        if (absolute > normal_threshold)
        {
            return false;
        }
    }

    return true;
}