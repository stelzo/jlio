#pragma once

#include <kernel/math/matrixXxX.h>
#include <Eigen/Dense>

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> toEigen(const rmagine::MatrixXxX<T> &mat)
{
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> target(mat.m_numRows, mat.m_numCols);
    for (int col = 0; col < mat.m_numCols; ++col)
    {
        for (int row = 0; row < mat.m_numRows; ++row)
        {
            target(row, col) = mat(row, col);
        }
    }
    return std::move(target);
}
