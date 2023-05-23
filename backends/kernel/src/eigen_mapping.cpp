#include <kernel/eigen_mapping.h>
/**
 * Copies(!) to a Eigen dynamic matrix.
 * TODO change to reinterpret castable structure
 */

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

/*
template <typename T>
const Eigen::Matrix<T, Eigen::Dynamic, 1> toEigen_x(const rmagine::MatrixXxX<T> &source)
{
    assert(m_numCols == 1);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> target(m_numRows, m_numCols);
    for (int col = 0; col < m_numCols; ++col)
    {
        for (int row = 0; row < m_numRows; ++row)
        {
            target(row, col) = mat()(row, col);
        }
    }
    return std::move(target);
}*/
