#include <kernel/math/matrixXxX.h>
#include <Eigen/Dense>

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> toEigen(const rmagine::MatrixXxX<T> &source);
