#pragma once

#include <vector>
#include <cassert>
#include <kernel/memory.h>
#include <kernel/common.h>

namespace rmagine
{
    template <typename T>
    struct MatrixXxX
    {
        T *m_data;
        int m_numRows;
        int m_numCols;

        MatrixXxX() : m_numRows(0), m_numCols(0) {}

        ~MatrixXxX()
        {
            if (m_data != nullptr)
            {
                jlio::free(m_data);
                m_data = nullptr;
            }
        }

        MatrixXxX(int numRows, int numCols) : m_numRows(numRows), m_numCols(numCols)
        {
            jlio::malloc((void**)&m_data, sizeof(T) * m_numRows * m_numCols);
        }

        JLIO_INLINE_FUNCTION
        int numRows() const { return m_numRows; }

        JLIO_INLINE_FUNCTION
        int numCols() const { return m_numCols; }

        JLIO_INLINE_FUNCTION
        T &operator()(int row, int col)
        {
            assert(row >= 0 && row < m_numRows);
            assert(col >= 0 && col < m_numCols);
            return m_data[row + col * m_numRows];
        }

        JLIO_INLINE_FUNCTION
        const T &operator()(int row, int col) const
        {
            assert(row >= 0 && row < m_numRows);
            assert(col >= 0 && col < m_numCols);
            return m_data[row + col * m_numRows];
        }
    };

    using MatrixXd = MatrixXxX<double>;
    using MatrixXf = MatrixXxX<float>;

}
