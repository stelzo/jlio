#pragma once

#include "memory.h"
#include "point.h"
#include <limits> // for infinity
#include <math.h>
#include <kernel/common.h>

#define K_NEAREST = 5; // TODO still hardcoded in kernels

template <typename PointType>
struct KD_TREE_NODE
{
    PointType point;
    int division_axis;
    int TreeSize = 1;
    int invalid_point_num = 0;
    int down_del_num = 0;
    bool point_deleted = false;
    bool tree_deleted = false;
    bool point_downsample_deleted = false;
    bool tree_downsample_deleted = false;
    bool need_push_down_to_left = false;
    bool need_push_down_to_right = false;
    bool working_flag = false;
    pthread_mutex_t push_down_mutex_lock;
    float node_range_x[2], node_range_y[2], node_range_z[2];
    float radius_sq;
    KD_TREE_NODE *left_son_ptr = nullptr;
    KD_TREE_NODE *right_son_ptr = nullptr;
    KD_TREE_NODE *father_ptr = nullptr;
    // For paper data record
    float alpha_del;
    float alpha_bal;
};

struct PointType_CMP_GPU
{
    jlio::PointXYZINormal point;
    float dist = 0.0;

    JLIO_FUNCTION
    PointType_CMP_GPU()
    {
        point.x = 0;
        point.y = 0;
        point.z = 0;
        dist = 0.0;
    }

    JLIO_FUNCTION
    PointType_CMP_GPU(jlio::PointXYZINormal p, float d = std::numeric_limits<float>::infinity())
    {
        this->point = p;
        this->dist = d;
    };

    JLIO_FUNCTION
    bool operator<(const PointType_CMP_GPU &a) const
    {
        if (fabs(dist - a.dist) < 1e-10)
            return point.x < a.point.x;
        else
            return dist < a.dist;
    }
};

struct MANUAL_HEAP_GPU
{
    PointType_CMP_GPU heap[10]; // knearest * 2
    int heap_size;
    int cap;

    JLIO_FUNCTION
    MANUAL_HEAP_GPU()
    {
        init();
    }

    JLIO_FUNCTION
    void init()
    {
        heap_size = 0;
        cap = 10;
    }

    JLIO_FUNCTION
    void pop()
    {
        if (heap_size == 0)
            return;
        heap[0] = heap[heap_size - 1];
        heap_size--;
        MoveDown(0);
        return;
    }

    JLIO_FUNCTION
    PointType_CMP_GPU top()
    {
        return heap[0];
    }

    JLIO_FUNCTION
    void push(PointType_CMP_GPU point)
    {
        if (heap_size >= cap)
            return;
        heap[heap_size] = point;
        FloatUp(heap_size);
        heap_size++;
        return;
    }

    JLIO_FUNCTION
    int size()
    {
        return heap_size;
    }

    JLIO_FUNCTION
    void clear()
    {
        heap_size = 0;
        return;
    }

    JLIO_FUNCTION
    void MoveDown(int heap_index)
    {
        int l = heap_index * 2 + 1;
        PointType_CMP_GPU tmp = heap[heap_index];
        while (l < heap_size)
        {
            if (l + 1 < heap_size && heap[l] < heap[l + 1])
                l++;
            if (tmp < heap[l])
            {
                heap[heap_index] = heap[l];
                heap_index = l;
                l = heap_index * 2 + 1;
            }
            else
                break;
        }
        heap[heap_index] = tmp;
        return;
    }

    JLIO_FUNCTION
    void FloatUp(int heap_index)
    {
        int ancestor = (heap_index - 1) / 2;
        PointType_CMP_GPU tmp = heap[heap_index];
        while (heap_index > 0)
        {
            if (heap[ancestor] < tmp)
            {
                heap[heap_index] = heap[ancestor];
                heap_index = ancestor;
                ancestor = (heap_index - 1) / 2;
            }
            else
                break;
        }
        heap[heap_index] = tmp;
        return;
    }
};

JLIO_FUNCTION
void Nearest_Search(void *root, jlio::PointXYZINormal point, size_t k_nearest,
                    jlio::PointXYZINormal *Nearest_Points, int *Nearest_Points_Size,
                    float *Point_Distance, size_t *Point_Distance_Size,
                    float max_dist);

void Raw_Nearest_Search(void *root, jlio::PointXYZINormal *point, size_t k_nearest,
                        jlio::PointXYZINormal *Nearest_Points, int *Nearest_Points_Size,
                        float *Point_Distance, size_t *Point_Distance_Size,
                        float max_dist);