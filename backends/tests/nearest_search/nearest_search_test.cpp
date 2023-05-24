#include <gtest/gtest.h>
#include <tree/tree.h>
#include <kernel/common.h>
#include <kernel/point_fitting.h>
#include <kernel/math/math.h>
#include <kernel/point.h>
#include <tree/tree.h>
#include <iostream>

jlio::PointXYZINormal createPoint(float x, float y, float z) {
    jlio::PointXYZINormal p;
    p.x = x;
    p.y = y;
    p.z = z;
    return p;
}

float calc_dist(jlio::PointXYZINormal a, jlio::PointXYZINormal b) {
    float dist = 0.0f;
    dist = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z);
    return dist;
}

// Important: Initialize globally to prevent Segfault
KD_TREE<jlio::PointXYZINormal> ikdTree;

TEST(ikdTree, nearestSearch) {
    constexpr size_t k = 6;

    // Define pointcloud
    std::vector<jlio::PointXYZINormal> h_points = {
        // Points outside of range
        createPoint(2, 2, 0),
        createPoint(2, -2, 0),
        createPoint(-2, 2, 0),
        createPoint(-2, -2, 0),
        createPoint(0, 2, 2),
        createPoint(0, 2, -2),
        createPoint(0, -2, 2),
        createPoint(0, -2, -2),
        createPoint(2, 0, 2),
        createPoint(2, 0, -2),
        createPoint(-2, 0, 2),
        createPoint(-2, 0, -2),
        // Points inside of range
        // Use odd numbers to compare the points better via sorting
        createPoint(1.1, 0, 0),
        createPoint(-1.2, 0, 0),
        createPoint(0, 1.3, 0),
        createPoint(0, -1.4, 0),
        createPoint(0, 0, 1.5),
        createPoint(0, 0, -1.6),
    };

    // Initialize GPU memory
    jlio::PointXYZINormal* managed_points = nullptr;
    jlio::malloc((void**)&managed_points, h_points.size() * sizeof(jlio::PointXYZINormal));
    for (size_t i = 0; i < h_points.size(); i++) {
        managed_points[i] = jlio::PointXYZINormal();
        managed_points[i].x = h_points[i].x;
        managed_points[i].y = h_points[i].y;
        managed_points[i].z = h_points[i].z;
    }

    std::vector<jlio::PointXYZINormal> d_points;
    d_points.insert(d_points.end(), &managed_points[0], &managed_points[h_points.size()]);

    ikdTree.Build(d_points);
    //ikdTree.consistent();

    jlio::PointXYZINormal query;
    query.x = 0;
    query.y = 0;
    query.z = 0;

    jlio::PointXYZINormal* d_query;
    jlio::malloc((void**)&d_query, sizeof(jlio::PointXYZINormal));
    d_query[0] = query;

    jlio::PointXYZINormal* Nearest_Points;
    jlio::malloc((void**)&Nearest_Points, k * sizeof(jlio::PointXYZINormal));
    for (size_t i = 0; i < k; i++) {
        Nearest_Points[i] = jlio::PointXYZINormal();
    }

    int* nearest_size;
    jlio::malloc((void**)&nearest_size, sizeof(int));
    nearest_size[0] = 0;

    float* pointSearchSqDis;
    jlio::malloc((void**)&pointSearchSqDis, k * sizeof(float));

    size_t* dist_size;
    jlio::malloc((void**)&dist_size, sizeof(size_t));
    dist_size[0] = k;

    // Perform search
    test_nearest_search((void*)ikdTree.Root_Node, d_query, k, Nearest_Points, nearest_size,
                       pointSearchSqDis, dist_size, 1e6);

    EXPECT_EQ(*nearest_size, k);

    // Sort nearest points by distance to query (Bubblesort)
    for (int i = 0; i < *nearest_size; i++) {
        for (int j = 0; j < *nearest_size - i - 1; j++) {
            if (calc_dist(Nearest_Points[j], query) > calc_dist(Nearest_Points[j + 1], query)) {
                auto dummy = Nearest_Points[j];
                Nearest_Points[j] = Nearest_Points[j + 1];
                Nearest_Points[j + 1] = dummy;
            }
        }
    }

    // Check output
    EXPECT_NEAR(Nearest_Points[0].x, 1.1, 0.01);
    EXPECT_NEAR(Nearest_Points[1].x, -1.2, 0.01);
    EXPECT_NEAR(Nearest_Points[2].y, 1.3, 0.01);
    EXPECT_NEAR(Nearest_Points[3].y, -1.4, 0.01);
    EXPECT_NEAR(Nearest_Points[4].z, 1.5, 0.01);
    EXPECT_NEAR(Nearest_Points[5].z, -1.6, 0.01);

    jlio::free(managed_points);
    jlio::free(d_query);
    jlio::free(Nearest_Points);
    jlio::free(nearest_size);
    jlio::free(pointSearchSqDis);
    jlio::free(dist_size);
}
