#include <kernel/point.h>

/**
 * @brief Filter points from ouster lidar
 * 
 * @param source the byte buffer from PointCloud2 on CPU
 * @param source_size the size of the original point cloud
 * @param point_step the size of each point. This is the same as the point_step in PointCloud2.
 * @param points_count the number of points in the original point cloud
 * @param out the output buffer on GPU, will be allocated by this function
 * @param out_size the number of points in the output buffer
 * @param near_dist the minimum distance of the points to be kept
 * @param far_dist the maximum distance of the points to be kept
*/
void filter_map_ouster(const u_int8_t* source, uint32_t source_size, uint32_t point_step, jlio::PointXYZINormal* out, uint32_t* out_size, float near_dist, float far_dist);
