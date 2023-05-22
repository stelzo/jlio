#pragma once

#include <iostream>

#ifdef USE_CUDA
#include <cuda_runtime.h>

struct __align__(16) PointXYZINormal_CUDA
{
  inline __host__ __device__ PointXYZINormal_CUDA() {}
  inline __host__ __device__ PointXYZINormal_CUDA(float _x, float _y, float _z, float _intensity, float4 _normal) : x(_x), y(_y), z(_z), intensity(_intensity), normal(_normal) {}

  // Declare a union for XYZI
  union
  {
    float4 xyzi;

    struct
    {
      float x;
      float y;
      float z;
      float intensity;
    };
  };
  union
  {
    float4 normal;
    struct
    {
      float normal_x;
      float normal_y;
      float normal_z;
      float curvature;
    };
  };

  inline __host__ __device__ bool operator==(const PointXYZINormal_CUDA &rhs)
  {
    return (x == rhs.x && y == rhs.y && z == rhs.z && intensity == rhs.intensity && normal_x == rhs.normal_x && normal_y == rhs.normal_y && normal_z == rhs.normal_z);
  }

  __device__ friend std::ostream &operator<<(std::ostream &os, const PointXYZINormal_CUDA &p);
};
#else
#include <pcl/point_types.h>
typedef pcl::PointXYZINormal PointXYZINormal_CUDA;
#endif