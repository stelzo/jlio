#pragma once

#include <iostream>

#ifdef USE_CUDA
#include <cuda_runtime.h>
namespace jlio
{
struct __align__(16) PointXYZINormal
{
  inline __host__ __device__ PointXYZINormal() {}
  inline __host__ __device__ PointXYZINormal(float _x, float _y, float _z, float _intensity, float4 _normal) : x(_x), y(_y), z(_z), intensity(_intensity), normal(_normal) {}

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

  inline __host__ __device__ bool operator==(const PointXYZINormal &rhs)
  {
    return (x == rhs.x && y == rhs.y && z == rhs.z && intensity == rhs.intensity && normal_x == rhs.normal_x && normal_y == rhs.normal_y && normal_z == rhs.normal_z);
  }

  __device__ friend std::ostream &operator<<(std::ostream &os, const PointXYZINormal &p);
};

struct __align__(16) OusterPoint
{
  inline __host__ __device__ OusterPoint() {}
  inline __host__ __device__ OusterPoint(float _x, float _y, float _z, float _intensity, uint32_t _t, uint16_t _reflectivity, uint8_t _ring, uint16_t _ambient, uint32_t _range) : x(_x), y(_y), z(_z), intensity(_intensity), t(_t), reflectivity(_reflectivity), ring(_ring), ambient(_ambient), range(_range) {}

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
  float intensity; // TODO should be inside the union, but in original code it is not
  uint32_t t;
  uint16_t reflectivity;
  uint8_t  ring;
  uint16_t ambient;
  uint32_t range;

};
}


#else
#include <pcl/point_types.h>
namespace jlio
{
typedef pcl::PointXYZINormal PointXYZINormal;

struct EIGEN_ALIGN16 OusterPoint
{
  PCL_ADD_POINT4D;
  float intensity;
  uint32_t t;
  uint16_t reflectivity;
  uint8_t  ring;
  uint16_t ambient;
  uint32_t range;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}
#endif