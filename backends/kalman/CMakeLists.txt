cmake_minimum_required(VERSION 3.10)
project(kalman VERSION 0.1.0)

add_library(kalman INTERFACE)
target_include_directories(kalman INTERFACE
  include
  ${EIGEN3_INCLUDE_DIR}
  ${BOOST_INCLUDE_DIRS}
)

target_link_libraries(kalman INTERFACE 
  Eigen3::Eigen
  OpenMP::OpenMP_CXX
  Boost::boost
)

if(WITH_TESTS)
  add_executable(ekfom_test src/main.cpp)
  target_link_libraries(ekfom_test kalman)
endif()

install(DIRECTORY include
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)
