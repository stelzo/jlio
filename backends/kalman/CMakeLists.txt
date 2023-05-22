cmake_minimum_required(VERSION 3.10)
project(kalman VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

find_package(Eigen3 REQUIRED)

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  include
)

add_library(kalman INTERFACE)
target_include_directories(kalman INTERFACE include/kalman.h)
target_link_libraries(kalman INTERFACE Eigen3::Eigen OpenMP::OpenMP_CXX)

option(BUILD_TEST "Build test" OFF)

if(BUILD_TEST)
add_executable(ekfom_test src/main.cpp)
target_link_libraries(ekfom_test kalman)
endif()

install(DIRECTORY include
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)
