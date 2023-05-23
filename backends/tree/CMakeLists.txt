cmake_minimum_required(VERSION 3.10)
project(tree VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)

find_package(PCL 1.8 REQUIRED QUIET)

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  include
  ../kernel/include
)

add_library(ikdtree
  src/tree.cpp
)
target_link_libraries(ikdtree
  cuda
  ${CMAKE_THREAD_LIBS_INIT}
  ${PCL_LIBRARIES}
)

add_compile_options(-std=c++17)
#add_compile_options(-Wall)
