cmake_minimum_required(VERSION 3.12)
project(tbb VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(CMAKE_BUILD_TYPE "Release")
option(TBB_TEST OFF)
option(TBB_EXAMPLES OFF)

add_subdirectory(oneTBB)