cmake_minimum_required(VERSION 3.12)
project(gtest VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

option(INSTALL_GTEST OFF)
option(gtest_disable_pthreads ON)
option(BUILD_GMOCK OFF)

add_subdirectory(googletest)