cmake_minimum_required(VERSION 3.17)
project(jlio-backends VERSION 0.1.0)

find_package(Threads)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(WITH_CUDA "With CUDA support" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

find_package(CUDAToolkit)
if(WITH_CUDA AND CUDAToolkit_FOUND)
    include(CheckLanguage)
    check_language(CUDA REQUIRED)
 
    set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE}) # override the CUDA version to the one found by the toolkit package
    message(STATUS "-::--::- CUDA support enabled -::--::-")

    add_definitions(-DUSE_CUDA) # used in the source code
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
      set(CMAKE_CUDA_STANDARD 14)
      set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()

    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
      message(STATUS "Running on ARM, assuming a NVIDIA Jetson and setting CUDA target include directories.")
      set(CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS} /usr/local/cuda/targets/${CMAKE_SYSTEM_PROCESSOR}-linux/include/)
    endif()

    set(CMAKE_CUDA_ARCHITECTURES 72)

    enable_language(CUDA)
else()
    if(WITH_THREADING)
        message(STATUS "-::--::- CPU only [all cores] -::--::-")
    else()
        message(STATUS "-::--::- CPU only [single core] -::--::-")
    endif()
endif()

find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
find_package(Boost REQUIRED COMPONENTS filesystem)

add_subdirectory(kalman)
add_subdirectory(kernel)
add_subdirectory(tree)

if (WITH_TESTS)
  add_subdirectory(tests)
endif()
