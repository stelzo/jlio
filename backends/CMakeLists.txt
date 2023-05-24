cmake_minimum_required(VERSION 3.10)
project(jlio-backends VERSION 0.1.0)

set(TBB_VERSION 2021.9.0)

find_package(Threads)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(WITH_THREADING "When CUDA is not used, distribute work to all cores, otherwise single core." true)
option(WITH_CUDA "With CUDA support" true)
option(BUILD_SHARED_LIBS "Build shared libraries" true)

find_package(CUDAToolkit)
if(WITH_CUDA AND CUDAToolkit_FOUND)
    include(CheckLanguage)
    check_language(CUDA REQUIRED)
 
    set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE}) # override the CUDA version to the one found by the toolkit package
    message(STATUS "-::--::- CUDA support enabled -::--::-")

    add_definitions(-DUSE_CUDA) # used in the source code
    if(NOT DEFINED CMAKE_CUDA_STANDARD)
      set(CMAKE_CUDA_STANDARD 11)
      set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()

    set(CMAKE_CUDA_ARCHITECTURES 75)

    enable_language(CUDA)    
else()
    if(WITH_THREADING)
        add_definitions(-DUSE_THREADING)
        message(STATUS "-::--::- CPU only [all cores] -::--::-")
    else()
        message(STATUS "-::--::- CPU only [single core] -::--::-")
    endif()

    find_package(TBB ${TBB_VERSION} QUIET)
    if(TBB_FOUND)
        message(STATUS "TBB ${TBB_VERSION} found.")
    else()
        message(STATUS "TBB ${TBB_VERSION} not found. Will be compiled when compiling.")

        include(ExternalProject)
        ExternalProject_Add(
            oneTBB
            PREFIX ${CMAKE_BINARY_DIR}/externalproject
            GIT_REPOSITORY https://github.com/oneapi-src/oneTBB
            GIT_TAG v${TBB_VERSION}
            CMAKE_ARGS -DCMAKE_BUILD_TYPE=Release -DTBB_TEST=Off -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/externalproject-install
        )
        set(TBB_ROOT ${CMAKE_BINARY_DIR}/externalproject-install)
        set(TBB_INCLUDE_DIRS ${TBB_ROOT}/include)
        set(TBB_LIBRARIES ${TBB_ROOT}/lib/libtbb.so)
        set(TBB_LIBRARY_DIRS ${TBB_ROOT}/lib)
        set(TBB_FOUND TRUE)
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
