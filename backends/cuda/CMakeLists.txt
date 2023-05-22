cmake_minimum_required(VERSION 3.10)
project(cuda VERSION 0.1.0)

if(WITH_THREADING)
  add_definitions(-DUSE_THREADING)
  message(STATUS "Building with threading support.")
else()
  message(STATUS "Building without threading support.")
endif()

set(TBB_VERSION 2021.9.0)

find_package(CUDA QUIET)
if(WITH_CUDA AND CUDA_FOUND)
    add_definitions(-DUSE_CUDA)
    set(CUDA_ENABLE ON)
    enable_language(CUDA)
    message(STATUS "CUDA is available. Building with CUDA support.")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-std=c++11)  # Set additional NVCC flags if needed
else()
    message(STATUS "CUDA is not available. Using CPU Threadpools with TBB.")

    find_package(TBB ${TBB_VERSION} QUIET)
    if(TBB_FOUND)
        message(STATUS "TBB ${TBB_VERSION} found.")
    else()
        message(STATUS "TBB ${TBB_VERSION} not found. Downloading and building TBB ${TBB_VERSION}.")
    
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

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CUDA_ENABLE)
include_directories(
  include
  ${CUDA_INCLUDE_DIRS}
)
else()

find_package(PCL 1.8 REQUIRED QUIET)
include_directories(
  include
  ${TBB_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)
endif()

#add_library(cuda_preprocess src/gpu_preprocess.cu)
#target_link_libraries(gpu_preprocess ${CUDA_LIBRARIES})

#add_library(cuda_search src/search.cu)
#target_link_libraries(cuda_search ${CUDA_LIBRARIES} ${CUSOLVER_LIB})

find_package(OpenMP REQUIRED QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

find_package(Eigen3 REQUIRED QUIET)

if(CUDA_ENABLE)
set_source_files_properties(src/point_jacobian.cpp src/point_fitting.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_library(kernels
 src/point_jacobian.cpp
 src/point_fitting.cpp
 src/common.cpp
 src/memory.cpp
 )
 if(TBB_FOUND)
 target_link_libraries(kernels PRIVATE ${TBB_LIBRARIES})
endif()

add_library(bindings
 src/bindings.cpp
 src/eigen_mapping.cpp
 )
 target_include_directories(bindings PUBLIC ${EIGEN3_INCLUDE_DIR} ../kalman/include)
 if(TBB_FOUND)
 target_link_libraries(bindings PRIVATE
 kernels ${TBB_LIBRARIES} kalman
 )
 else()
  target_link_libraries(bindings PRIVATE
  kernels kalman
  )
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(bindings PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
  -O3 -march=native
  >)
else()
  target_compile_options(bindings PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
  -O0 -g -fasynchronous-unwind-tables
  >)
endif()

#if (BUILD_TESTS)
#  enable_testing()
  #add_subdirectory(ext/googletest)
#  add_subdirectory(test)
#endif()