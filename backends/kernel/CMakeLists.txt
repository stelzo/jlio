cmake_minimum_required(VERSION 3.11)
project(kernel VERSION 0.1.0)

include_directories(include)

set(KERNEL_LINK_LIBS
  kalman
)


if(CMAKE_CUDA_COMPILER)
  include_directories(${CUDA_INCLUDE_DIRS})
  set_source_files_properties(src/point_jacobian.cpp src/point_fitting.cpp PROPERTIES LANGUAGE CUDA)
else()
  if(TBB_FOUND)
    list(APPEND KERNEL_LINK_LIBS TBB::tbb)
  endif()

  include_directories(
    ${TBB_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
  )
endif()

add_library(${PROJECT_NAME} SHARED
  src/point_jacobian.cpp
  src/point_fitting.cpp
  src/preprocess.cpp
  src/memory.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  ${KERNEL_LINK_LIBS}
)

set(BINDINGS_LINK_LIBS
  kernel
  kalman
)

set(BINDINGS_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../kalman/include
)

set(BINDINGS_SRC
  src/bindings.cpp
)

if(TBB_FOUND)
  list(APPEND BINDINGS_LINK_LIBS TBB::tbb)
endif()

if(BUILD_SHARED_LIBS)
  add_library(bindings SHARED ${BINDINGS_SRC})
else()
  add_library(bindings STATIC ${BINDINGS_SRC})
endif()

target_include_directories(bindings PRIVATE ${BINDINGS_INCLUDE_DIRS})
target_link_libraries(bindings PRIVATE ${BINDINGS_LINK_LIBS})
