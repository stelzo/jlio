cmake_minimum_required(VERSION 3.11)
project(kernel VERSION 0.1.0)

include_directories(include)

set(KERNEL_LINK_LIBS
  kalman
)

set(KERNEL_SRC
  src/preprocess.cpp
  src/memory.cpp
  src/point_jacobian.cpp
  src/point_fitting.cpp
  src/common.cpp
)

set(KERNEL_INCLUDES 
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)


if(CMAKE_CUDA_COMPILER)
  include_directories(${CUDA_INCLUDE_DIRS})
  set_source_files_properties(src/common.cpp src/point_jacobian.cpp src/point_fitting.cpp src/preprocess.cpp PROPERTIES LANGUAGE CUDA)
  list(APPEND KERNEL_LINK_LIBS
    CUDA::cudart
  )
else()
  if(WITH_THREADING)
    list(APPEND KERNEL_LINK_LIBS
      TBB::tbb
    )
  endif()

  list(APPEND KERNEL_INCLUDES
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
  )
endif()

if(BUILD_SHARED_LIBS)
  add_library(${PROJECT_NAME} SHARED ${KERNEL_SRC})
else()
  add_library(${PROJECT_NAME} STATIC ${KERNEL_SRC})
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE
  ${KERNEL_LINK_LIBS}
)

target_include_directories(${PROJECT_NAME} PRIVATE ${KERNEL_INCLUDES})
add_dependencies(${PROJECT_NAME} kalman)

# bindings library
set(BINDINGS_LINK_LIBS
  kernel
  kalman
)

set(BINDINGS_SRC
  src/bindings.cpp
)

set(BINDINGS_INCLUDE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../kalman/include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

if(WITH_THREADING)
  list(APPEND BINDINGS_LINK_LIBS
    TBB::tbb
  )
endif()

if(BUILD_SHARED_LIBS)
  add_library(bindings SHARED ${BINDINGS_SRC})
else()
  add_library(bindings STATIC ${BINDINGS_SRC})
endif()

target_include_directories(bindings PRIVATE ${BINDINGS_INCLUDE})
target_link_libraries(bindings PRIVATE ${BINDINGS_LINK_LIBS})
add_dependencies(bindings ${PROJECT_NAME} kalman)

