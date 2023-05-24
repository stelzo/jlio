cmake_minimum_required(VERSION 3.10)
project(tree VERSION 0.1.0)

set(TREE_SRC 
  src/tree.cpp
)

set(TREE_INCLUDES
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  include
  ../kernel/include
)

set(TREE_LIBS
  kernel
  ${CMAKE_THREAD_LIBS_INIT}
  ${PCL_LIBRARIES}
)

if(BUILD_SHARED_LIBS)
  add_library(${PROJECT_NAME} SHARED ${TREE_SRC})
else()
  add_library(${PROJECT_NAME} STATIC ${TREE_SRC})
endif()

target_include_directories(${PROJECT_NAME} PRIVATE ${TREE_INCLUDES})

target_link_libraries(${PROJECT_NAME} PRIVATE ${TREE_LIBS})
