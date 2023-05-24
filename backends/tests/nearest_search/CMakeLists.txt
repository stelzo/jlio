cmake_minimum_required(VERSION 3.14)
set(TEST_TARGET nearest_search_test)
project(${TEST_TARGET} CXX)

set(TEST_SRC
  ${TEST_TARGET}.cpp
)

set(TEST_LINKS
  GTest::gtest_main
  kernel
  tree
)

if(TBB_FOUND)
  list(APPEND TEST_LINKS
    TBB::tbb
  )
  list(APPEND TEST_INCLUDES
    ${TBB_INCLUDE_DIRS}
  )
endif()

set(TEST_INCLUDES
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../kernel/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../tree/include
)

if(CMAKE_CUDA_COMPILER)
  set_source_files_properties(${PROJECT_NAME}.cpp PROPERTIES LANGUAGE CUDA)

  list(APPEND TEST_LINKS 
    ${CUDA_LIBRARIES}
  )
  list(APPEND TEST_INCLUDES
    ${CUDA_INCLUDE_DIRS}
  )
else()
  list(APPEND TEST_LINKS
    Eigen3::Eigen
    ${PCL_LIBRARIES}
  )
  list(APPEND TEST_INCLUDES
    ${Eigen3_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
  )
endif()


add_executable(${TEST_TARGET} ${TEST_SRC})
target_link_libraries(${TEST_TARGET} ${TEST_LINKS})
target_include_directories(${TEST_TARGET} PRIVATE ${TEST_INCLUDES})
gtest_discover_tests(${TEST_TARGET})