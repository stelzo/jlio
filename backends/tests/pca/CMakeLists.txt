cmake_minimum_required(VERSION 3.14)
set(TEST_TARGET pca_test)
project(${TEST_TARGET} CXX)

set(TEST_SRC
  ${TEST_TARGET}.cpp
)

set(TEST_LINKS
  GTest::gtest_main
  ${PCL_LIBRARIES}
)

set(TEST_INCLUDES
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../kernel/include
  ${PCL_INCLUDE_DIRS}
)

if(WITH_THREADING)
  list(APPEND TEST_LINKS
    TBB::tbb
  )
endif()

list(APPEND TEST_LINKS
  Eigen3::Eigen
  ${PCL_LIBRARIES}
)
list(APPEND TEST_INCLUDES
  ${Eigen3_INCLUDE_DIRS}
)

add_executable(${TEST_TARGET} ${TEST_SRC})
target_link_libraries(${TEST_TARGET} ${TEST_LINKS})
target_include_directories(${TEST_TARGET} PRIVATE ${TEST_INCLUDES})
gtest_discover_tests(${TEST_TARGET})
