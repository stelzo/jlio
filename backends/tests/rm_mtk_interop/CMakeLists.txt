cmake_minimum_required(VERSION 3.14)
set(TEST_TARGET rm_mtk_interop_test)
project(${TEST_TARGET} CXX)

set(TEST_SRC
  ${TEST_TARGET}.cpp
)

set(TEST_LINKS
  GTest::gtest_main
  kernel
)

set(TEST_INCLUDES
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../kernel/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../kalman/include
)

if(TBB_FOUND)
  list(APPEND TEST_LINKS
    TBB::tbb
  )
  list(APPEND TEST_INCLUDES
    ${TBB_INCLUDE_DIRS}
  )
endif()

list(APPEND TEST_LINKS
  Eigen3::Eigen
)
list(APPEND TEST_INCLUDES
  ${Eigen3_INCLUDE_DIRS}
)

add_executable(${TEST_TARGET} ${TEST_SRC})
target_link_libraries(${TEST_TARGET} ${TEST_LINKS})
target_include_directories(${TEST_TARGET} PRIVATE ${TEST_INCLUDES})
gtest_discover_tests(${TEST_TARGET})