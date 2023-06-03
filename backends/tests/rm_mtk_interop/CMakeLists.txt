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
  ${CMAKE_CURRENT_SOURCE_DIR}/../rm_mtk_interop_helper
)

if(CMAKE_CUDA_COMPILER)
  add_definitions(-DUSE_CUDA)
  list(APPEND TEST_INCLUDES
    ${CUDA_INCLUDE_DIRS}
  )
endif()
if(WITH_THREADING)
  list(APPEND TEST_LINKS
    TBB::tbb
  )
endif()
list(APPEND TEST_LINKS
  Eigen3::Eigen
)
list(APPEND TEST_INCLUDES
  ${Eigen3_INCLUDE_DIRS}
)

add_executable(${TEST_TARGET} ${TEST_SRC})
target_link_libraries(${TEST_TARGET} ${TEST_LINKS} interop_test_helper)
target_include_directories(${TEST_TARGET} PRIVATE ${TEST_INCLUDES})
gtest_add_tests(TARGET ${TEST_TARGET})
add_dependencies(${TEST_TARGET} kernel interop_test_helper)
