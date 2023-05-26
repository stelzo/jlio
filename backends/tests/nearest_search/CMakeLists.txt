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

if(WITH_THREADING)
  list(APPEND TEST_LINKS
    TBB::tbb
  )
endif()

set(TEST_INCLUDES
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../kernel/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../tree/include
)

if(CMAKE_CUDA_COMPILER)
  add_definitions(-DUSE_CUDA)
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
  )
  list(APPEND TEST_INCLUDES
    ${Eigen3_INCLUDE_DIRS}
  )
endif()


add_executable(${TEST_TARGET} ${TEST_SRC})
target_link_libraries(${TEST_TARGET} ${TEST_LINKS})
target_include_directories(${TEST_TARGET} PRIVATE ${TEST_INCLUDES})
gtest_discover_tests(${TEST_TARGET} PROPERTIES TIMEOUT 600)
add_dependencies(${TEST_TARGET} kernel tree)