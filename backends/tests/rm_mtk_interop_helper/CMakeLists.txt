cmake_minimum_required(VERSION 3.14)
project(${TEST_TARGET} CXX)

set(KERNEL_HELPER_SRC
  rm_mtk_interop_test_no_eigen.cpp
)

set(KERNEL_HELPER_LINKS
)

set(KERNEL_HELPER_INCLUDES
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../kernel/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../rm_mtk_interop
)

if(CMAKE_CUDA_COMPILER)
  add_definitions(-DUSE_CUDA)

  set_source_files_properties(rm_mtk_interop_test_no_eigen.cpp PROPERTIES LANGUAGE CUDA)

  list(APPEND KERNEL_HELPER_LINKS 
    ${CUDA_LIBRARIES}
  )
  list(APPEND KERNEL_HELPER_INCLUDES
    ${CUDA_INCLUDE_DIRS}
  )
else()
  if(WITH_THREADING)
    list(APPEND KERNEL_HELPER_LINKS
      TBB::tbb
    )
  endif()
endif()

add_library(interop_test_helper STATIC ${KERNEL_HELPER_SRC})
target_include_directories(interop_test_helper PRIVATE ${KERNEL_HELPER_INCLUDES})
target_link_libraries(interop_test_helper PRIVATE ${KERNEL_HELPER_LINKS})
