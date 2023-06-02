cmake_minimum_required(VERSION 3.17)
project(jlio VERSION 0.1.0)

option(WITH_TESTS "Build tests" ON)
option(LTO_OPT "Enable LTO" OFF)
option(WITH_BACKENDS "Build backends" ON)
option(WITH_THREADING "When CUDA is not used, distribute work to all cores, otherwise single core." ON)

if (LTO_OPT)
  MESSAGE(STATUS "Using Link Time Optimizations, this will take time...")
  add_compile_options(-flto)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  execute_process(
    COMMAND bash -c "brew --prefix qt@5"
    OUTPUT_VARIABLE QT5_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(Qt5_DIR ${QT5_CMAKE_PREFIX_PATH}/lib/cmake/Qt5)
endif()

include(GNUInstallDirs)

set(NEEDED_TBB_VERSION 2021.9.0) # 2020.1

if(WITH_THREADING)
  add_definitions(-DUSE_THREADING=1)
  find_package(TBB ${NEEDED_TBB_VERSION})
  if(TBB_FOUND)
      message(STATUS "TBB ${NEEDED_TBB_VERSION} found.")
  else()
    message(STATUS "TBB ${NEEDED_TBB_VERSION} not found. Will be compiled.")
    add_subdirectory(ext/tbb)
  endif()
endif()

if(WITH_TESTS)
  add_subdirectory(ext/gtest)

  enable_testing()
  include(GoogleTest)
endif()

if(WITH_BACKENDS)
  add_subdirectory(backends)
endif()

