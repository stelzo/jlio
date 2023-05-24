cmake_minimum_required(VERSION 3.10)
project(jlio VERSION 0.1.0)

option(WITH_TESTS "Build tests" false)
option(LTO_OPT "Enable LTO" false)
option(WITH_BACKENDS "Build backends" true)

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

if(WITH_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
  )
  FetchContent_MakeAvailable(googletest)
endif()

if(WITH_BACKENDS)
  add_subdirectory(backends)
endif()
