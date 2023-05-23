cmake_minimum_required(VERSION 3.10)
project(jlio VERSION 0.1.0)

option(BUILD_TESTS "Build tests" false)
option(LTO_OPT "Enable LTO" false)
option(BUILD_LIBS "Build libraries" true)

if (LTO_OPT)
  MESSAGE(STATUS "using Link Time Optimizations, this will take time...")
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

add_subdirectory(backends)
