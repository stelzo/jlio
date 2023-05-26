cmake_minimum_required(VERSION 3.10)

include_directories(${GTEST_INCLUDE_DIRS})

add_subdirectory(nearest_search)
add_subdirectory(rm_mtk_interop_helper)
add_subdirectory(rm_mtk_interop)
add_subdirectory(pca)
