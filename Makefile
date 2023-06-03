.DEFAULT_GOAL := all

# Specify the target executable name
TARGET = jlio

# Specify the CMake build directory
BUILD_DIR = build

# Specify the CMake command
CMAKE_COMMAND = cmake

# Specify any additional flags or options for CMake
CMAKE_FLAGS = -Wno-dev

ifeq ($(shell uname), Darwin)
    C_COMPILER := -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_COMPILER=g++-11
else
    C_COMPILER := -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
endif

MAKE_FLAGS = --no-print-directory --jobs 2

DEBUG_FLAGS = -DCMAKE_BUILD_TYPE=Debug -Wno-pedantic

RELEASE_FLAGS = -DCMAKE_BUILD_TYPE=Release -Wno-pedantic

prepare-build:
	@mkdir -p $(BUILD_DIR)

all-dev: prepare-build
	@cd $(BUILD_DIR) && $(CMAKE_COMMAND) $(C_COMPILER) $(CMAKE_FLAGS) -DCMAKE_BUILD_TYPE=Debug ..
	@cd $(BUILD_DIR) && $(MAKE) $(MAKE_FLAGS)

# Default target
all: all-release

all-release: prepare-build
	@cd $(BUILD_DIR) && $(CMAKE_COMMAND) $(C_COMPILER) $(CMAKE_FLAGS) $(RELEASE_FLAGS) ..
	@cd $(BUILD_DIR) && $(MAKE) $(MAKE_FLAGS)

all-cuda-dev: prepare-build
	@cd $(BUILD_DIR) && $(CMAKE_COMMAND) $(C_COMPILER) $(CMAKE_FLAGS) $(DEBUG_FLAGS) -DWITH_CUDA=ON ..
	@cd $(BUILD_DIR) && $(MAKE) $(MAKE_FLAGS)

all-cuda-release: prepare-build
	@cd $(BUILD_DIR) && $(CMAKE_COMMAND) $(C_COMPILER) $(CMAKE_FLAGS) $(RELEASE_FLAGS) -DWITH_CUDA=ON ..
	@cd $(BUILD_DIR) && $(MAKE) $(MAKE_FLAGS)

all-cpu-full-dev: prepare-build
	@cd $(BUILD_DIR) && $(CMAKE_COMMAND) $(C_COMPILER) $(CMAKE_FLAGS) $(DEBUG_FLAGS) -DWITH_CUDA=OFF -DWITH_THREADING=ON ..
	@cd $(BUILD_DIR) && $(MAKE) $(MAKE_FLAGS)

all-cpu-single-dev: prepare-build
	@cd $(BUILD_DIR) && $(CMAKE_COMMAND) $(C_COMPILER) $(CMAKE_FLAGS) $(DEBUG_FLAGS) -DWITH_CUDA=OFF -DWITH_THREADING=OFF ..
	@cd $(BUILD_DIR) && $(MAKE) $(MAKE_FLAGS)

all-cpu-full-release: prepare-build
	@cd $(BUILD_DIR) && $(CMAKE_COMMAND) $(C_COMPILER) $(CMAKE_FLAGS) $(RELEASE_FLAGS) -DWITH_CUDA=OFF -DWITH_THREADING=ON ..
	@cd $(BUILD_DIR) && $(MAKE) $(MAKE_FLAGS)

all-cpu-single-release: prepare-build
	@cd $(BUILD_DIR) && $(CMAKE_COMMAND) $(C_COMPILER) $(CMAKE_FLAGS) $(RELEASE_FLAGS) -DWITH_CUDA=OFF -DWITH_THREADING=OFF ..
	@cd $(BUILD_DIR) && $(MAKE) $(MAKE_FLAGS)

all-cuda: all-cuda-release

all-cpu: all-cpu-full-release

# Rule to clean the build directory
clean:
	@$(MAKE) -C $(BUILD_DIR) clean $(MAKE_FLAGS)
	@rm -rf $(BUILD_DIR)

.PHONY: all all-dev all-release clean all-cpu all-cuda all-cpu-single-release all-cpu-full-release all-cpu-single-dev all-cpu-full-dev all-cuda-dev test
