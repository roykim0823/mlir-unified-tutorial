cmake_minimum_required(VERSION 3.24)
project(mlir-unified-tutorial LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# MLIR comes from the mlir-wheel pip package (see requirements.txt).
# Configure with:
#   cmake .. -DCMAKE_PREFIX_PATH=$(python -m mlir_wheel --root-dir) \
#            -DLLVM_EXTERNAL_LIT=$(which lit)
find_package(MLIR CONFIG)
if(NOT MLIR_FOUND)
  message(FATAL_ERROR
    "MLIR not found. Install the wheel and point CMake at it:\n"
    "  pip install -r requirements.txt\n"
    "  cmake .. -DCMAKE_PREFIX_PATH=$(python -m mlir_wheel --root-dir) -DLLVM_EXTERNAL_LIT=$(which lit)")
endif()

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

# The or-tools fetch must happen BEFORE HandleLLVMOptions: its dependency tree
# (coinutils etc.) uses __DATE__ and would trip LLVM's -Werror=date-time.
option(ENABLE_SOLVER "Build the ch23 noisy-dialect optimizer (fetches or-tools; big build)" OFF)
if(ENABLE_SOLVER)
  message(STATUS "Fetching or-tools (this is a large download/build)...")
  set(BUILD_DEPS ON)
  include(FetchContent)
  FetchContent_Declare(
    or-tools
    GIT_REPOSITORY https://github.com/google/or-tools.git
    GIT_TAG        v9.11
  )
  FetchContent_MakeAvailable(or-tools)
  message(STATUS "Done fetching or-tools")
  # or-tools defaults BUILD_SHARED_LIBS to ON in the cache; our MLIR libraries
  # must stay static or they fail to link against the wheel's static archives.
  set(BUILD_SHARED_LIBS OFF)
endif()

include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_BINARY_DIR})

add_subdirectory(lib)
if(ENABLE_SOLVER)
  add_subdirectory(lib/Analysis)
endif()
add_subdirectory(tools)
add_subdirectory(test)

# Preserved "before" code snapshots used by individual chapters:
# ch10 keeps the hand-written (pre-tablegen) passes as tutorial-opt-ch10;
# ch16 keeps the C++ (pre-DRR) canonicalization pattern as tutorial-opt-ch16.
add_subdirectory(part2-writing-passes/ch10-first-pass-cpp/code)
add_subdirectory(part3-building-a-dialect/ch16-canonicalizers-drr/code)
