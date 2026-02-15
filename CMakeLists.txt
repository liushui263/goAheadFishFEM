cmake_minimum_required(VERSION 3.18)
project(FEM_Maxwell_Solver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_GPU "Use GPU/cuDSS solver" ON)

# Include directories for all targets

include_directories(${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)

# Main source files

file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)
list(FILTER SRC_FILES EXCLUDE REGEX "/main\\.cpp$")

# Create a library for the solver to share code between main and tests
add_library(fem_lib ${SRC_FILES})
target_include_directories(fem_lib PUBLIC ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)

# Main executable

add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE fem_lib)

# Test executables

add_executable(test_simple test/test_simple.cpp)
target_link_libraries(test_simple PRIVATE fem_lib)

add_executable(test_anisotropic test/test_anisotropic.cpp)
target_link_libraries(test_anisotropic PRIVATE fem_lib)

add_executable(benchmark test/benchmark.cpp)
target_link_libraries(benchmark PRIVATE fem_lib)

# GoogleTest Unit Tests
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

add_executable(unit_tests test/test_cases.cpp)
target_link_libraries(unit_tests PRIVATE fem_lib GTest::gtest_main)
gtest_discover_tests(unit_tests)

# Link libraries

find_package(Eigen3 REQUIRED)
target_link_libraries(main PRIVATE Eigen3::Eigen)

#include(FetchContent)

#set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
#set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
#set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

#FetchContent_Declare(
#  Eigen3
#  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
#  GIT_TAG        3.4.0
#)
#FetchContent_MakeAvailable(Eigen3)

if(USE_GPU)
message(STATUS "Building with GPU/cuDSS support")
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# Find cuDSS manually as it is often a separate installation
find_path(CUDSS_INCLUDE_DIR cudss.h
    HINTS /usr/local/cudss/include /usr/local/include ${CUDAToolkit_INCLUDE_DIRS}
)
find_library(CUDSS_LIBRARY cudss
    HINTS /usr/local/cudss/lib64 /usr/local/cudss/lib /usr/local/lib
)

if(CUDSS_INCLUDE_DIR AND CUDSS_LIBRARY)
    target_include_directories(fem_lib PUBLIC ${CUDSS_INCLUDE_DIR})
    target_link_libraries(fem_lib PUBLIC ${CUDSS_LIBRARY})
else()
    message(FATAL_ERROR "cuDSS not found. Please set CUDSS_INCLUDE_DIR and CUDSS_LIBRARY.")
endif()

set(GPU_LIBS CUDA::cudart CUDA::cusparse CUDA::cublas)
target_compile_definitions(fem_lib PUBLIC USE_GPU)

target_link_libraries(fem_lib PUBLIC Eigen3::Eigen ${GPU_LIBS})

else()
message(STATUS "Building with CPU sparse solver")
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

target_link_libraries(fem_lib PUBLIC Eigen3::Eigen ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

endif()

# Optional: Set CUDA include and library dirs if not found automatically

if(USE_GPU AND NOT CUDAToolkit_FOUND)
set(CUDA_DIR "/usr/local/cuda")  # adjust your CUDA path here
include_directories(${CUDA_DIR}/include)
link_directories(${CUDA_DIR}/lib64)
endif()
