cmake_minimum_required(VERSION 3.18)
project(FEM_Maxwell_Solver LANGUAGES CXX CUDA)

# ----------------------------------
# Default build type
# ----------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (default Release)" FORCE)
endif()

# ----------------------------------
# Option for GPU/cuDSS
# ----------------------------------
option(USE_GPU "Use GPU/cuDSS solver" ON)

# Set a user-writable default if the user did not provide one
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if(USE_GPU)
        set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/CEM/FEM3D/GPU_build" CACHE PATH "Install path" FORCE)
    else()
        set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/CEM/FEM3D/CPU_build" CACHE PATH "Install path" FORCE)
    endif()
endif()

# ----------------------------------
# Compiler settings
# ----------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)

# ----------------------------------
# Library source files
# ----------------------------------
file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)
list(FILTER SRC_FILES EXCLUDE REGEX "/main\\.cpp$")

add_library(fem_lib ${SRC_FILES})
target_include_directories(fem_lib PUBLIC ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)

# ----------------------------------
# Executables
# ----------------------------------
add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE fem_lib)

add_executable(test_simple test/test_simple.cpp)
target_link_libraries(test_simple PRIVATE fem_lib)

add_executable(test_anisotropic test/test_anisotropic.cpp)
target_link_libraries(test_anisotropic PRIVATE fem_lib)

add_executable(benchmark test/benchmark.cpp)
target_link_libraries(benchmark PRIVATE fem_lib)

# ----------------------------------
# GoogleTest (local, avoids permission errors)
# ----------------------------------
include(FetchContent)
set(INSTALL_GTEST OFF CACHE BOOL "Do not install GTest system-wide" FORCE) # prevents /usr/local install
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE) # optional, avoids conflicts
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

# ----------------------------------
# Libraries
# ----------------------------------
find_package(Eigen3 REQUIRED)
target_link_libraries(main PRIVATE Eigen3::Eigen)

# ----------------------------------
# GPU / CPU build configuration
# ----------------------------------
if(USE_GPU)
    message(STATUS "Building with GPU/cuDSS support")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

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

    # ---------------------------
    # RPATH settings for GPU build
    # ---------------------------
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

    get_target_property(CUDA_LIB_DIR CUDA::cudart LOCATION)
    get_filename_component(CUDA_LIB_DIR ${CUDA_LIB_DIR} DIRECTORY)
    get_filename_component(CUDSS_LIB_DIR ${CUDSS_LIBRARY} DIRECTORY)

    set(RPATH_LIST "${CUDA_LIB_DIR};${CUDSS_LIB_DIR}")
    set_target_properties(fem_lib main test_simple test_anisotropic benchmark unit_tests
        PROPERTIES
            INSTALL_RPATH "${RPATH_LIST}"
            BUILD_WITH_INSTALL_RPATH TRUE
    )

else()
    message(STATUS "Building CPU-only version: skipping CUDA and NVPL entirely")
    set(CMAKE_DISABLE_FIND_PACKAGE_nvpl TRUE)

    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)

    if(NOT BLAS_FOUND OR NOT LAPACK_FOUND)
        message(FATAL_ERROR "BLAS/LAPACK not found. Install libblas-dev and liblapack-dev")
    endif()

    target_link_libraries(fem_lib PUBLIC Eigen3::Eigen ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

# Optional: set CUDA dirs manually if needed
if(USE_GPU AND NOT CUDAToolkit_FOUND)
    set(CUDA_DIR "/usr/local/cuda")
    include_directories(${CUDA_DIR}/include)
    link_directories(${CUDA_DIR}/lib64)
endif()

# ----------------------------------
# Installation
# ----------------------------------
# Library
install(TARGETS fem_lib
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
)

# Executables
install(TARGETS main test_simple test_anisotropic benchmark unit_tests
        RUNTIME DESTINATION bin
)

# Headers
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY src/ DESTINATION src FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

# GPU-specific headers/libs installed only if USE_GPU=ON
if(USE_GPU)
    install(DIRECTORY ${CUDSS_INCLUDE_DIR}/ DESTINATION include/cudss FILES_MATCHING PATTERN "*.h")
    install(FILES ${CUDSS_LIBRARY} DESTINATION lib)
endif()
