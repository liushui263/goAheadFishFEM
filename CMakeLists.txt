cmake_minimum_required(VERSION 3.18)
project(FEM_Maxwell_Solver LANGUAGES CXX CUDA)

# --- 添加 Iterative Solver 选项 ---
option(USE_ITERATIVE_SOLVER "Use iterative solver (BiCGSTAB) instead of direct solver" OFF)

if(USE_ITERATIVE_SOLVER)
    message(STATUS "FEM Solver: Using Iterative Solver (BiCGSTAB)")
else()
    message(STATUS "FEM Solver: Using Direct Solver (SparseLU)")
endif()

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
        if(USE_ITERATIVE_SOLVER)
            set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/CEM/FEM3D/ITER_build" CACHE PATH "Install path" FORCE)
        else()
            set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/CEM/FEM3D/CPU_build" CACHE PATH "Install path" FORCE)
        endif()
    endif()
endif()

# ----------------------------------
# Compiler settings
# ----------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------------
# Library source files
# ----------------------------------
file(GLOB SRC_FILES CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/src/*.cpp)
list(FILTER SRC_FILES EXCLUDE REGEX "/main\\.cpp$")

add_library(fem_lib ${SRC_FILES})
target_include_directories(fem_lib PUBLIC ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)

if(USE_ITERATIVE_SOLVER)
    target_compile_definitions(fem_lib PUBLIC USE_ITERATIVE_SOLVER)
endif()

# ----------------------------------
# Executables
# ----------------------------------
add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE fem_lib)

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

# --- nlohmann_json (Reliable C++ JSON API) ---
# 1. 优先查找本地安装的版本 (如 Conda 环境中)，避免下载
find_package(nlohmann_json 3.11 QUIET)

if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found locally, downloading via FetchContent...")
    set(JSON_BuildTests OFF CACHE INTERNAL "") # 关键：关闭 JSON 库自身的测试以减少编译时间
    FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.2
        GIT_SHALLOW TRUE  # 关键：只下载最新 commit，不下载整个历史
    )
    FetchContent_MakeAvailable(json)
endif()

enable_testing()
include(GoogleTest)

add_executable(unit_tests test/test_cases.cpp)
target_link_libraries(unit_tests PRIVATE fem_lib GTest::gtest_main)
gtest_discover_tests(unit_tests DISCOVERY_TIMEOUT 60)

# ----------------------------------
# Libraries
# ----------------------------------
find_package(Eigen3 REQUIRED)
target_link_libraries(main PRIVATE Eigen3::Eigen)
target_link_libraries(fem_lib PUBLIC nlohmann_json::nlohmann_json)

# ----------------------------------
# GPU / CPU build configuration
# ----------------------------------
if(USE_GPU)
    message(STATUS "Building with GPU/cuDSS support")
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
    # We must set properties on targets because they were created before this block
    get_filename_component(CUDSS_LIB_DIR ${CUDSS_LIBRARY} DIRECTORY)
    set_target_properties(main unit_tests PROPERTIES
        INSTALL_RPATH "${CUDSS_LIB_DIR}"
        INSTALL_RPATH_USE_LINK_PATH TRUE
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
install(TARGETS main unit_tests
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
