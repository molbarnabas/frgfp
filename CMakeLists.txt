cmake_minimum_required(VERSION 3.15...3.27)
project(frgfp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)

# --- Eigen3 -------------------------------------------------------------------
# Prefer a system / conda Eigen (passed through Eigen3_DIR or CMAKE_PREFIX_PATH).
# If none is available - the normal situation on the bare GitHub runners and in
# the manylinux containers used by cibuildwheel - fall back to a pinned,
# header-only download so that `pip install .` works out of the box on Linux,
# macOS and Windows.
find_package(Eigen3 3.3 CONFIG QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found, fetching Eigen 3.4.0 (set Eigen3_DIR to use a local copy)")
    include(FetchContent)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        eigen
        URL      https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
    )
    FetchContent_MakeAvailable(eigen)
    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${eigen_SOURCE_DIR}")
    endif()
endif()

# --- OpenMP -------------------------------------------------------------------
# The LPA backend calls the OpenMP runtime directly (omp_in_parallel,
# omp_get_thread_num, omp_get_max_threads in frgfp/lpa/src/_solver.cpp), so
# OpenMP is a hard requirement.  AppleClang ships no OpenMP runtime, so on macOS
# locate the Homebrew `libomp` keg instead of requiring the user to export flags
# by hand (install it with `brew install libomp`).
find_package(OpenMP QUIET)
if(NOT OpenMP_CXX_FOUND AND APPLE)
    find_program(BREW_EXECUTABLE brew)
    if(BREW_EXECUTABLE)
        execute_process(
            COMMAND "${BREW_EXECUTABLE}" --prefix libomp
            OUTPUT_VARIABLE FRGFP_LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE FRGFP_LIBOMP_RESULT
            ERROR_QUIET
        )
        if(FRGFP_LIBOMP_RESULT EQUAL 0)
            message(STATUS "Using Homebrew libomp from ${FRGFP_LIBOMP_PREFIX}")
            set(OpenMP_ROOT "${FRGFP_LIBOMP_PREFIX}" CACHE PATH "Homebrew libomp prefix")
            list(APPEND CMAKE_PREFIX_PATH "${FRGFP_LIBOMP_PREFIX}")
        endif()
    endif()
endif()
find_package(OpenMP REQUIRED)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -Wall -Wextra)
endif()

include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(frgfp/_core)
add_subdirectory(frgfp/lpa)