cmake_minimum_required(VERSION 3.15...3.27)
project(frgfp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenMP REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -Wall -Wextra)
endif()

include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(frgfp/_core)
add_subdirectory(frgfp/lpa)