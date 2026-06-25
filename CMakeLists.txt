cmake_minimum_required(VERSION 3.15...3.27)
project(frgfp LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native -Wall -Wextra)
endif()

add_subdirectory(frgfp/_core)