add_library(frgfp_core_lib STATIC
    src/_ansatz.cpp
    src/_func.cpp
)

set_target_properties(frgfp_core_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(frgfp_core_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(frgfp_core_lib PUBLIC Eigen3::Eigen)
target_compile_features(frgfp_core_lib PUBLIC cxx_std_17)

pybind11_add_module(_core_cpp src/_bindings.cpp)

target_link_libraries(_core_cpp PRIVATE frgfp_core_lib)

install(TARGETS _core_cpp DESTINATION frgfp/_core)