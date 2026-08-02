pybind11_add_module(_lpa_cpp
    src/_bindings.cpp
    src/_solver.cpp
)

target_include_directories(_lpa_cpp PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(_lpa_cpp PRIVATE 
    frgfp_core_lib 
    OpenMP::OpenMP_CXX
)

install(TARGETS _lpa_cpp DESTINATION frgfp/lpa)