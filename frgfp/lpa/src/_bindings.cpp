#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "_solver.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_lpa_cpp, m) {
    m.doc() = "C++ backend for the frgfp LPA module";

    py::class_<frgfp::lpa::OptimizeResult>(m, "OptimizeResult")
        .def_readonly("coeffs", &frgfp::lpa::OptimizeResult::coeffs)
        .def_readonly("success", &frgfp::lpa::OptimizeResult::success)
        .def_readonly("iterations", &frgfp::lpa::OptimizeResult::iterations)
        .def_readonly("error", &frgfp::lpa::OptimizeResult::error);

    py::class_<frgfp::lpa::LPACollSolver_cpp>(m, "LPACollSolver_cpp")
        .def(py::init<const Eigen::MatrixXd&, std::shared_ptr<frgfp::core::Ansatz>, size_t, int, int>())
        .def("local_optimize", &frgfp::lpa::LPACollSolver_cpp::local_optimize,
             py::arg("init_coeffs"), py::arg("maxiter"), py::arg("tol"), 
             py::arg("flow_params"), py::arg("multithread") = false)
        .def("param_path_following", &frgfp::lpa::LPACollSolver_cpp::param_path_following,
             py::arg("init_coeffs"), py::arg("maxiter"), py::arg("tol"), 
             py::arg("flow_params_path"), py::arg("multithread") = true)
        .def("multistart_optimize", &frgfp::lpa::LPACollSolver_cpp::multistart_optimize,
             py::arg("init_coeffs_list"), py::arg("maxiter"), py::arg("tol"), py::arg("flow_params"));
}