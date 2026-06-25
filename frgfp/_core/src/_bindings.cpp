#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <memory>
#include "_ansatz.hpp"
#include "_func.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_core_cpp, m) {
    m.doc() = "High-performance C++ backend for the frgfp core module";

    using namespace frgfp::core;

    py::class_<Ansatz, std::shared_ptr<Ansatz>>(m, "Ansatz")
        .def_property_readonly("inv_dim", &Ansatz::get_inv_dim)
        .def_property_readonly("num_coeffs", &Ansatz::get_num_coeffs)
        .def("evaluate_basis", &Ansatz::evaluate_basis, py::arg("grid"))
        .def("evaluate_basis_grad", &Ansatz::evaluate_basis_grad, py::arg("grid"))
        .def("evaluate_basis_hess", &Ansatz::evaluate_basis_hess, py::arg("grid"));

    py::class_<ChebyshevAnsatz, Ansatz, std::shared_ptr<ChebyshevAnsatz>>(m, "ChebyshevAnsatz")
        .def(py::init<int, const Eigen::VectorXi&, const Eigen::MatrixXd&>(),
             py::arg("inv_dim"), py::arg("order"), py::arg("interval"));

    py::class_<PolyAnsatz, Ansatz, std::shared_ptr<PolyAnsatz>>(m, "PolyAnsatz")
        .def(py::init<int, const Eigen::VectorXi&, const Eigen::VectorXd&>(),
             py::arg("inv_dim"), py::arg("order"), py::arg("center"));

    py::class_<FuncFromAnsatz, std::shared_ptr<FuncFromAnsatz>>(m, "FuncFromAnsatz")
        .def(py::init<std::shared_ptr<Ansatz>, const Eigen::VectorXd&>(),
             py::arg("ansatz"), py::arg("coeffs"))
        .def_property("coeffs", &FuncFromAnsatz::get_coeffs, &FuncFromAnsatz::set_coeffs)
        .def_property_readonly("ansatz", &FuncFromAnsatz::get_ansatz)
        .def_property_readonly("num_coeffs", &FuncFromAnsatz::get_num_coeffs)
        .def("evaluate", &FuncFromAnsatz::evaluate, py::arg("grid"),
             "Evaluates the functional expansion on the provided collocation grid.")
        .def("evaluate_grad", &FuncFromAnsatz::evaluate_grad, py::arg("grid"),
             "Evaluates the functional gradient on the provided collocation grid.")
        .def("evaluate_hess", &FuncFromAnsatz::evaluate_hess, py::arg("grid"),
             "Evaluates the functional Hessian on the provided collocation grid.");

    m.def("fit_coeffs_from_grid", &fit_coeffs_from_grid, 
          py::arg("ansatz"), py::arg("grid"), py::arg("vals"),
          "Calculates optimal coefficients for a grid via SVD pseudo-inverse.");
}