"""
Functional Renormalization Group Fixed Point (frgfp) Solver.

A high-performance computational package designed to find global collocation
solutions for Functional Renormalization Group (FRG) flow equations. The package 
provides a robust, generalized framework for evaluating effective potentials 
and their derivatives on freely specifiable grids.

The core of the package utilizes a highly optimized Eigen C++ backend, heavily
leveraging pre-calculated 1D caches and row-major tensor products to perform
lightning-fast evaluations of multi-dimensional basis functions, gradients,
and Hessian matrices.

Available Modules
-----------------
core
    Fundamental data structures, ansatz definitions, collocation grid
    generators, and function evaluators powered by the C++ backend. 
    (Internal components are wrapped and exposed directly in this namespace 
    for convenience).

Notes
-----
This package represents physical variables (invariants) as dimensions in
multivariate expansions. All arrays and matrices are rigorously validated
and pushed to the C++ backend as memory-safe Eigen objects to ensure numerical
stability during multidimensional root-finding and Newton-Gauss iterations.
"""

from ._core import (
    PolyAnsatz, 
    ChebyshevAnsatz, 
    uniform_collgrid, 
    chebyshev_collgrid,
    FuncFromAnsatz
)

__all__ = [
    "PolyAnsatz",
    "ChebyshevAnsatz",
    "uniform_collgrid",
    "chebyshev_collgrid",
    "FuncFromAnsatz",
]

from . import lpa

__all__ = ["lpa"]