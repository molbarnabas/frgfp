"""
FRGfp: Functional Renormalization Group Fixed Point solver.

A high-performance Python package for finding global collocation solutions of
Functional Renormalization Group (FRG) flow equations. It provides a robust,
generalized framework for evaluating effective potentials and their derivatives
on freely specifiable grids, and for locating fixed points with a Newton-Gauss
iteration.

The numerical core is a highly optimized Eigen C++ backend that relies on
pre-computed 1D caches and row-major tensor products to evaluate
multi-dimensional basis functions, gradients and Hessian matrices quickly.
User-defined flow equations are JIT-compiled with Numba and passed to the
backend through zero-overhead C-callbacks.

Features
--------
* Multidimensional basis expansions: polynomial (Taylor-like) and Chebyshev
  ansätze for an arbitrary number of invariants.
* Collocation grid generators: uniform and Chebyshev-Gauss-Lobatto grids with
  lexicographic (row-major) point ordering.
* Fast function evaluation: analytic basis, gradient and Hessian evaluation
  through the C++/Eigen backend.
* LPA fixed-point solver: Newton-Gauss collocation solver with optional OpenMP
  multithreading, parameter path following and multistart.
* Numba integration: user-defined flow equations are JIT-compiled and bridged
  to C++ as C-callbacks.

Modules
-------
core
    Fundamental data structures, ansatz definitions, collocation grid
    generators and function evaluators powered by the C++ backend. Its public
    objects are re-exported directly from this top-level namespace.
lpa
    Collocation solver for the fixed points of LPA flow equations; it bridges a
    user-defined Python flow equation with the optimized C++ Newton-Gauss
    backend. Further solvers will be added here.

Notes
-----
Physical variables (invariants) are represented as dimensions of a multivariate
expansion. All arrays and matrices are validated in Python and passed to the
C++ backend as memory-safe Eigen objects, which keeps multidimensional
root-finding and Newton-Gauss iterations numerically stable.
"""

from ._core import (
    PolyAnsatz,
    ChebyshevAnsatz,
    uniform_collgrid,
    chebyshev_collgrid,
    FuncFromAnsatz,
)

from . import lpa

__all__ = [
    "PolyAnsatz",
    "ChebyshevAnsatz",
    "uniform_collgrid",
    "chebyshev_collgrid",
    "FuncFromAnsatz",
    "lpa",
]
