"""
Core module containing the foundational structures of the frgfp package.

This module provides the primary classes and functions required to set up
the function space for solving FRG flow equations. It includes polynomial
and orthogonal basis expansions (ansätze) evaluated via the C++ Eigen backend,
as well as rigorous multidimensional collocation grid generators and function
evaluators.

All inputs are heavily type-checked to ensure seamless memory translation
between Python/NumPy and the Pybind11 C++ interface.
"""

from ._ansatz import PolyAnsatz, ChebyshevAnsatz
from ._grids import uniform_collgrid, chebyshev_collgrid
from ._func import FuncFromAnsatz

__all__ = [
    "PolyAnsatz",
    "ChebyshevAnsatz",
    "uniform_collgrid",
    "chebyshev_collgrid",
    "FuncFromAnsatz",
]