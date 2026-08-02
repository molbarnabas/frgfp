"""
Local Potential Approximation (LPA) Fixed Point Solver.

This module provides a high-performance collocation solver for finding the 
fixed points of Functional Renormalization Group (FRG) flow equations.
It bridges a user-defined Python flow equation with an optimized C++ 
Newton-Gauss backend using zero-overhead Numba C-callbacks.
"""

from ._solver import LPACollSolver

__all__ = ["LPACollSolver"]