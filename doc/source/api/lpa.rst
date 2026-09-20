LPA
===

The :mod:`frgfp.lpa` submodule provides solvers for the fixed points of Local
Potential Approximation (LPA) flow equations. It bridges a user-defined Python
flow equation with the optimized C++ Newton–Gauss backend through zero-overhead
Numba C-callbacks.

.. autosummary::
   :toctree: generated

   frgfp.lpa.LPACollSolver
