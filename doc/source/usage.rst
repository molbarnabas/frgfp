Usage
=====

This page walks through the whole public API of FRGfp with one complete,
runnable example: the O(N) Wilson-Fisher flow at order
:math:`\mathcal{O}(\bar{\rho}^8)` in the local potential approximation. The
snippets below are meant to be read in order; together they form a single
script.

A Jupyter notebook version of this page is available in
``examples/usage.ipynb`` in the repository.

The flow equation
-----------------

The flow is a user-defined function with the signature
``(I, V, dV, ddV, params)``. It is JIT-compiled with Numba, so define it at
module level to enable disk caching:

.. code-block:: python

   import math
   import numpy as np
   import frgfp


   def ON_flowrhs(I, V, dV, ddV, params):
       """O(N) flow right-hand side in the local potential approximation."""
       d, N = params
       omega_d = 2.0 / ((4.0 * math.pi) ** (d / 2.0) * math.gamma(d / 2.0))

       return (omega_d / d) * (
           1.0 / (1.0 + dV + 2.0 * I * ddV) + (N - 1.0) / (1.0 + dV)
       ) - d * V + (d - 2.0) * I * dV

Basis expansions
----------------

A basis is created with :class:`~frgfp.ChebyshevAnsatz` (orthogonal, on a
physical interval) or :class:`~frgfp.PolyAnsatz` (a Taylor-like expansion
around a centre). Both work for an arbitrary number of invariants:

.. code-block:: python

   # 1D Chebyshev basis of order 8 on the interval [0, 0.4] -> 9 coefficients.
   ansatz = frgfp.ChebyshevAnsatz(inv_dim=1, order=8, interval=(0.0, 0.4))
   print(ansatz.inv_dim, ansatz.num_coeffs)     # 1 9

   # 2D polynomial basis of order (2, 2) around (0, 0) -> 9 coefficients.
   poly = frgfp.PolyAnsatz(inv_dim=2, order=(2, 2), center=(0.0, 0.0))
   print(poly.inv_dim, poly.num_coeffs)         # 2 9

Collocation grids
-----------------

Grids are returned as 2D arrays of shape ``(N_points, inv_dim)`` in row-major
(lexicographic) order. Use :func:`~frgfp.chebyshev_collgrid` for the basis
nodes and :func:`~frgfp.uniform_collgrid` for equidistant points:

.. code-block:: python

   # Chebyshev-Gauss-Lobatto nodes: the natural collocation points of the solver.
   coll_grid = frgfp.chebyshev_collgrid(inv_dim=1, num_points=9, interval=(0.0, 0.4))

   # Equidistant grid: convenient for fitting a function from sampled values.
   fit_grid = frgfp.uniform_collgrid(inv_dim=1, num_points=9, interval=(0.0, 0.4))

   print(coll_grid.shape, fit_grid.shape)       # (9, 1) (9, 1)

Function expansions
-------------------

:class:`~frgfp.FuncFromAnsatz` represents a function expanded in the basis of
an ansatz. The coefficients can be given explicitly, or fitted to sampled
values with :meth:`~frgfp.FuncFromAnsatz.from_grid`, which solves the
least-squares problem through Eigen's SVD pseudo-inverse:

.. code-block:: python

   def perturb(rho):
       """Perturbative profile used as the initial guess."""
       return (-0.341419 * rho + 0.876914 * rho**2 + 1.45014 * rho**3
               + 1.5513 * rho**4 - 0.167052 * rho**5 - 1.55923 * rho**6
               + 10.488 * rho**7 + 37.6868 * rho**8 + 0.0683767)


   fit = frgfp.FuncFromAnsatz.from_grid(
       ansatz=ansatz, grid=fit_grid, vals=perturb(fit_grid))

   print(fit.num_coeffs, fit.coeffs.shape)      # 9 (9,)
   print(fit.evaluate(fit_grid))                # V on the fit grid
   print(fit.evaluate_grad(fit_grid)[0])        # dV/drho (length inv_dim)
   print(fit.evaluate_hess(fit_grid)[0])        # d2V/drho2 (length inv_dim**2)

The constructor and the ``coeffs`` property work on a coefficient vector that
can be updated in place:

.. code-block:: python

   func = frgfp.FuncFromAnsatz(ansatz=ansatz, coeffs=fit.coeffs)
   func.coeffs = 0.5 * fit.coeffs              # updates the expansion in place
   print(func.ansatz.num_coeffs)               # 9

:meth:`~frgfp.FuncFromAnsatz.evaluate_rhs` evaluates the expansion, its
derivatives and the flow equation in one call:

.. code-block:: python

   x = np.linspace(0.0, 0.4, 100)
   rhs = fit.evaluate_rhs(x, 1, ON_flowrhs, (3.0, 8.0))
   print(rhs.shape)                             # (100,)

Solving an LPA fixed point
--------------------------

:class:`frgfp.lpa.LPACollSolver` performs the Newton-Gauss iteration. It takes
the collocation grid, the ansatz, the flow function and the two dimensions:
``inv_dim`` (invariants) and ``param_dim`` (flow parameters).

.. code-block:: python

   solver = frgfp.lpa.LPACollSolver(
       ansatz=ansatz, coll_grid=coll_grid, flowrhs_func=ON_flowrhs,
       inv_dim=1, param_dim=2)

   print(solver.inv_dim, solver.param_dim)      # 1 2

   fixed_point = solver.local_optimize(
       init_coeffs=fit.coeffs, flow_params=(3.0, 8.0),
       maxiter=1000, tol=1e-8, multithread=True)

   if fixed_point is not None:
       print(fixed_point.evaluate(coll_grid))   # the fixed-point potential

At the fixed point the right-hand side of the flow vanishes, which is a
convenient convergence check:

.. code-block:: python

   residual = fixed_point.evaluate_rhs(coll_grid, 1, ON_flowrhs, (3.0, 8.0))
   print(np.max(np.abs(residual)))              # ~1e-11

The solver offers three further entry points. Each returns ``None`` for the
runs that did not converge:

.. code-block:: python

   # A fixed number of iterations, without a convergence criterion.
   stepped = solver.local_optimize_fixed_iterations(
       init_coeffs=fit.coeffs, flow_params=(3.0, 8.0),
       num_iter=3, multithread=True)

   # Follow a path of flow parameters, restarting from the previous solution.
   path = solver.param_path_following(
       init_coeffs=fixed_point.coeffs,
       flow_params_path=[(3.0, 8.0), (3.0, 9.0)],
       maxiter=1000, tol=1e-8, multithread=True)

   # Several initial guesses at once (the starts run in parallel).
   starts = solver.multistart_optimize(
       init_coeffs_list=[fixed_point.coeffs, fit.coeffs],
       flow_params=(3.0, 8.0), maxiter=1000, tol=1e-8)

   print([fp is not None for fp in path])       # [True, True]
   print([fp is not None for fp in starts])     # [True, True]

The result can be compared with the perturbative input by evaluating both on a
dense grid (``fixed_point.evaluate(x)`` versus ``fit.evaluate(x)``); the
collocation points themselves are exactly the rows of ``coll_grid``.
