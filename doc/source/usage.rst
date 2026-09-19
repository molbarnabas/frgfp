Usage
=====

Basis expansions and grids
--------------------------

Every functional is described by an *ansatz* (a tensor-product basis) and is
evaluated on a *collocation grid*. Both are created for an arbitrary invariant
dimension ``inv_dim``.

.. code-block:: python

   import numpy as np
   import frgfp

   # A 1D Chebyshev basis of order 5 on [-1, 1].
   ansatz = frgfp.ChebyshevAnsatz(inv_dim=1, order=5, interval=(-1.0, 1.0))

   # A uniform collocation grid with 20 points on the same interval.
   grid = frgfp.uniform_collgrid(inv_dim=1, num_points=20, interval=(-1.0, 1.0))

   # Pair the ansatz with a coefficient vector.
   coeffs = np.random.rand(ansatz.num_coeffs)
   func = frgfp.FuncFromAnsatz(ansatz=ansatz, coeffs=coeffs)

   values = func.evaluate(grid)
   gradient = func.evaluate_grad(grid)
   hessian = func.evaluate_hess(grid)

For multi-dimensional problems ``order`` and ``interval`` become tuples, one
entry per invariant:

.. code-block:: python

   ansatz = frgfp.ChebyshevAnsatz(
       inv_dim=2, order=(2, 2), interval=((0.0, 0.4), (0.0, 0.4)))
   grid = frgfp.chebyshev_collgrid(
       inv_dim=2, num_points=(4, 4), interval=((0.0, 0.4), (0.0, 0.4)))

Coefficients can also be fitted directly to target values on a grid with the
least-squares pseudo-inverse:

.. code-block:: python

   func = frgfp.FuncFromAnsatz.from_grid(ansatz=ansatz, grid=grid, vals=values)

Solving an LPA fixed point
--------------------------

:class:`frgfp.lpa.LPACollSolver` finds fixed points of a flow equation

.. math::

   \partial_t V = \mathcal{F}(I, V, \partial V, \partial^2 V; \text{params}),

where the right-hand side is a user-supplied function with the signature
``(I, V, dV, ddV, params)``. The function is JIT-compiled with Numba, so it
should be defined at module level to enable disk caching.

.. code-block:: python

   import math
   import numpy as np
   import frgfp

   def gaussian_flow(I, V, dV, ddV, params):
       """O(N) Gaussian limit: a purely potential-driven flow."""
       d, N = params
       omega_d = 2.0 / ((4.0 * math.pi) ** (d / 2.0) * math.gamma(d / 2.0))
       return (omega_d / d) * N - d * V

   inv_dim, param_dim = 1, 2
   ansatz = frgfp.PolyAnsatz(inv_dim=inv_dim, order=2, center=0.0)
   grid = frgfp.uniform_collgrid(
       inv_dim=inv_dim, num_points=5, interval=(0.0, 0.4))

   solver = frgfp.lpa.LPACollSolver(
       grid, ansatz, gaussian_flow, inv_dim=inv_dim, param_dim=param_dim)

   init_coeffs = np.zeros(ansatz.num_coeffs)
   flow_params = np.array([3.0, 8.0])

   fixed_point = solver.local_optimize(init_coeffs, flow_params=flow_params)
   if fixed_point is not None:
       print(fixed_point.evaluate(grid))

Set ``multithread=True`` to parallelise the flow-evaluation kernels and follow a
parameter path with :meth:`~frgfp.lpa.LPACollSolver.param_path_following`, or run
several initial guesses at once with
:meth:`~frgfp.lpa.LPACollSolver.multistart_optimize`.
