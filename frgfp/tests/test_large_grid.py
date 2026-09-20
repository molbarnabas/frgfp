"""
Large-grid scaling benchmarks for the frgfp LPA collocation solver.

The existing end-to-end tests (``test_o8_fp.py``, ``test_o8_multistart.py``) use
a 101-point collocation grid, which stays below every internal parallelisation
threshold of the solver.  This module benchmarks grids that are large enough to
cross them:

* ``kSensitivityParallelMinPoints = 512`` -- from 512 collocation points upwards
  the rank-1 sensitivity kernel is spread over all cores when
  ``multithread=True``.  Below that both threading modes run the same serial
  kernel, so they are expected to be equally fast.
* ``kSensitivityThreshold = 16`` -- below this many coefficients the solver keeps
  the scalar finite-difference Jacobian; above it the rank-1 sensitivity
  Jacobian is used for every invariant dimension.  The 2D case below uses 289
  coefficients on a 17x17 grid, which exercises the tensor rank-1 kernel for
  ``inv_dim == 2`` (14 flow evaluations per point instead of 578 calls).
* ``kParallelThreshold = 256`` -- the scalar finite-difference Jacobian (the
  fallback when the rank-1 kernels are unavailable) is parallelised once the
  number of coefficients reaches this value.
* ``kSensitivityParallelMinPoints = 512`` -- the rank-1 sensitivity kernel is
  spread over all cores only from this many collocation points upwards.  The 2D
  case below stays under it, so it is equally fast in both threading modes.

Note the rank-1 Jacobian *assembly* is intentionally serial: it is a
bandwidth-bound streaming loop that gets no benefit from threads (it measured
0.92x-0.98x when parallelised), so ``multithread`` only affects the
compute-bound flow-evaluation kernels.

Every benchmark performs a fixed number of Newton steps, so both threading
modes execute identical work and the comparison is not biased by differing
convergence speeds.  Correctness is asserted separately: each result must
reproduce the analytic constant Gaussian fixed point.
"""

import math

import numpy as np
import pytest

import frgfp as fp
from frgfp.lpa import LPACollSolver

#: 1D collocation grid sizes.  256 and 512 bracket the sensitivity-kernel
#: parallelisation threshold; 1024 and 2048 show the scaling beyond it.
GRID_SIZES_1D = (256, 512, 1024, 2048)

#: Polynomial order used for the 1D scaling study (yields 21 coefficients, so
#: the system is strongly overdetermined and the rank-1 path is active).
POLY_ORDER_1D = 20

#: Newton steps performed per benchmark iteration (fixed work, see module docs).
NUM_ITER_1D = 2

#: 2D Chebyshev order.  (16, 16) yields 289 coefficients, which exceeds the
#: solver's 256-coefficient threshold for the parallel finite-difference path.
CHEB_ORDER_2D = 16

#: Newton steps performed per 2D benchmark iteration.
NUM_ITER_2D = 1

#: Flow parameters (d, N) used throughout.
FLOW_PARAMS = (3.0, 8.0)


def ON_gaussian_regulated_flow(I, V, dV, ddV, params):
    """
    Regulator-dressed Gaussian limit of the O(N) Wilson-Fisher flow (1D).

    The ``1 / (1 + dV)`` factor makes the flow depend on the field derivative,
    so the rank-1 sensitivity Jacobian picks up a non-trivial ``f_dV``
    component.  The exact fixed point is still the constant potential
    ``V = omega_d * N / d**2``, where ``dV == 0``.
    """
    d = params[0]
    N = params[1]

    omega_d = 2.0 / ((4.0 * math.pi) ** (d / 2.0) * math.gamma(d / 2.0))

    return (omega_d / d) * N / (1.0 + dV) - d * V


def ON_gaussian_coupled_flow(I, V, dV, ddV, params):
    """
    Gaussian limit of the O(N) flow with derivative couplings (``inv_dim >= 2``).

    Unlike a potential-only flow, the regulator ``1 / (1 + dV[0] + I[0] *
    ddV[0, 0])`` makes ``f_dV`` and ``f_ddV`` non-zero, so the rank-1 Jacobian
    actually exercises its gradient and Hessian blocks.  Both extra terms vanish
    at ``dV == 0`` and ``ddV == 0``, so the exact fixed point is still the
    constant potential ``V = omega_d * N / d**2``.

    Note this form indexes ``I``/``dV``/``ddV`` as arrays, so it is only valid
    for ``inv_dim >= 2``.
    """
    d = params[0]
    N = params[1]

    omega_d = 2.0 / ((4.0 * math.pi) ** (d / 2.0) * math.gamma(d / 2.0))

    return (omega_d / d) * N / (1.0 + dV[0] + I[0] * ddV[0, 0]) - d * V


def linear_coupled_flow_2d(I, V, dV, ddV, params):
    """
    Affine flow in ``(V, dV, ddV)`` for two invariants.

    The Jacobian of an affine flow is known exactly, which lets the rank-1
    assembly be checked against an analytic reference.
    """
    return (1.0 * V + 0.5 * dV[0] - 0.25 * dV[1]
            + 0.1 * ddV[0, 0] + 0.2 * ddV[0, 1] + 0.2 * ddV[1, 0]
            - 0.05 * ddV[1, 1] + 0.3)


def linear_coupled_flow_3d(I, V, dV, ddV, params):
    """Affine flow in ``(V, dV, ddV)`` for three invariants (analytic Jacobian)."""
    return (1.0 * V + 0.5 * dV[0] - 0.25 * dV[1] + 0.15 * dV[2]
            + 0.1 * ddV[0, 0] + 0.2 * ddV[0, 1] + 0.2 * ddV[1, 0]
            - 0.05 * ddV[1, 1] - 0.07 * ddV[0, 2] - 0.07 * ddV[2, 0]
            + 0.03 * ddV[2, 2] + 0.3)


def expected_gaussian_V(d, N):
    """Exact constant potential of the Gaussian fixed point."""
    omega_d = 2.0 / ((4.0 * math.pi) ** (d / 2.0) * math.gamma(d / 2.0))

    return (omega_d * N) / (d ** 2)


#: (inv_dim, order, num_points, flow, f_dV coefficients, f_ddV coefficients)
_LINEAR_CASES = (
    (2, (3, 3), (4, 4), linear_coupled_flow_2d,
     (0.5, -0.25), ((0.1, 0.2), (0.2, -0.05))),
    (3, (2, 2, 2), (3, 3, 3), linear_coupled_flow_3d,
     (0.5, -0.25, 0.15),
     ((0.1, 0.2, -0.07), (0.2, -0.05, 0.0), (-0.07, 0.0, 0.03))),
)


@pytest.mark.parametrize(
    "inv_dim,order,num_points,flow,lin_b,lin_c", _LINEAR_CASES,
    ids=["inv_dim=2", "inv_dim=3"])
def test_rank1_jacobian_matches_analytic(inv_dim, order, num_points, flow, lin_b, lin_c):
    """
    Check the rank-1 Jacobian assembly against an analytically known Jacobian.

    The flow is affine in ``(V, dV, ddV)``, so its exact Jacobian is
    ``M_val + sum_a lin_b[a] * M_grad_a + sum_ab lin_c[a][b] * M_hess_ab``.
    A single Newton step must therefore solve that linear system, so the step's
    *backward error* is asserted.  A direct coefficient comparison is not usable
    here because the collocation system is badly conditioned
    (``cond(J) ~ 3e9`` with these settings), which amplifies any tiny difference
    in the computed Jacobian into a large change of the step.  An indexing error
    in the assembly would instead make the backward error O(1).
    """
    interval = tuple((0.0, 0.4) for _ in range(inv_dim))

    ansatz = fp.ChebyshevAnsatz(inv_dim=inv_dim, order=order, interval=interval)
    coll_grid = fp.uniform_collgrid(
        inv_dim=inv_dim, num_points=num_points, interval=interval)

    solver = LPACollSolver(
        ansatz=ansatz,
        coll_grid=coll_grid,
        flowrhs_func=flow,
        inv_dim=inv_dim,
        param_dim=2,
    )

    M_val = ansatz.evaluate_basis(coll_grid)
    M_grad = ansatz.evaluate_basis_grad(coll_grid)
    M_hess = ansatz.evaluate_basis_hess(coll_grid)

    J = M_val.copy()
    for a in range(inv_dim):
        J = J + lin_b[a] * M_grad[a]
    for a in range(inv_dim):
        for b in range(inv_dim):
            J = J + lin_c[a][b] * M_hess[a * inv_dim + b]

    init_coeffs = np.linspace(0.02, 0.1, ansatz.num_coeffs)
    residual = J @ init_coeffs + 0.3

    result = solver.local_optimize_fixed_iterations(
        init_coeffs, num_iter=1, flow_params=FLOW_PARAMS, multithread=False)

    assert result is not None, (
        f"Test failed: inv_dim={inv_dim} produced a non-finite residual."
    )

    step = result.coeffs - init_coeffs
    backward_error = np.linalg.norm(J @ step + residual) / (
        np.linalg.norm(J, "fro") * np.linalg.norm(step) + np.linalg.norm(residual))

    assert backward_error < 1e-8, (
        f"Test failed: rank-1 Jacobian backward error {backward_error:e} "
        f"is too large for inv_dim={inv_dim}, so the assembly is inconsistent "
        "with the analytic Jacobian."
    )



@pytest.fixture(scope="module", params=GRID_SIZES_1D)
def large_grid_1d(request):
    """Build (once per grid size) a 1D solver on a large collocation grid."""
    num_points = request.param
    ansatz = fp.PolyAnsatz(inv_dim=1, order=POLY_ORDER_1D, center=0.1)

    coll_grid = fp.uniform_collgrid(
        inv_dim=1, num_points=num_points, interval=(0.0, 0.4))

    solver = LPACollSolver(
        ansatz=ansatz,
        coll_grid=coll_grid,
        flowrhs_func=ON_gaussian_regulated_flow,
        inv_dim=1,
        param_dim=2,
    )

    return solver, coll_grid, num_points


@pytest.fixture(scope="module")
def large_grid_2d():
    """Build a 2D solver whose coefficient count crosses the FD threshold."""
    n_per_dim = CHEB_ORDER_2D + 1
    interval = ((0.0, 0.4), (0.0, 0.4))

    ansatz = fp.ChebyshevAnsatz(
        inv_dim=2, order=(CHEB_ORDER_2D, CHEB_ORDER_2D), interval=interval)

    coll_grid = fp.uniform_collgrid(
        inv_dim=2, num_points=(n_per_dim, n_per_dim), interval=interval)

    solver = LPACollSolver(
        ansatz=ansatz,
        coll_grid=coll_grid,
        flowrhs_func=ON_gaussian_coupled_flow,
        inv_dim=2,
        param_dim=2,
    )

    return solver, coll_grid


@pytest.mark.parametrize("multithread", [False, True])
@pytest.mark.benchmark(group="large_grid_1d")
def test_large_grid_benchmark(benchmark, large_grid_1d, multithread):
    """
    Benchmark fixed-work Newton steps on 1D grids of increasing size.

    Crossing 512 points should make ``multithread=True`` visibly faster, while
    below it both modes run the identical serial kernel and should match.
    """
    solver, coll_grid, num_points = large_grid_1d
    init_coeffs = np.zeros(solver.ansatz.num_coeffs)

    result = benchmark(
        solver.local_optimize_fixed_iterations,
        init_coeffs,
        num_iter=NUM_ITER_1D,
        flow_params=FLOW_PARAMS,
        multithread=multithread,
    )

    assert result is not None, (
        f"Test failed: {num_points} points with multithread={multithread} "
        "produced a non-finite residual."
    )

    V_eval = result.evaluate(coll_grid)
    V_expected = np.full(coll_grid.shape[0], expected_gaussian_V(*FLOW_PARAMS))

    np.testing.assert_allclose(V_eval, V_expected, atol=1e-6)


def test_large_grid_threading_consistency(large_grid_1d):
    """
    Check that both threading modes converge to the same large-grid solution.

    The rank-1 Jacobian is mathematically identical in both modes, so the
    converged coefficient vectors must agree and both must reproduce the exact
    Gaussian fixed point.
    """
    solver, coll_grid, num_points = large_grid_1d
    init_coeffs = np.zeros(solver.ansatz.num_coeffs)

    serial = solver.local_optimize(
        init_coeffs, maxiter=100, tol=1e-10,
        flow_params=FLOW_PARAMS, multithread=False)

    threaded = solver.local_optimize(
        init_coeffs, maxiter=100, tol=1e-10,
        flow_params=FLOW_PARAMS, multithread=True)

    assert serial is not None, (
        f"Test failed: serial run diverged at {num_points} points."
    )
    assert threaded is not None, (
        f"Test failed: threaded run diverged at {num_points} points."
    )

    V_expected = np.full(coll_grid.shape[0], expected_gaussian_V(*FLOW_PARAMS))

    np.testing.assert_allclose(serial.evaluate(coll_grid), V_expected, atol=1e-6)
    np.testing.assert_allclose(threaded.evaluate(coll_grid), V_expected, atol=1e-6)

    np.testing.assert_allclose(
        serial.coeffs, threaded.coeffs, rtol=0.0, atol=1e-6,
        err_msg="Test failed: threading mode changed the converged coefficients.")

@pytest.mark.benchmark(group="large_grid_2d")
@pytest.mark.parametrize("multithread", [False, True])
def test_large_grid_2d_benchmark(benchmark, large_grid_2d, multithread):
    """
    Benchmark the 2D rank-1 sensitivity Jacobian on a larger grid.

    ``inv_dim == 2`` uses the tensor rank-1 kernel: ``f_V``, a length-2 ``f_dV``
    and a 2x2 ``f_ddV`` per point, so 14 flow evaluations replace the 578
    callback invocations of the coefficient-space finite-difference Jacobian.
    The Jacobian is parallelised over collocation points when
    ``multithread=True``.
    """
    solver, coll_grid = large_grid_2d
    init_coeffs = np.zeros(solver.ansatz.num_coeffs)

    result = benchmark(
        solver.local_optimize_fixed_iterations,
        init_coeffs,
        num_iter=NUM_ITER_2D,
        flow_params=FLOW_PARAMS,
        multithread=multithread,
    )

    assert result is not None, (
        f"Test failed: 2D run with multithread={multithread} produced a "
        "non-finite residual."
    )

    V_eval = result.evaluate(coll_grid)
    V_expected = np.full(coll_grid.shape[0], expected_gaussian_V(*FLOW_PARAMS))

    np.testing.assert_allclose(V_eval, V_expected, atol=1e-6)
