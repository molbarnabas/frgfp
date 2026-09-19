"""
Regression tests for the eager auxiliary-kernel compilation gates.

The rank-1 sensitivity kernels added in v0.1.0a5 (and the pre-existing batched
finite-difference kernel) are Numba closures, so Numba cannot disk-cache them:
they are recompiled on every process start.  The C++ solver, however, only
selects them above fixed *total* coefficient thresholds (``kBatchThreshold = 64``
and ``kSensitivityThreshold = 16``).  Compiling them for smaller problems
therefore only added start-up latency (see the O(8) Chebyshev example, which has
9 coefficients and runs far slower on v0.1.0a5 than on v0.1.0a4 despite the
solver iteration being identical).

These tests pin the gating without measuring wall-clock time:

* below the batch coefficient threshold no batched finite-difference callback
  may be built at all;
* below the coefficient threshold no sensitivity callback may be built at all;
* at or above it both addresses must be populated;
* the parallel warm-up must additionally be gated on the collocation-grid size;
* the callback cache must not leak a large problem's kernels into a small
  problem built with the same flow function.
"""

import math

import numpy as np
import pytest

import frgfp as fp
import frgfp.lpa._solver as _solver_mod
from frgfp.lpa import LPACollSolver

FLOW_PARAMS = (3.0, 8.0)


def ON_gaussian_flow(I, V, dV, ddV, params):
    """Regulator-dressed Gaussian flow: exact constant fixed point, dV-dependent."""
    d = params[0]
    N = params[1]

    omega_d = 2.0 / ((4.0 * math.pi) ** (d / 2.0) * math.gamma(d / 2.0))

    return (omega_d / d) * N / (1.0 + dV) - d * V


def expected_gaussian_V(d, N):
    """Exact constant potential of the Gaussian fixed point."""
    omega_d = 2.0 / ((4.0 * math.pi) ** (d / 2.0) * math.gamma(d / 2.0))
    return (omega_d * N) / (d ** 2)


@pytest.fixture(autouse=True)
def clear_callback_cache():
    """Keep the process-wide callback cache from hiding the gating logic."""
    _solver_mod._CALLBACK_CACHE.clear()
    yield
    _solver_mod._CALLBACK_CACHE.clear()


def make_solver(order, num_points):
    """Build a 1D Chebyshev solver with ``order + 1`` coefficients."""
    ansatz = fp.ChebyshevAnsatz(inv_dim=1, order=order, interval=(0.0, 0.4))
    grid = fp.chebyshev_collgrid(inv_dim=1, num_points=num_points, interval=(0.0, 0.4))
    return LPACollSolver(grid, ansatz, ON_gaussian_flow, inv_dim=1, param_dim=2)


def test_small_problem_skips_unused_kernels():
    """A 9-coefficient problem must not compile the batch/sensitivity kernels."""
    solver = make_solver(order=8, num_points=9)

    assert solver.ansatz.num_coeffs < _solver_mod._SENSITIVITY_THRESHOLD
    assert solver._batch_cfunc_address == 0
    assert solver._sens_cfunc_address == 0
    assert solver._sens_par_cfunc_address == 0


def test_sensitivity_kernel_gated_on_coefficient_count():
    """At the sensitivity threshold the rank-1 callbacks must exist.

    With 16 coefficients the sensitivity threshold (16) is reached but the batch
    threshold (64) is not, so the batched callback must still be absent.
    """
    solver = make_solver(order=15, num_points=16)

    assert solver.ansatz.num_coeffs >= _solver_mod._SENSITIVITY_THRESHOLD
    assert solver.ansatz.num_coeffs < _solver_mod._BATCH_THRESHOLD
    assert solver._batch_cfunc_address == 0
    assert solver._sens_cfunc_address != 0
    assert solver._sens_par_cfunc_address != 0


def test_batch_kernel_gated_on_coefficient_count():
    """The batched FD kernel is built exactly at its coefficient threshold."""
    below = make_solver(order=_solver_mod._BATCH_THRESHOLD - 2,
                        num_points=_solver_mod._BATCH_THRESHOLD - 1)
    assert below.ansatz.num_coeffs == _solver_mod._BATCH_THRESHOLD - 1
    assert below._batch_cfunc_address == 0

    at = make_solver(order=_solver_mod._BATCH_THRESHOLD - 1,
                     num_points=_solver_mod._BATCH_THRESHOLD)
    assert at.ansatz.num_coeffs == _solver_mod._BATCH_THRESHOLD
    assert at._batch_cfunc_address != 0


def test_parallel_warmup_gated_on_grid_size(monkeypatch):
    """Warm-up only happens for grids large enough to select the parallel kernel."""
    real = _solver_mod._make_sensitivity_callbacks
    seen = []

    def recording(jitted_func, param_dim, inv_dim, warm_parallel=True):
        seen.append(warm_parallel)
        return real(jitted_func, param_dim, inv_dim, warm_parallel=warm_parallel)

    monkeypatch.setattr(_solver_mod, "_make_sensitivity_callbacks", recording)

    # Enough coefficients for the rank-1 path, but a grid far below
    # _SENSITIVITY_PARALLEL_MIN_POINTS: built, but not warmed up.
    make_solver(order=20, num_points=21)
    assert seen, "The rank-1 kernel should have been built for 21 coefficients."
    assert all(flag is False for flag in seen)

    # Below the coefficient threshold the kernel must not be built at all.
    seen.clear()
    make_solver(order=8, num_points=9)
    assert seen == []


def test_cache_isolates_small_and_large_problems():
    """A cached large-problem entry must not satisfy a small-problem request."""
    large = make_solver(order=20, num_points=21)
    assert large._sens_cfunc_address != 0

    small = make_solver(order=8, num_points=9)
    assert small._batch_cfunc_address == 0
    assert small._sens_cfunc_address == 0
    assert small._sens_par_cfunc_address == 0


def ON_gaussian_flow_any_dim(I, V, dV, ddV, params):
    """Dimension-agnostic Gaussian flow (ignores dV/ddV, so it accepts vectors)."""
    d = params[0]
    N = params[1]

    omega_d = 2.0 / ((4.0 * math.pi) ** (d / 2.0) * math.gamma(d / 2.0))

    return (omega_d / d) * N - d * V


def make_solver_2d(order, points_per_axis):
    """Build a 2D Chebyshev solver with ``(order + 1)**2`` coefficients."""
    interval = (0.0, 0.4)
    ansatz = fp.ChebyshevAnsatz(
        inv_dim=2, order=(order, order), interval=(interval, interval))
    axis = np.linspace(0.0, 0.4, points_per_axis, dtype=np.float64)
    I0, I1 = np.meshgrid(axis, axis)
    grid = np.vstack([I0.ravel(), I1.ravel()]).T
    return LPACollSolver(grid, ansatz, ON_gaussian_flow_any_dim, inv_dim=2, param_dim=2)


def test_gate_uses_total_coefficient_count_regardless_of_inv_dim():
    """A 2D order-(3, 3) ansatz has 16 coefficients and must build the rank-1 kernel.

    This pins that the thresholds are compared against the *total* coefficient
    count, not a per-dimension order: 16 coefficients come from order 15 in 1D
    but already from order 3 in 2D. The batched kernel stays absent because it
    only exists for a single invariant.
    """
    solver = make_solver_2d(order=3, points_per_axis=4)

    assert solver.ansatz.num_coeffs == (3 + 1) ** 2
    assert solver.ansatz.num_coeffs >= _solver_mod._SENSITIVITY_THRESHOLD
    assert solver._sens_cfunc_address != 0
    assert solver._batch_cfunc_address == 0

    tiny = make_solver_2d(order=2, points_per_axis=3)
    assert tiny.ansatz.num_coeffs < _solver_mod._SENSITIVITY_THRESHOLD
    assert tiny._sens_cfunc_address == 0
    assert tiny._sens_par_cfunc_address == 0


def test_small_problem_still_converges():
    """The gating must not change the converged solution of a small problem."""
    solver = make_solver(order=8, num_points=9)
    grid = solver.coll_grid
    init_coeffs = np.zeros(solver.ansatz.num_coeffs)
    V_expected = np.full(grid.shape[0], expected_gaussian_V(*FLOW_PARAMS))

    for multithread in (False, True):
        result = solver.local_optimize(
            init_coeffs, flow_params=FLOW_PARAMS, multithread=multithread)
        assert result is not None, (
            f"Test failed: small problem diverged with multithread={multithread}."
        )
        np.testing.assert_allclose(result.evaluate(grid), V_expected, atol=1e-6)
