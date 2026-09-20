"""
First-construction benchmarks for the frgfp LPA collocation solver.

Every other benchmark in this package builds the solver outside the timed
region and measures only the Newton iterations.  This module measures the
opposite: the **first construction** of ``LPACollSolver(...)`` -- the call a user
actually pays for at start-up, including the one-off Numba compilation of the
flow callbacks.

The benchmarked callable clears the module-level callback cache before every
call, so no timed round can reuse a previously compiled callback.  This makes
the measurement reproducible across rounds inside one pytest process and
reflects a fresh-interpreter start-up (Numba's on-disk cache may still be warm,
which is also true for every run after the user's very first one).

First-construction time is governed by *which* Numba kernels get compiled, and
that is a step function of the number of coefficients: the C++ solver only
selects the rank-1 sensitivity kernels from ``kSensitivityThreshold = 16``
coefficients and the batched finite-difference kernel from ``kBatchThreshold =
64``.  Compilation itself does not scale with the coefficient count, and the
grid only enters through the (comparatively negligible) basis-matrix assembly.
Each case therefore uses a **square** system (``num_points == num_coeffs == n``)
and only ``n`` varies, bracketing both thresholds -- including ``n = 8``, which
is below the 16-coefficient sensitivity gate.
"""

import math

import pytest

import frgfp as fp
import frgfp.lpa._solver as _solver_mod
from frgfp.lpa import LPACollSolver

#: Square-system sizes compared (``num_points == num_coeffs == n``).  8 is below
#: ``kSensitivityThreshold = 16``, 16 and 32 sit between the two gates, and 64
#: and above cross ``kBatchThreshold = 64``.
SYSTEM_SIZES = (8, 16, 32, 64, 128, 256)

#: Cold constructions timed per benchmark case (each clears the callback cache).
COLD_ROUNDS = 3


def construction_flow(I, V, dV, ddV, params):
    """dV-dependent Gaussian flow: exact constant fixed point, non-trivial f_dV."""
    d = params[0]
    N = params[1]

    omega_d = 2.0 / ((4.0 * math.pi) ** (d / 2.0) * math.gamma(d / 2.0))

    return (omega_d / d) * N / (1.0 + dV) - d * V


def make_ansatz(num_coeffs):
    """A ``PolyAnsatz`` with exactly ``num_coeffs`` basis functions."""
    return fp.PolyAnsatz(inv_dim=1, order=num_coeffs - 1, center=0.15)


def make_grid(num_points):
    """A uniform 1D collocation grid with ``num_points`` points."""
    return fp.uniform_collgrid(inv_dim=1, num_points=num_points, interval=(0.0, 0.4))


def first_construction(grid, ansatz):
    """Build a solver from a cleared callback cache (the first-construction path).

    Clearing ``_CALLBACK_CACHE`` is what forces ``_build_callbacks`` to recompile
    the Numba C-callbacks on every call; without it the Python-level cache would
    make every round after the first look artificially cheap.
    """
    _solver_mod._CALLBACK_CACHE.clear()
    return LPACollSolver(grid, ansatz, construction_flow, inv_dim=1, param_dim=2)


@pytest.fixture(params=SYSTEM_SIZES)
def construction_case(request):
    """A square (ansatz, grid, size) triple: ``num_points == num_coeffs == size``."""
    size = request.param
    return make_ansatz(size), make_grid(size), size


@pytest.mark.benchmark(group="solver_construction")
def test_solver_construction_benchmark(benchmark, construction_case):
    """Benchmark the *first* construction of square systems of increasing size.

    Only the system size ``n`` varies (``num_points == num_coeffs == n``), so the
    result exposes the step-wise cost of the kernel-compilation gates rather than
    a grid-size trend.  ``pedantic`` mode is used so exactly ``COLD_ROUNDS`` cold
    constructions are timed per case (a single warm-up round, excluded from the
    statistics, absorbs the very first Numba/disk-cache initialisation).
    """
    ansatz, grid, size = construction_case

    solver = benchmark.pedantic(
        first_construction,
        args=(grid, ansatz),
        rounds=COLD_ROUNDS,
        warmup_rounds=1,
        iterations=1,
    )

    assert solver is not None, (
        f"Test failed: construction returned None for a {size}x{size} system."
    )
    assert solver.coll_grid.shape[0] == size
    assert solver.ansatz.num_coeffs == size
