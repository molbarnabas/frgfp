"""
End-to-end O(8) fixed-point tests for the frgfp LPA collocation solver.

Solves the O(8) model fixed point in both threading modes and checks the leading
coefficients against the reference data in ``o8_fp.dat`` using a relative
tolerance.
"""

import pytest
import numpy as np
import math
from pathlib import Path
import frgfp as fp

# Number of leading coefficients compared against the reference.
N_CHECK = 6
# Relative tolerance: |calc - ref|_inf / |ref|_inf over the compared coefficients.
RTOL = 5e-3

EXPECTED_COEFFS = np.loadtxt(Path(__file__).parent / 'o8_fp.dat')
REF_SCALE = np.max(np.abs(EXPECTED_COEFFS[:N_CHECK]))

def ON_flowrhs(I, V, dV, ddV, params):
    """The O(N) model flow equations"""
    d = params[0]
    N = params[1]
    
    omega_d = 2.0 / ((4.0 * math.pi)**(d / 2.0) * math.gamma(d / 2.0))
    
    rhs = (omega_d / d) * (
        1.0 / (1.0 + dV + 2.0 * I * ddV) + 
        (N - 1.0) / (1.0 + dV) 
    ) - d * V + (d - 2.0) * I * dV 
    
    return rhs

def high_perturb(rho):
    """Perturbative solution for the O(8) model in d=3 dimensions"""
    return -0.341419*rho + 0.876914*rho**2 + 1.45014*rho**3 + 1.5513*rho**4 - \
           0.167052*rho**5 - 1.55923*rho**6 + 10.488*rho**7 + 37.6868*rho**8 + 0.0683767

@pytest.fixture(scope="module")
def o8_solver_and_guess():
    """Build the O(8) ansatz, solver and initial guess once (outside the timed region)."""
    ansatz = fp.PolyAnsatz(inv_dim=1, order=100, center=0.15)
    coll_grid = fp.uniform_collgrid(inv_dim=1, num_points=101, interval=(0, 0.4))
    
    solver = fp.lpa.LPACollSolver(
        ansatz=ansatz, 
        coll_grid=coll_grid, 
        flowrhs_func=ON_flowrhs, 
        inv_dim=1, 
        param_dim=2
    )
    
    fit_poly = fp.FuncFromAnsatz.from_grid(
        ansatz=ansatz, 
        grid=coll_grid, 
        vals=high_perturb(coll_grid)+5  # Adding noise
    )
    
    return solver, fit_poly.coeffs

@pytest.mark.parametrize("multithread", [False, True])
@pytest.mark.benchmark(group="o8_fixed_point")
def test_o8_fixed_point_benchmark(benchmark, o8_solver_and_guess, multithread):
    """
    Benchmark only the O(8) fixed-point optimization (the solver and initial
    guess are built by the fixture, outside the timed region) for each threading
    mode, and check the leading coefficients against ``o8_fp.dat`` within a
    relative tolerance.
    """
    solver, init_coeffs = o8_solver_and_guess
    
    result = benchmark(
        solver.local_optimize,
        init_coeffs,
        flow_params=(3.0, 8.0),
        maxiter=10000,
        tol=1e-7,
        multithread=multithread
    )
    
    assert result is not None, (
        f"Test failed: multithread={multithread} did not converge."
    )
    
    rel_deviation = np.max(np.abs(result.coeffs[:N_CHECK] - EXPECTED_COEFFS[:N_CHECK])) / REF_SCALE
    
    assert rel_deviation <= RTOL, (
        f"Test failed: multithread={multithread} relative deviation "
        f"of the first {N_CHECK} coefficients is {rel_deviation:e} > {RTOL:e}."
    )
