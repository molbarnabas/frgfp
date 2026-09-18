"""
End-to-end O(8) multistart tests for the frgfp LPA collocation solver.

Runs the built-in multistart and checks every converged result against the
reference coefficients in ``o8_fp.dat`` using a relative tolerance.
"""

import pytest
import numpy as np
import math
from pathlib import Path
import frgfp as fp

N_CHECK = 6
RTOL = 5e-3

EXPECTED_COEFFS = np.loadtxt(Path(__file__).parent / 'o8_fp.dat')
REF_SCALE = np.max(np.abs(EXPECTED_COEFFS[:N_CHECK]))

def ON_flowrhs(I, V, dV, ddV, params):
    """The O(N) model flow right-hand side."""
    d = params[0]
    N = params[1]
    
    omega_d = 2.0 / ((4.0 * math.pi)**(d / 2.0) * math.gamma(d / 2.0))
    
    rhs = (omega_d / d) * (
        1.0 / (1.0 + dV + 2.0 * I * ddV) + 
        (N - 1.0) / (1.0 + dV) 
    ) - d * V + (d - 2.0) * I * dV 
    
    return rhs

def high_perturb(rho):
    """Perturbative O(8) solution for the O(N) model in d=3 dimensions."""
    return -0.341419*rho + 0.876914*rho**2 + 1.45014*rho**3 + 1.5513*rho**4 - \
           0.167052*rho**5 - 1.55923*rho**6 + 10.488*rho**7 + 37.6868*rho**8 + 0.0683767

@pytest.fixture(scope="module")
def o8_multistart_context():
    """Build the O(8) solver and 50 perturbed initial guesses once (outside the timed region)."""
    ansatz = fp.PolyAnsatz(inv_dim=1, order=100, center=0.15)
    coll_grid = fp.uniform_collgrid(inv_dim=1, num_points=101, interval=(0, 0.4))
    
    solver = fp.lpa.LPACollSolver(
        ansatz=ansatz, 
        coll_grid=coll_grid, 
        flowrhs_func=ON_flowrhs, 
        inv_dim=1, 
        param_dim=2
    )
    
    offsets = np.linspace(0, 2, 50)
    init_coeffs_list = []
    
    for offset in offsets:
        vals = high_perturb(coll_grid) + offset
        fit_poly = fp.FuncFromAnsatz.from_grid(
            ansatz=ansatz, 
            grid=coll_grid, 
            vals=vals
        )
        init_coeffs_list.append(fit_poly.coeffs)
        
    return solver, np.array(init_coeffs_list)

def test_o8_multistart_benchmark(benchmark, o8_multistart_context):
    """
    Benchmark only the built-in multistart (the solver and initial guesses are
    built by the fixture, outside the timed region) and verify that each
    converged result matches ``o8_fp.dat`` within a relative tolerance.
    """
    solver, init_coeffs_array = o8_multistart_context
    
    multistart_results = benchmark(
        solver.multistart_optimize,
        init_coeffs_list=init_coeffs_array,
        flow_params=(3.0, 8.0),
        maxiter=10000,
        tol=1e-7
    )
    
    valid_results = [FP for FP in multistart_results if FP is not None]
    
    assert len(valid_results) > 0, (
        "Test failed: All 50 optimization attempts diverged (returned None)."
    )
    
    for FP in valid_results:
        rel_deviation = np.max(np.abs(FP.coeffs[:N_CHECK] - EXPECTED_COEFFS[:N_CHECK])) / REF_SCALE
        assert rel_deviation <= RTOL, (
            f"Test failed: converged result relative deviation "
            f"{rel_deviation:e} is greater than {RTOL:e}."
        )
