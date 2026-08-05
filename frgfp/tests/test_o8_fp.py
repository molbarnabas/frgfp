import pytest
import numpy as np
import math
from pathlib import Path
import frgfp as fp

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

def run_o8_multithreadTrue():
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
    
    FP = solver.local_optimize(
        init_coeffs=fit_poly.coeffs, 
        flow_params=(3.0, 8.0), 
        maxiter=10000, 
        tol=1e-7, 
        multithread=True
    )
    
    return FP.coeffs

def test_o8_multithreadTrue_benchmark(benchmark):
    """
    Benchmark test to check the fixed point coefficients against expected values from 'o8_fp.dat'.
    The test will pass if the maximum deviation from the first 6 expected coefficients is within the threshold.
    """
    current_dir = Path(__file__).parent
    data_file_path = current_dir / 'o8_fp.dat'
    
    expected_coeffs = np.loadtxt(data_file_path)
    
    calculated_coeffs = benchmark(run_o8_multithreadTrue)
    
    threshold = 1e-4
    
    max_deviation = np.max(np.abs(calculated_coeffs - expected_coeffs)[:6]) 
    
    assert max_deviation <= threshold, (
        f"Test failed: Maximum deviation from expected coefficients is {max_deviation:e}"
    )

def run_o8_multithreadFalse():
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
    
    FP = solver.local_optimize(
        init_coeffs=fit_poly.coeffs, 
        flow_params=(3.0, 8.0), 
        maxiter=10000, 
        tol=1e-7, 
        multithread=False
    )
    
    return FP.coeffs

def test_o8_multithreadFalse_benchmark(benchmark):
    """
    Benchmark test to check the fixed point coefficients against expected values from 'o8_fp.dat'.
    The test will pass if the maximum deviation from the first 6 expected coefficients is within the threshold.
    """
    current_dir = Path(__file__).parent
    data_file_path = current_dir / 'o8_fp.dat'
    
    expected_coeffs = np.loadtxt(data_file_path)
    
    calculated_coeffs = benchmark(run_o8_multithreadFalse)
    
    threshold = 1e-3
    
    max_deviation = np.max(np.abs(calculated_coeffs - expected_coeffs)[:6]) 
    
    assert max_deviation <= threshold, (
        f"Test failed: Maximum deviation from expected coefficients is {max_deviation:e}"
    )