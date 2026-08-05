import pytest
import numpy as np
import math
from pathlib import Path
import frgfp as fp

def ON_flowrhs(I, V, dV, ddV, params):
    d = params[0]
    N = params[1]
    
    omega_d = 2.0 / ((4.0 * math.pi)**(d / 2.0) * math.gamma(d / 2.0))
    
    rhs = (omega_d / d) * (
        1.0 / (1.0 + dV + 2.0 * I * ddV) + 
        (N - 1.0) / (1.0 + dV) 
    ) - d * V + (d - 2.0) * I * dV 
    
    return rhs

def high_perturb(rho):
    return -0.341419*rho + 0.876914*rho**2 + 1.45014*rho**3 + 1.5513*rho**4 - \
           0.167052*rho**5 - 1.55923*rho**6 + 10.488*rho**7 + 37.6868*rho**8 + 0.0683767

def run_builtin_multistart():
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
        
    init_coeffs_array = np.array(init_coeffs_list)
    
    results = solver.multistart_optimize(
        init_coeffs_list=init_coeffs_array, 
        flow_params=(3.0, 8.0), 
        maxiter=10000, 
        tol=1e-7
    )
    
    return results

def test_o8_multistart_benchmark(benchmark):
    current_dir = Path(__file__).parent
    data_file_path = current_dir / 'o8_fp.dat'
    expected_coeffs = np.loadtxt(data_file_path)
    
    multistart_results = benchmark(run_builtin_multistart)
    
    threshold = 1e-3
    
    valid_results = [FP for FP in multistart_results if FP is not None]
    
    assert len(valid_results) > 0, "Test failed: All 50 optimization attempts diverged (returned None)."
    
    for FP in valid_results:
        dev = np.max(np.abs(FP.coeffs - expected_coeffs)[:6])
        assert dev <= threshold, (
            f"Test failed: A converged result exceeded the threshold! "
            f"Deviation {dev:e} is greater than {threshold:e}."
        )