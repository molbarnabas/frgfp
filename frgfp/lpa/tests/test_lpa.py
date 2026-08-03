"""
Unit tests for the frgfp.lpa module.
Run from the project root using: pytest tests/test_lpa.py -v
"""

import math
import pytest
import numpy as np
import frgfp as fp
from frgfp.lpa import LPACollSolver

# =====================================================================
# 1. Dummy Flow Equations for Guaranteed Convergence Testing
# =====================================================================

def dummy_1d_flow(I, V, dV, ddV, params):
    """
    1D test equation
    """
    d = params[0]
    N = params[1]
    
    omega_d = 2.0 / ((4.0 * math.pi)**(d / 2.0) * math.gamma(d / 2.0))
    
    rhs = (omega_d / d) * (
        1.0 / (1.0 + dV+2.0 * I * ddV) + 
        (N - 1.0) / (1.0 + dV) 
    ) - d * V + (d - 2.0) * I * dV 
    
    return rhs

def dummy_2d_flow(I, V, dV, ddV, params):
    """
    2D dummy flow
    """
    d = params[0]
    N = params[1]
    
    omega_d = 2.0 / ((4.0 * math.pi)**(d / 2.0) * math.gamma(d / 2.0))
    
    rhs = (omega_d / d) * (
        1.0 / (1.0 + dV[0]+2.0 * I * ddV[0,0]) + 
        (N - 1.0) / (1.0 + dV[0]) 
    ) - d * V + (d - 2.0) * I * dV[0] 
    
    return rhs


# =====================================================================
# 2. Fixtures for Common Test Objects
# =====================================================================

@pytest.fixture
def setup_1d():
    """Initializes a 1D Chebyshev ansatz and collocation grid."""
    inv_dim = 1
    ansatz = fp.ChebyshevAnsatz(inv_dim=inv_dim, order=2, interval=(0.0, 1.0))
    grid = np.linspace(0.0, 1.0, 5, dtype=np.float64) # 5 points on [0, 1]
    return ansatz, grid

@pytest.fixture
def setup_2d():
    """Initializes a 2D Chebyshev ansatz and a sufficient collocation grid."""
    inv_dim = 2
    ansatz = fp.ChebyshevAnsatz(inv_dim=inv_dim, order=(2, 2), interval=((0.0, 1.0), (0.0, 1.0)))
    
    i0 = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    i1 = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    I0, I1 = np.meshgrid(i0, i1)
    
    grid = np.vstack([I0.ravel(), I1.ravel()]).T 
    
    return ansatz, grid


# =====================================================================
# 3. Initialization and Type Tests
# =====================================================================

def test_solver_init_success(setup_1d):
    """Tests successful initialization and Numba JIT compilation."""
    ansatz, grid = setup_1d
    solver = LPACollSolver(
        coll_grid=grid, 
        ansatz=ansatz, 
        flowrhs_func=dummy_1d_flow, 
        inv_dim=1, 
        param_dim=1
    )
    assert solver is not None
    assert solver.inv_dim == 1
    assert hasattr(solver, '_cfunc_address')

def test_solver_init_type_errors(setup_1d):
    """Tests error handling for invalid initialization parameters."""
    ansatz, _ = setup_1d
    with pytest.raises((TypeError, ValueError)):
        # Passing a string instead of a numpy array for the grid
        LPACollSolver(
            coll_grid="invalid_grid", 
            ansatz=ansatz, 
            flowrhs_func=dummy_1d_flow, 
            inv_dim=1, 
            param_dim=1
        )


# =====================================================================
# 4. Local Optimization Tests (Newton-Gauss Core)
# =====================================================================

def test_local_optimize_1d(setup_1d):
    """1D fixed-point search (Zero-overhead Numba bridge test)."""
    ansatz, grid = setup_1d
    solver = LPACollSolver(grid, ansatz, dummy_1d_flow, inv_dim=1, param_dim=1)
    
    init_coeffs = np.full(ansatz.num_coeffs, 0.05)
    flow_params = np.array([2.5]) # p = 2.5
    
    # Run WITHOUT multithreading
    result = solver.local_optimize(init_coeffs, flow_params=flow_params, multithread=False)
    
    assert result is not None
    # API REQUIREMENT: Must return a FuncFromAnsatz object (checking evaluate capability)
    assert hasattr(result, 'evaluate') 
    
    # Verify the analytical solution: V(I) = I**2 + 2.5
    V_eval = result.evaluate(grid)
    V_expected = grid**2 + 2.5
    np.testing.assert_allclose(V_eval, V_expected, atol=1e-5)

def test_local_optimize_2d_multithread(setup_2d):
    """
    2D fixed-point search testing interleaved memory mapping, multidimensional 
    Numba arrays (dV[i], ddV[i, j]) and OpenMP Thread-Local Storage.
    """
    ansatz, grid = setup_2d
    solver = LPACollSolver(grid, ansatz, dummy_2d_flow, inv_dim=2, param_dim=1)
    
    init_coeffs = np.full(ansatz.num_coeffs, 0.05)
    flow_params = np.array([1.0]) # p = 1.0
    
    # Run WITH multithreading
    result = solver.local_optimize(init_coeffs, flow_params=flow_params, multithread=True)
    
    assert result is not None
    assert hasattr(result, 'evaluate')
    
    # Verify the exact analytical solution: V(I0, I1) = I0**2 + I1**2 + I0*I1 + 1.0
    V_eval = result.evaluate(grid)
    V_expected = grid[:, 0]**2 + grid[:, 1]**2 + (grid[:, 0] * grid[:, 1]) + 1.0
    
    # If the C++ std::vector flattening logic was even slightly wrong, 
    # the dV or ddV values would be garbage and this assertion would fail!
    np.testing.assert_allclose(V_eval, V_expected, atol=1e-5)


# =====================================================================
# 5. Parameter Path Following Tests
# =====================================================================

def test_param_path_following(setup_1d):
    """Tests the C++ loop and dynamic list conversion of parameter following."""
    ansatz, grid = setup_1d
    solver = LPACollSolver(grid, ansatz, dummy_1d_flow, inv_dim=1, param_dim=1)
    
    init_coeffs = np.full(ansatz.num_coeffs, 0.05)
    # Parameter trajectory: 1.0 -> 2.0 -> 3.0
    flow_params_path = [np.array([1.0]), np.array([2.0]), np.array([3.0])]
    
    results = solver.param_path_following(init_coeffs, flow_params_path, multithread=True)
    
    assert len(results) == 3
    for i, res in enumerate(results):
        assert res is not None
        assert hasattr(res, 'evaluate')
        
        expected_param = flow_params_path[i][0]
        V_eval = res.evaluate(grid)
        V_expected = grid**2 + expected_param
        np.testing.assert_allclose(V_eval, V_expected, atol=1e-5)


# =====================================================================
# 6. Multistart Optimization Tests (OpenMP)
# =====================================================================

def test_multistart_optimize(setup_1d):
    """Tests the outer OpenMP loop and automatic disabling of nested threading."""
    ansatz, grid = setup_1d
    solver = LPACollSolver(grid, ansatz, dummy_1d_flow, inv_dim=1, param_dim=1)
    
    # Three completely random starting points
    init_coeffs_list = [
        np.full(ansatz.num_coeffs, 0.05),
        np.full(ansatz.num_coeffs, 0.05),
        np.full(ansatz.num_coeffs, 0.05)
    ]
    
    flow_params = np.array([4.2])
    
    results = solver.multistart_optimize(init_coeffs_list, flow_params=flow_params)
    
    assert len(results) == 3
    # Since the dummy flow has a single stable global minimum, 
    # all initial guesses should converge to the exact same potential.
    for res in results:
        assert res is not None
        assert hasattr(res, 'evaluate')
        
        V_eval = res.evaluate(grid)
        V_expected = grid**2 + 4.2
        np.testing.assert_allclose(V_eval, V_expected, atol=1e-5)


# =====================================================================
# 7. Divergence / Failure Handling Tests
# =====================================================================

def test_divergence_handling(setup_1d):
    """Ensures that non-converging flows safely return None instead of crashing."""
    ansatz, grid = setup_1d
    
    # A flow with no exact fixed-point solution so the solver must fail safely.
    def hard_flow(I, V, dV, ddV, params):
        return V + params[0] + 1.0
        
    solver = LPACollSolver(grid, ansatz, hard_flow, inv_dim=1, param_dim=1)
    init_coeffs = np.zeros(ansatz.num_coeffs)
    
    # Force failure with a ridiculously low maxiter (e.g., 2 iterations)
    result = solver.local_optimize(init_coeffs, flow_params=np.array([1.0]), maxiter=2)
    
    # API REQUIREMENT: Must return None on failure
    assert result is None
