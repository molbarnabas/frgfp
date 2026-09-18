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
# 1. Dummy Flow Equations (O(N) Gaussian Limit)
# =====================================================================

def O_N_gaussian_flow_1d(I, V, dV, ddV, params):
    """
    1D Gaussian limit of the O(N) Wilson-Fisher flow.
    By explicitly ignoring dV and ddV, the flow becomes purely potential-driven.
    The exact fixed point is a constant: V = (omega_d * N) / d^2.
    """
    d = params[0]
    N = params[1]
    
    omega_d = 2.0 / ((4.0 * math.pi)**(d / 2.0) * math.gamma(d / 2.0))
    
    # RHS with dV=0 and ddV=0
    rhs = (omega_d / d) * N - d * V
    return rhs

def O_N_gaussian_flow_2d(I, V, dV, ddV, params):
    """
    2D Gaussian limit of the O(N) flow.
    Works identically to 1D since the Gaussian fixed point is a constant potential.
    """
    d = params[0]
    N = params[1]
    
    omega_d = 2.0 / ((4.0 * math.pi)**(d / 2.0) * math.gamma(d / 2.0))
    
    rhs = (omega_d / d) * N - d * V
    return rhs

def expected_gaussian_V(d, N):
    """Calculates the exact constant potential for the Gaussian fixed point."""
    omega_d = 2.0 / ((4.0 * math.pi)**(d / 2.0) * math.gamma(d / 2.0))
    return (omega_d * N) / (d**2)


# =====================================================================
# 2. Fixtures for Common Test Objects
# =====================================================================

@pytest.fixture
def setup_1d():
    """Initializes a 1D PolyAnsatz and collocation grid."""
    inv_dim = 1
    # A smaller order is sufficient to find a constant potential
    ansatz = fp.PolyAnsatz(inv_dim=inv_dim, order=2, center=0.0)
    grid = np.linspace(0.0, 0.4, 5, dtype=np.float64) 
    return ansatz, grid

@pytest.fixture
def setup_2d():
    """Initializes a 2D Chebyshev ansatz and a sufficient collocation grid."""
    inv_dim = 2
    ansatz = fp.ChebyshevAnsatz(inv_dim=inv_dim, order=(2, 2), interval=((0.0, 0.4), (0.0, 0.4)))
    
    i0 = np.linspace(0.0, 0.4, 4, dtype=np.float64)
    i1 = np.linspace(0.0, 0.4, 4, dtype=np.float64)
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
        flowrhs_func=O_N_gaussian_flow_1d, 
        inv_dim=1, 
        param_dim=2
    )
    assert solver is not None
    assert solver.inv_dim == 1
    assert hasattr(solver, '_cfunc_address')


# =====================================================================
# 4. Local Optimization Tests
# =====================================================================

def test_local_optimize_1d(setup_1d):
    """1D Gaussian fixed-point search."""
    ansatz, grid = setup_1d
    solver = LPACollSolver(grid, ansatz, O_N_gaussian_flow_1d, inv_dim=1, param_dim=2)
    
    init_coeffs = np.zeros(ansatz.num_coeffs)
    flow_params = np.array([3.0, 8.0]) 
    
    result = solver.local_optimize(init_coeffs, flow_params=flow_params, multithread=False) 
    
    assert result is not None
    assert hasattr(result, 'evaluate') 
    
    V_eval = result.evaluate(grid)
    V_expected = np.full_like(grid, expected_gaussian_V(3.0, 8.0))
    np.testing.assert_allclose(V_eval, V_expected, atol=1e-5)

def test_local_optimize_2d_multithread(setup_2d):
    """2D Gaussian fixed-point search testing multi-dimensional mapping."""
    ansatz, grid = setup_2d
    solver = LPACollSolver(grid, ansatz, O_N_gaussian_flow_2d, inv_dim=2, param_dim=2)
    
    init_coeffs = np.zeros(ansatz.num_coeffs)
    flow_params = np.array([3.0, 8.0]) 
    
    result = solver.local_optimize(init_coeffs, flow_params=flow_params, multithread=True) 
    
    assert result is not None
    assert hasattr(result, 'evaluate')
    
    V_eval = result.evaluate(grid)
    V_expected = np.full(grid.shape[0], expected_gaussian_V(3.0, 8.0))
    
    np.testing.assert_allclose(V_eval, V_expected, atol=1e-5)


# =====================================================================
# 5. Parameter Path Following Tests
# =====================================================================

def test_param_path_following(setup_1d):
    """Tests parameter following by changing the value of N."""
    ansatz, grid = setup_1d
    solver = LPACollSolver(grid, ansatz, O_N_gaussian_flow_1d, inv_dim=1, param_dim=2)
    
    init_coeffs = np.zeros(ansatz.num_coeffs)
    
    flow_params_path = [np.array([3.0, 4.0]), np.array([3.0, 8.0]), np.array([3.0, 12.0])]
    
    results = solver.param_path_following(init_coeffs, flow_params_path, multithread=True) 
    
    assert len(results) == 3
    for i, res in enumerate(results):
        assert res is not None
        assert hasattr(res, 'evaluate')
        
        current_N = flow_params_path[i][1]
        V_eval = res.evaluate(grid)
        V_expected = np.full_like(grid, expected_gaussian_V(3.0, current_N))
        np.testing.assert_allclose(V_eval, V_expected, atol=1e-5)


# =====================================================================
# 6. Multistart Optimization Tests
# =====================================================================

def test_multistart_optimize(setup_1d):
    """Tests optimization starting from multiple random initial guesses."""
    ansatz, grid = setup_1d
    solver = LPACollSolver(grid, ansatz, O_N_gaussian_flow_1d, inv_dim=1, param_dim=2)
    
    init_coeffs_list = [
        np.zeros(ansatz.num_coeffs),
        np.ones(ansatz.num_coeffs) * 0.1,
        np.random.rand(ansatz.num_coeffs) * 0.05
    ]
    
    flow_params = np.array([3.0, 8.0])
    
    results = solver.multistart_optimize(init_coeffs_list, flow_params=flow_params) 
    
    assert len(results) == 3
    for res in results:
        assert res is not None
        assert hasattr(res, 'evaluate')
        
        V_eval = res.evaluate(grid)
        V_expected = np.full_like(grid, expected_gaussian_V(3.0, 8.0))
        np.testing.assert_allclose(V_eval, V_expected, atol=1e-5)


# =====================================================================
# 7. Divergence / Failure Handling Tests
# =====================================================================

def test_divergence_handling(setup_1d):
    """Ensures that non-converging flows safely return None instead of crashing."""
    ansatz, grid = setup_1d
    
    def non_converging_flow(I, V, dV, ddV, params):
        """A simple flow that can never be zero for a finite potential."""
        return V**2 + 10.0 
        
    solver = LPACollSolver(grid, ansatz, non_converging_flow, inv_dim=1, param_dim=2)
    init_coeffs = np.zeros(ansatz.num_coeffs)
    
    result = solver.local_optimize(init_coeffs, flow_params=np.array([3.0, 8.0]), maxiter=5)
    
    assert result is None