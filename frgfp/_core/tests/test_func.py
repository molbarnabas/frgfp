import pytest
import numpy as np
import frgfp as fp

def test_func_properties_and_setters():
    ansatz = fp.PolyAnsatz(inv_dim=1, order=2, center=0.0)
    func = fp.FuncFromAnsatz(ansatz, [1.0, 2.0, 3.0])
    
    assert func.num_coeffs == 3
    np.testing.assert_array_equal(func.coeffs, [1.0, 2.0, 3.0])
    
    func.coeffs = np.array([4.0, 5.0, 6.0])
    np.testing.assert_array_equal(func.coeffs, [4.0, 5.0, 6.0])
    
    with pytest.raises(ValueError):
        func.coeffs = [1.0, 2.0]

def test_func_grad_hess_evaluation():
    ansatz = fp.PolyAnsatz(inv_dim=1, order=3, center=0.0)
    # Function: V(x) = 1 + 2x + 3x^2 + x^3
    coeffs = np.array([1.0, 2.0, 3.0, 1.0])
    func = fp.FuncFromAnsatz(ansatz, coeffs)
    
    grid = np.array([[0.0], [1.0], [2.0]])
    
    vals = func.evaluate(grid)
    grads = func.evaluate_grad(grid)
    hess = func.evaluate_hess(grid)
    
    # Analytic
    # V(x)   = 1 + 2x + 3x^2 + x^3
    # V'(x)  = 2 + 6x + 3x^2
    # V''(x) = 6 + 6x
    
    np.testing.assert_allclose(vals, [1.0, 7.0, 25.0])
    np.testing.assert_allclose(grads[0], [2.0, 11.0, 26.0])
    np.testing.assert_allclose(hess[0], [6.0, 12.0, 18.0])

def test_func_from_grid_chebyshev_fitting():
    ansatz = fp.ChebyshevAnsatz(inv_dim=1, order=4, interval=(-2.0, 2.0))
    grid = fp.chebyshev_collgrid(1, 10, (-2.0, 2.0))
    
    target_vals = 0.5 - 1.2 * grid[:, 0] + 0.3 * (grid[:, 0]**2) - 0.1 * (grid[:, 0]**3)
    
    fitted_func = fp.FuncFromAnsatz.from_grid(ansatz, grid, target_vals)
    
    reconstructed_vals = fitted_func.evaluate(grid)
    np.testing.assert_allclose(reconstructed_vals, target_vals, atol=1e-12)

def test_func_invalid_inputs():
    ansatz = fp.PolyAnsatz(inv_dim=2, order=(1, 1), center=(0.0, 0.0))
    with pytest.raises(ValueError):
        fp.FuncFromAnsatz(ansatz, [1.0, 2.0])
    
    grid = np.array([[1.0], [2.0]]) 
    vals = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        fp.FuncFromAnsatz.from_grid(ansatz, grid, vals)