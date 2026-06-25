import pytest
import numpy as np
import frgfp as fp

def test_polyansatz_properties_and_validation():
    ansatz = fp.PolyAnsatz(inv_dim=2, order=(2, 3), center=(0.0, 0.0))
    assert ansatz.inv_dim == 2
    assert ansatz.num_coeffs == 12 

    with pytest.raises(ValueError):
        fp.PolyAnsatz(inv_dim=1, order=-1, center=0.0)
    with pytest.raises(ValueError):
        fp.PolyAnsatz(inv_dim=2, order=(2, 2), center=(0.0,))

def test_polyansatz_evaluation():
    # 1D Taylor sor x=1 körül
    ansatz = fp.PolyAnsatz(inv_dim=1, order=2, center=1.0)
    grid = np.array([[0.0], [1.0], [2.0], [3.0]])
    M = ansatz.evaluate_basis(grid)
    
    # x=1 körül a bázis: [1, (x-1), (x-1)^2]
    expected_M = np.array([
        [1.0, -1.0, 1.0],
        [1.0,  0.0, 0.0],
        [1.0,  1.0, 1.0],
        [1.0,  2.0, 4.0]
    ])
    np.testing.assert_allclose(M, expected_M)

def test_chebyshevansatz_evaluation():
    ansatz = fp.ChebyshevAnsatz(inv_dim=1, order=2, interval=(-1.0, 1.0))
    grid = np.array([[-1.0], [0.0], [0.5], [1.0]])
    M = ansatz.evaluate_basis(grid)
    
    expected_M = np.zeros((4, 3))
    expected_M[:, 0] = 1.0
    expected_M[:, 1] = grid[:, 0]
    expected_M[:, 2] = 2.0 * (grid[:, 0]**2) - 1.0
    
    np.testing.assert_allclose(M, expected_M, atol=1e-14)

def test_ansatz_grad_hess_shapes():
    ansatz = fp.PolyAnsatz(inv_dim=2, order=(2, 2), center=(0.0, 0.0))
    grid = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    grads = ansatz.evaluate_basis_grad(grid)
    hess = ansatz.evaluate_basis_hess(grid)
    
    assert len(grads) == 2 # 2 invariant, 2 gradient vector
    assert len(hess) == 4 # 2 invariánt, 2x2 Hess-matrix
    assert grads[0].shape == (2, 9) # (N_points, N_coeffs)
    assert hess[1].shape == (2, 9)