import pytest
import numpy as np
import frgfp as fp

def test_uniform_collgrid_1d():
    grid = fp.uniform_collgrid(inv_dim=1, num_points=5, interval=(0.0, 1.0))
    assert grid.shape == (5, 1)
    np.testing.assert_allclose(grid.flatten(), [0.0, 0.25, 0.5, 0.75, 1.0])

def test_uniform_collgrid_multidim():
    grid = fp.uniform_collgrid(inv_dim=2, num_points=(3, 4), interval=((-1.0, 1.0), (0.0, 3.0)))
    assert grid.shape == (12, 2)
    assert grid[0, 0] == -1.0 and grid[0, 1] == 0.0
    assert grid[-1, 0] == 1.0 and grid[-1, 1] == 3.0

def test_chebyshev_collgrid_1d():
    grid = fp.chebyshev_collgrid(inv_dim=1, num_points=3, interval=(-1.0, 1.0))
    assert grid.shape == (3, 1)
    np.testing.assert_allclose(grid.flatten(), [-1.0, 0.0, 1.0], atol=1e-15)

def test_chebyshev_collgrid_boundaries():
    grid = fp.chebyshev_collgrid(inv_dim=2, num_points=(10, 15), interval=((-5.0, 5.0), (2.0, 8.0)))
    assert grid.shape == (150, 2)
    assert np.isclose(np.min(grid[:, 0]), -5.0)
    assert np.isclose(np.max(grid[:, 0]), 5.0)
    assert np.isclose(np.min(grid[:, 1]), 2.0)
    assert np.isclose(np.max(grid[:, 1]), 8.0)

def test_grid_input_validation():
    with pytest.raises(TypeError):
        fp.uniform_collgrid(inv_dim=1.5, num_points=5, interval=(0, 1))
    with pytest.raises(ValueError):
        fp.uniform_collgrid(inv_dim=2, num_points=(5,), interval=((-1, 1), (-1, 1)))
    with pytest.raises(ValueError):
        fp.uniform_collgrid(inv_dim=1, num_points=5, interval=(1.0, 0.0)) # min > max
    with pytest.raises(ValueError):
        fp.chebyshev_collgrid(inv_dim=1, num_points=1, interval=(0.0, 1.0)) # min 2 points