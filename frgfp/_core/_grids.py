import numpy as np
from typing import Union, Tuple

def _normalize_grid_inputs(inv_dim: int, num_points: Union[int, Tuple[int, ...]], interval: Union[Tuple[float, float], Tuple[Tuple[float, float], ...]]):
    """
    Internal helper to rigorously validate and normalize grid parameters.
    """
    if not isinstance(inv_dim, int) or isinstance(inv_dim, bool):
        raise TypeError(f"The 'inv_dim' parameter must be an integer. Received type: {type(inv_dim).__name__}")
    if inv_dim < 1:
        raise ValueError(f"The 'inv_dim' must be a strictly positive integer. Received: {inv_dim}")

    if inv_dim == 1:
        num_points = (num_points,) if isinstance(num_points, (int, np.integer)) else num_points
        if isinstance(interval, tuple) and len(interval) == 2 and isinstance(interval[0], (int, float, np.number)):
            interval = (interval,)

    if not isinstance(num_points, tuple):
        raise TypeError(f"The 'num_points' parameter must be an int (for 1D) or a tuple of ints. Received: {type(num_points).__name__}")
    if len(num_points) != inv_dim:
        raise ValueError(f"The length of the 'num_points' tuple ({len(num_points)}) must match 'inv_dim' ({inv_dim}).")
    for i, n in enumerate(num_points):
        if not isinstance(n, (int, np.integer)) or isinstance(n, bool):
            raise TypeError(f"All elements in 'num_points' must be integers. Found type {type(n).__name__} at index {i}.")
        if n < 2:
            raise ValueError(f"The number of points in any dimension must be at least 2. Found {n} at index {i}.")

    if not isinstance(interval, tuple):
        raise TypeError(f"The 'interval' parameter must be a tuple. Received: {type(interval).__name__}")
    if len(interval) != inv_dim:
        raise ValueError(f"The length of the 'interval' tuple ({len(interval)}) must match 'inv_dim' ({inv_dim}).")
    
    for i, bounds in enumerate(interval):
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise ValueError(f"Each dimension in 'interval' must be a tuple of length 2 (min, max). Error at dimension {i}: {bounds}")
        if not all(isinstance(x, (int, float, np.number)) and not isinstance(x, bool) for x in bounds):
            raise TypeError(f"Interval bounds must be numeric (int or float). Error at dimension {i}: {bounds}")
        if bounds[0] >= bounds[1]:
            raise ValueError(f"In 'interval', minimum values must be strictly less than maximum values. Error at dimension {i}: {bounds[0]} >= {bounds[1]}")

    pts_arr = np.array(num_points, dtype=np.int32)
    int_arr = np.array(interval, dtype=np.float64)

    return pts_arr, int_arr

def uniform_collgrid(inv_dim: int, num_points: Union[int, Tuple[int, ...]], interval: Union[Tuple[float, float], Tuple[Tuple[float, float], ...]]) -> np.ndarray:
    """
    Generates a multidimensional uniform collocation grid.

    Constructs a Cartesian product of linearly spaced 1D grids. The resulting 
    multidimensional grid is flattened into a 2D array where each row represents 
    the coordinates of a single collocation point across all invariants. The 
    points are strictly ordered lexicographically (row-major format) to align 
    with the backend's multi-index tensor evaluations.

    Parameters
    ----------
    inv_dim : int
        The total number of physical invariants (dimensions) in the system.
    num_points : int or tuple of int
        The number of uniform divisions along each dimension.
    interval : tuple of float or tuple of tuples of float
        The `(min, max)` boundaries for the grid. For multi-dimensional systems,
        this must be a nested tuple, e.g., `((0.0, 1.0), (-1.0, 1.0))`.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape `(N_total, inv_dim)`.
    """
    pts, ints = _normalize_grid_inputs(inv_dim, num_points, interval)
    
    grids_1d = [np.linspace(ints[d, 0], ints[d, 1], pts[d]) for d in range(inv_dim)]
    mesh = np.meshgrid(*grids_1d, indexing='ij')
    
    return np.stack(mesh, axis=-1).reshape(-1, inv_dim)

def chebyshev_collgrid(inv_dim: int, num_points: Union[int, Tuple[int, ...]], interval: Union[Tuple[float, float], Tuple[Tuple[float, float], ...]]) -> np.ndarray:
    """
    Generates a multidimensional Chebyshev-Gauss-Lobatto collocation grid.

    Calculates the extrema of Chebyshev polynomials of the first kind, mapping 
    nodes from `[-1, 1]` to the user-specified intervals `[a, b]`. These nodes 
    cluster near the boundaries, minimizing Runge's phenomenon.

    Parameters
    ----------
    inv_dim : int
        The total number of physical invariants (dimensions).
    num_points : int or tuple of int
        The number of Chebyshev nodes along each dimension.
    interval : tuple of float or tuple of tuples of float
        The physical boundaries `(min, max)` for the expansion in each dimension.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape `(N_total, inv_dim)` containing the mapped nodes.
    """
    pts, ints = _normalize_grid_inputs(inv_dim, num_points, interval)
    
    grids_1d = []
    for d in range(inv_dim):
        n = pts[d]
        a, b = ints[d, 0], ints[d, 1]
        
        k = np.arange(n)
        nodes_standard = -np.cos(k * np.pi / (n - 1))
        nodes_scaled = 0.5 * (a + b) + 0.5 * (b - a) * nodes_standard
        grids_1d.append(nodes_scaled)

    mesh = np.meshgrid(*grids_1d, indexing='ij')
    return np.stack(mesh, axis=-1).reshape(-1, inv_dim)