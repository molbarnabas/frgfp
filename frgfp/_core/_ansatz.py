"""
Polynomial and Chebyshev ansatz definitions for the frgfp core module.

This module exposes the user-facing basis-expansion classes. Their inputs are
validated and normalized in Python, while the actual basis, gradient, and
Hessian evaluations are delegated to the optimised ``_core_cpp`` Eigen backend.
"""

import numpy as np
from typing import Union, Tuple, List
from . import _core_cpp

def _validate_common_ansatz_inputs(inv_dim: int, order: Union[int, Tuple[int, ...]]):
    """
    Validates inv_dim and order, returning a normalized order tuple.
    """
    if not isinstance(inv_dim, int) or isinstance(inv_dim, bool):
        raise TypeError(f"The 'inv_dim' parameter must be an integer. Received: {type(inv_dim).__name__}")
    if inv_dim < 1:
        raise ValueError(f"The 'inv_dim' must be a strictly positive integer. Received: {inv_dim}")

    if inv_dim == 1 and isinstance(order, (int, np.integer)):
        order = (order,)

    if not isinstance(order, tuple):
        raise TypeError(f"The 'order' parameter must be an int (for 1D) or a tuple of ints. Received: {type(order).__name__}")
    if len(order) != inv_dim:
        raise ValueError(f"The length of the 'order' tuple ({len(order)}) must match 'inv_dim' ({inv_dim}).")
    for i, ord_val in enumerate(order):
        if not isinstance(ord_val, (int, np.integer)) or isinstance(ord_val, bool):
            raise TypeError(f"All elements in 'order' must be integers. Found type {type(ord_val).__name__} at index {i}.")
        if ord_val < 0:
            raise ValueError(f"Polynomial orders cannot be negative. Found {ord_val} at index {i}.")

    return order

class ChebyshevAnsatz(_core_cpp.ChebyshevAnsatz):
    """
    Multidimensional Chebyshev polynomial ansatz defining the basis expansion.

    Constructs a highly optimized, orthogonal Chebyshev basis relying on an 
    Eigen C++ backend. The implementation maps the physical intervals to `[-1, 1]` 
    and evaluates the basis utilizing a stable recursive relation internally.

    Parameters
    ----------
    inv_dim : int
        The number of invariants (dimensions).
    order : int or tuple of int
        The maximum polynomial order for each invariant.
    interval : tuple of float or tuple of tuples of float
        The physical interval for expansion, which will be linearly mapped to the 
        standard `[-1, 1]` domain internally.
    """
    def __init__(self, inv_dim: int, order: Union[int, Tuple[int, ...]], interval: Union[Tuple[float, float], Tuple[Tuple[float, float], ...]]):
        """
        Initialize a Chebyshev basis expansion.

        Parameters
        ----------
        inv_dim : int
            The number of invariants (dimensions).
        order : int or tuple of int
            The maximum polynomial order for each invariant.
        interval : tuple of float or tuple of tuples of float
            The physical expansion interval per dimension, mapped internally to
            the standard ``[-1, 1]`` domain.
        """
        order = _validate_common_ansatz_inputs(inv_dim, order)

        if inv_dim == 1 and isinstance(interval, tuple) and len(interval) == 2 and isinstance(interval[0], (int, float, np.number)):
            interval = (interval,)

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

        order_arr = np.array(order, dtype=np.int32)
        interval_arr = np.array(interval, dtype=np.float64)

        super().__init__(inv_dim, order_arr, interval_arr)


    @property
    def inv_dim(self) -> int:
        """The dimensionality of the invariant space."""
        return super().inv_dim

    @property
    def num_coeffs(self) -> int:
        """The total number of coefficients in the tensor-product basis."""
        return super().num_coeffs

    def evaluate_basis(self, grid: np.ndarray) -> np.ndarray:
        """
        Evaluates the multidimensional basis functions on the given grid.

        Parameters
        ----------
        grid : np.ndarray
            A 2D array of shape `(N_points, inv_dim)` containing the grid coordinates.

        Returns
        -------
        np.ndarray
            A 2D array of shape `(N_points, N_coeffs)` containing the basis function values.
        """
        return super().evaluate_basis(grid)

    def evaluate_basis_grad(self, grid: np.ndarray) -> List[np.ndarray]:
        """
        Evaluates the gradient of the basis functions on the given grid.

        Parameters
        ----------
        grid : np.ndarray
            A 2D array of shape `(N_points, inv_dim)` containing the grid coordinates.

        Returns
        -------
        list of np.ndarray
            A list of length `inv_dim`, where each element is a 2D array of shape 
            `(N_points, N_coeffs)` representing the basis gradient components.
        """
        return super().evaluate_basis_grad(grid)

    def evaluate_basis_hess(self, grid: np.ndarray) -> List[np.ndarray]:
        """
        Evaluates the Hessian of the basis functions on the given grid.

        Parameters
        ----------
        grid : np.ndarray
            A 2D array of shape `(N_points, inv_dim)` containing the grid coordinates.

        Returns
        -------
        list of np.ndarray
            A list of length `inv_dim ** 2` (row-major order), where each element is 
            a 2D array of shape `(N_points, N_coeffs)` representing the basis Hessian components.
        """
        return super().evaluate_basis_hess(grid)


class PolyAnsatz(_core_cpp.PolyAnsatz):
    """
    Standard multidimensional polynomial ansatz defining a local Taylor-like basis.

    Constructs a polynomial expansion evaluated around a designated central 
    point (the expansion minimum). The basis functions follow the form 
    $(x - x_0)^n$. Evaluations are strictly processed in the Eigen C++ backend.

    Parameters
    ----------
    inv_dim : int
        The number of invariants (dimensions).
    order : int or tuple of int
        The maximum polynomial order for each invariant.
    center : float or tuple of float
        The local expansion point $x_0$ for each invariant. 
    """
    def __init__(self, inv_dim: int, order: Union[int, Tuple[int, ...]], center: Union[float, Tuple[float, ...]]):
        """
        Initialize a polynomial (Taylor) basis expansion.

        Parameters
        ----------
        inv_dim : int
            The number of invariants (dimensions).
        order : int or tuple of int
            The maximum polynomial order for each invariant.
        center : float or tuple of float
            The local expansion point ``x_0`` for each invariant.
        """
        order = _validate_common_ansatz_inputs(inv_dim, order)

        if inv_dim == 1 and isinstance(center, (int, float, np.number)) and not isinstance(center, bool):
            center = (center,)

        if not isinstance(center, tuple):
            raise TypeError(f"The 'center' parameter must be a float (for 1D) or a tuple of floats. Received: {type(center).__name__}")
        if len(center) != inv_dim:
            raise ValueError(f"The length of the 'center' tuple ({len(center)}) must match 'inv_dim' ({inv_dim}).")
        for i, c_val in enumerate(center):
            if not isinstance(c_val, (int, float, np.number)) or isinstance(c_val, bool):
                raise TypeError(f"All elements in 'center' must be numeric. Found type {type(c_val).__name__} at index {i}.")

        order_arr = np.array(order, dtype=np.int32)
        center_arr = np.array(center, dtype=np.float64)

        super().__init__(inv_dim, order_arr, center_arr)


    @property
    def inv_dim(self) -> int:
        """The dimensionality of the invariant space."""
        return super().inv_dim

    @property
    def num_coeffs(self) -> int:
        """The total number of coefficients in the tensor-product basis."""
        return super().num_coeffs


    def evaluate_basis(self, grid: np.ndarray) -> np.ndarray:
        """
        Evaluates the multidimensional basis functions on the given grid.

        Parameters
        ----------
        grid : np.ndarray
            A 2D array of shape `(N_points, inv_dim)` containing the grid coordinates.

        Returns
        -------
        np.ndarray
            A 2D array of shape `(N_points, N_coeffs)` containing the basis function values.
        """
        return super().evaluate_basis(grid)

    def evaluate_basis_grad(self, grid: np.ndarray) -> List[np.ndarray]:
        """
        Evaluates the gradient of the basis functions on the given grid.

        Parameters
        ----------
        grid : np.ndarray
            A 2D array of shape `(N_points, inv_dim)` containing the grid coordinates.

        Returns
        -------
        list of np.ndarray
            A list of length `inv_dim`, where each element is a 2D array of shape 
            `(N_points, N_coeffs)` representing the basis gradient components.
        """
        return super().evaluate_basis_grad(grid)

    def evaluate_basis_hess(self, grid: np.ndarray) -> List[np.ndarray]:
        """
        Evaluates the Hessian of the basis functions on the given grid.

        Parameters
        ----------
        grid : np.ndarray
            A 2D array of shape `(N_points, inv_dim)` containing the grid coordinates.

        Returns
        -------
        list of np.ndarray
            A list of length `inv_dim ** 2` (row-major order), where each element is 
            a 2D array of shape `(N_points, N_coeffs)` representing the basis Hessian components.
        """
        return super().evaluate_basis_hess(grid)