"""
Function expansion built from an ansatz and a coefficient vector.

This module exposes ``FuncFromAnsatz``, which represents a function expanded in
the basis of an ansatz. The expansion coefficients are stored explicitly, and
the expanded function together with its gradient and Hessian is evaluated on
collocation grids via the ``_core_cpp`` Eigen backend.
"""

import numpy as np
from typing import Union, List
from . import _core_cpp

class FuncFromAnsatz(_core_cpp.FuncFromAnsatz):
    """
    A function expanded in the basis of an ansatz.

    This class binds an instantiated Ansatz (which defines the multidimensional
    basis functions) with a specific vector of expansion coefficients, so that
    the expanded function, its gradient and its Hessian can be evaluated on
    collocation grids using fast matrix-vector products in the Eigen C++ backend.

    Parameters
    ----------
    ansatz : frgfp.PolyAnsatz or frgfp.ChebyshevAnsatz
        The basis expansion object defining the function space.
    coeffs : list of float or np.ndarray
        A 1D array of coefficients. Its length must exactly match the total 
        number of basis functions defined in the ansatz.
    """
    def __init__(self, ansatz: _core_cpp.Ansatz, coeffs: Union[List[float], np.ndarray]):
        """
        Initialize the function expansion.

        Parameters
        ----------
        ansatz : _core_cpp.Ansatz
            The instantiated basis expansion defining the function space.
        coeffs : list of float or np.ndarray
            A 1D array of coefficients whose length matches ``ansatz.num_coeffs``.

        Raises
        ------
        TypeError
            If ``ansatz`` is not a valid Ansatz instance.
        ValueError
            If the coefficient count does not match ``ansatz.num_coeffs``.
        """
        if not isinstance(ansatz, _core_cpp.Ansatz):
            raise TypeError("The 'ansatz' parameter must be a valid instantiated Ansatz object.")
        
        coeffs_arr = np.array(coeffs, dtype=np.float64).flatten()
        if len(coeffs_arr) != ansatz.num_coeffs:
            raise ValueError(f"Coefficient array length ({len(coeffs_arr)}) must exactly match the ansatz's num_coeffs ({ansatz.num_coeffs}).")
        
        super().__init__(ansatz, coeffs_arr)

    @classmethod
    def from_grid(cls, ansatz: _core_cpp.Ansatz, grid: np.ndarray, vals: np.ndarray):
        """
        Creates a FuncFromAnsatz object by fitting coefficients to target values on a grid.

        This method utilizes Eigen's Singular Value Decomposition (SVD) backend
        to compute the Moore-Penrose pseudo-inverse. Because the basis expansion 
        is linear with respect to its coefficients, this direct pseudo-inverse 
        solution analytically finds the exact global minimum (least-squares fit) 
        in a single deterministic step, rendering initial guesses unnecessary.

        Parameters
        ----------
        ansatz : frgfp.Ansatz
            The basis function expansion object.
        grid : np.ndarray
            A 2D NumPy array of shape `(N_points, inv_dim)` containing the coordinates.
        vals : np.ndarray
            A 1D NumPy array of shape `(N_points,)` containing the target values.

        Returns
        -------
        frgfp.FuncFromAnsatz
            A new instance initialized with the optimally fitted coefficients.
        """
        if not isinstance(ansatz, _core_cpp.Ansatz):
            raise TypeError("The 'ansatz' parameter must be a valid instantiated Ansatz object.")
        
        grid_arr = np.array(grid, dtype=np.float64)
        vals_arr = np.array(vals, dtype=np.float64).flatten()
        
        if grid_arr.ndim != 2 or grid_arr.shape[1] != ansatz.inv_dim:
            raise ValueError(f"The 'grid' must be a 2D array with shape (N_points, {ansatz.inv_dim}).")
        if grid_arr.shape[0] != vals_arr.shape[0]:
            raise ValueError(f"The number of grid points ({grid_arr.shape[0]}) must match the number of target values ({vals_arr.shape[0]}).")
        
        fitted_coeffs = _core_cpp.fit_coeffs_from_grid(ansatz, grid_arr, vals_arr)
        return cls(ansatz, fitted_coeffs)

    @property
    def ansatz(self) -> _core_cpp.Ansatz:
        """The associated basis expansion object."""
        return super().ansatz

    @property
    def num_coeffs(self) -> int:
        """The total number of coefficients in the expansion."""
        return super().num_coeffs

    @property
    def coeffs(self) -> np.ndarray:
        """
        The current expansion coefficients. 
        Re-assigning this attribute updates the expanded function immediately.
        """
        return super().coeffs

    @coeffs.setter
    def coeffs(self, new_coeffs: Union[List[float], np.ndarray]):
        """
        Update the expansion coefficients in place.

        Parameters
        ----------
        new_coeffs : list of float or np.ndarray
            The new coefficients; their length must match ``num_coeffs``.
        """
        _core_cpp.FuncFromAnsatz.coeffs.__set__(self, new_coeffs)

    def evaluate(self, grid: np.ndarray) -> np.ndarray:
        """
        Evaluates the expanded function on the given collocation grid.

        Parameters
        ----------
        grid : np.ndarray
            A 2D array of shape `(N_points, inv_dim)` containing the grid coordinates.

        Returns
        -------
        np.ndarray
            A 1D array of shape `(N_points,)` with the evaluated function values.
        """
        return super().evaluate(grid)

    def evaluate_grad(self, grid: np.ndarray) -> List[np.ndarray]:
        """
        Evaluates the gradient of the expanded function on the given grid.

        Parameters
        ----------
        grid : np.ndarray
            A 2D array of shape `(N_points, inv_dim)` containing the grid coordinates.

        Returns
        -------
        list of np.ndarray
            A list of length `inv_dim`, where each element is a 1D array of shape 
            `(N_points,)` representing the evaluated gradient components.
        """
        return super().evaluate_grad(grid)

    def evaluate_hess(self, grid: np.ndarray) -> List[np.ndarray]:
        """
        Evaluates the Hessian of the expanded function on the given grid.

        The Hessian components are returned as a flattened, row-major list. 
        For an invariant dimension of `N`, the list will contain `N^2` elements.

        Parameters
        ----------
        grid : np.ndarray
            A 2D array of shape `(N_points, inv_dim)` containing the grid coordinates.

        Returns
        -------
        list of np.ndarray
            A list of length `inv_dim ** 2`, where each element is a 1D array of 
            shape `(N_points,)` representing the evaluated Hessian components.
        """
        return super().evaluate_hess(grid)

    def evaluate_rhs(self, grid, inv_dim, flowrhs_func, flow_params: Union[List[float], np.ndarray, tuple]) -> np.ndarray:
        """
        Evaluates the right-hand side of the flow equation on the given grid.

        The expanded function and its derivatives are evaluated on the grid and
        fed, together with the flow parameters, into the user-supplied flow
        equation.

        Parameters
        ----------
        grid : np.ndarray
            The collocation grid: either a 2D array of shape `(N_points, inv_dim)`
            or a 1D array of shape `(N_points,)` when `inv_dim` is 1.
        inv_dim : int
            The invariant dimension of the problem.
        flowrhs_func : callable
            A user-defined function that computes the right-hand side of the flow 
            equation. It should accept arguments `(I, V, dV, ddV, params)`.
        flow_params : list or np.ndarray or tuple
            A 1D array of parameters required by the `flowrhs_func`.

        Returns
        -------
        np.ndarray
            A 1D array of shape `(N_points,)` with the evaluated right-hand side values.
        """
        if inv_dim == 1:
            invariants = np.asarray(grid, dtype=np.float64).reshape(-1)
            return flowrhs_func(
                invariants, self.evaluate(grid), self.evaluate_grad(grid)[0],
                self.evaluate_hess(grid)[0], flow_params)
        return flowrhs_func(
            grid, self.evaluate(grid), self.evaluate_grad(grid),
            self.evaluate_hess(grid), flow_params)