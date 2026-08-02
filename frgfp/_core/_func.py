import numpy as np
from typing import Union, List
from . import _core_cpp

class FuncFromAnsatz(_core_cpp.FuncFromAnsatz):
    """
    Evaluates and manages the functional expansion of the effective potential.

    This class binds an instantiated Ansatz (which defines the multidimensional 
    basis functions) with a specific vector of expansion coefficients. It leverages 
    the Eigen C++ backend to evaluate the function, its gradient, and its Hessian 
    on collocation grids using ultra-fast matrix-vector multiplications.

    Parameters
    ----------
    ansatz : frgfp.PolyAnsatz or frgfp.ChebyshevAnsatz
        The basis function expansion object defining the functional space.
    coeffs : list of float or np.ndarray
        A 1D array of coefficients. Its length must exactly match the total 
        number of basis functions defined in the ansatz.
    """
    def __init__(self, ansatz: _core_cpp.Ansatz, coeffs: Union[List[float], np.ndarray]):
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
        Re-assigning this attribute updates the functional dynamically.
        """
        return super().coeffs

    @coeffs.setter
    def coeffs(self, new_coeffs: Union[List[float], np.ndarray]):
        _core_cpp.FuncFromAnsatz.coeffs.__set__(self, new_coeffs)

    def evaluate(self, grid: np.ndarray) -> np.ndarray:
        """
        Evaluates the functional expansion on the given collocation grid.

        Parameters
        ----------
        grid : np.ndarray
            A 2D array of shape `(N_points, inv_dim)` containing the grid coordinates.

        Returns
        -------
        np.ndarray
            A 1D array of shape `(N_points,)` with the evaluated functional values.
        """
        return super().evaluate(grid)

    def evaluate_grad(self, grid: np.ndarray) -> List[np.ndarray]:
        """
        Evaluates the gradient of the functional expansion on the given grid.

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
        Evaluates the Hessian of the functional expansion on the given grid.

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

    def evaluate_rhs(self, grid, inv_dim,flowrhs_func, flow_params: Union[List[float], np.ndarray, tuple]) -> np.ndarray:
        """
        Evaluates the right-hand side of the flow equation on the given grid.

        This method computes the flow equation's right-hand side using the current 
        functional expansion, its derivatives, and the provided flow parameters.

        Parameters
        ----------
        grid : np.ndarray
            A 2D array of shape `(N_points, inv_dim)` containing the grid coordinates.
        flow_params : list or np.ndarray or tuple
            A 1D array of parameters required by the flowrhs_func.
        flowrhs_func : callable
            A user-defined function that computes the right-hand side of the flow 
            equation. It should accept arguments `(I, V, dV, ddV, params)`.

        Returns
        -------
        np.ndarray
            A 1D array of shape `(N_points,)` with the evaluated right-hand side values.
        """
        if inv_dim==1:
            return flowrhs_func(grid, self.evaluate(grid), self.evaluate_grad(grid)[0], self.evaluate_hess(grid)[0], flow_params)
        else:
            return flowrhs_func(grid, self.evaluate(grid), self.evaluate_grad(grid), self.evaluate_hess(grid), flow_params)