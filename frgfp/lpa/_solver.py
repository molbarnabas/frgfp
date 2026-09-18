"""
Internal implementation of the LPA collocation solver and Numba C-callback bridges.
"""

import inspect
import warnings

import numpy as np
import numba as nb
from numba import types, cfunc
from frgfp._core import FuncFromAnsatz
from frgfp._core import _core_cpp
from . import _lpa_cpp

c_sig = types.void(
    types.CPointer(types.float64), types.CPointer(types.float64),
    types.CPointer(types.float64), types.CPointer(types.float64),
    types.CPointer(types.float64), types.CPointer(types.float64),
    types.intc, types.intc
)

# Batched callback: (I, V, dV, ddV, params, rhs_out, n_points, n_cols, inv_dim)
batch_sig = types.void(
    types.CPointer(types.float64), types.CPointer(types.float64),
    types.CPointer(types.float64), types.CPointer(types.float64),
    types.CPointer(types.float64), types.CPointer(types.float64),
    types.intc, types.intc, types.intc
)

# Compiled callbacks are cached per (flow function, inv_dim, param_dim) so that
# constructing many solvers does not recompile the expensive parallel kernel.
_CALLBACK_CACHE = {}

def _validate_flow_signature(flowrhs_func):
    """
    Validate that the flow callable has the required ``(I, V, dV, ddV, params)`` signature.

    Parameters
    ----------
    flowrhs_func : callable
        The user-supplied flow right-hand side.

    Raises
    ------
    TypeError
        If ``flowrhs_func`` is not callable or does not accept five positional arguments.
    """
    if not callable(flowrhs_func):
        raise TypeError(
            f"'flowrhs_func' must be callable. Received: {type(flowrhs_func).__name__}"
        )
    try:
        parameters = list(inspect.signature(flowrhs_func).parameters.values())
    except (TypeError, ValueError):
        return  # Signature unavailable (e.g. builtin/C callable): skip the arity check.
    if any(p.kind == p.VAR_POSITIONAL for p in parameters):
        return  # Accepts *args: assume it can accept the five flow arguments.
    positional = [p for p in parameters
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if len(positional) != 5:
        raise TypeError(
            "flowrhs_func must accept the five arguments (I, V, dV, ddV, params); "
            f"received a callable with {len(positional)} positional parameter(s)."
        )

def _make_jitted(flowrhs_func):
    """
    Numba-compile the flow function, falling back to no disk cache when needed.

    Parameters
    ----------
    flowrhs_func : callable
        The user-supplied flow right-hand side.

    Returns
    -------
    numba.core.registry.CPUDispatcher
        The lazily compiled flow function.
    """
    try:
        return nb.njit(flowrhs_func, fastmath=True, cache=True)
    except RuntimeError as exc:
        warnings.warn(
            f"flowrhs_func '{getattr(flowrhs_func, '__name__', flowrhs_func)}' cannot be "
            f"cached by Numba ({exc}). Compiling without disk caching instead; define the "
            "flow function at module level to enable caching.",
            RuntimeWarning,
            stacklevel=2,
        )
        return nb.njit(flowrhs_func, fastmath=True)

def _make_scalar_callback(jitted_func, inv_dim, param_dim):
    """
    Compile the scalar (one flow evaluation per collocation point) C-callback.

    Parameters
    ----------
    jitted_func : numba.core.registry.CPUDispatcher
        The Numba-compiled flow function.
    inv_dim : int
        The invariant dimension (selects the 1D or multi-dimensional buffer layout).
    param_dim : int
        The number of flow parameters.

    Returns
    -------
    numba.core.ccallback.CFunc
        The compiled C-callback.
    """
    if inv_dim == 1:
        @cfunc(c_sig)
        def scalar_callback(I_ptr, V_ptr, dV_ptr, ddV_ptr, params_ptr, rhs_out_ptr, n_pts, i_dim):
            """Evaluate the flow point-by-point for a one-dimensional invariant."""
            I_arr = nb.carray(I_ptr, n_pts)
            V_arr = nb.carray(V_ptr, n_pts)
            dV_arr = nb.carray(dV_ptr, n_pts)
            ddV_arr = nb.carray(ddV_ptr, n_pts)
            params_arr = nb.carray(params_ptr, param_dim)
            rhs_out = nb.carray(rhs_out_ptr, n_pts)
            for i in range(n_pts):
                rhs_out[i] = jitted_func(I_arr[i], V_arr[i], dV_arr[i], ddV_arr[i], params_arr)
    else:
        @cfunc(c_sig)
        def scalar_callback(I_ptr, V_ptr, dV_ptr, ddV_ptr, params_ptr, rhs_out_ptr, n_pts, i_dim):
            """Evaluate the flow point-by-point for a multi-dimensional invariant."""
            I_arr = nb.carray(I_ptr, (n_pts, i_dim))
            V_arr = nb.carray(V_ptr, n_pts)
            # The interleaved C++ memory maps exactly onto these 2D/3D shapes.
            dV_arr = nb.carray(dV_ptr, (n_pts, i_dim))
            ddV_arr = nb.carray(ddV_ptr, (n_pts, i_dim, i_dim))
            params_arr = nb.carray(params_ptr, param_dim)
            rhs_out = nb.carray(rhs_out_ptr, n_pts)
            for i in range(n_pts):
                rhs_out[i] = jitted_func(I_arr[i], V_arr[i], dV_arr[i], ddV_arr[i], params_arr)
    return scalar_callback

def _make_batch_callback(jitted_func, param_dim):
    """
    Compile the batched (one parallel launch for all perturbation columns) C-callback.

    Parameters
    ----------
    jitted_func : numba.core.registry.CPUDispatcher
        The Numba-compiled flow function.
    param_dim : int
        The number of flow parameters.

    Returns
    -------
    numba.core.ccallback.CFunc
        The compiled C-callback.
    """
    # One prange launch evaluates every (point, coefficient-column) pair.
    @nb.njit(parallel=True, fastmath=True)
    def batch_kernel(I_arr, V_arr, dV_arr, ddV_arr, params_arr, rhs_arr):
        """Evaluate the flow for every (point, coefficient-column) pair in parallel."""
        n_pts, n_cols = V_arr.shape
        for i in nb.prange(n_pts):
            for j in range(n_cols):
                rhs_arr[i, j] = jitted_func(
                    I_arr[i], V_arr[i, j], dV_arr[i, j], ddV_arr[i, j], params_arr)

    @cfunc(batch_sig)
    def batch_callback(I_ptr, V_ptr, dV_ptr, ddV_ptr, params_ptr,
                       rhs_out_ptr, n_pts, n_cols, i_dim):
        """Wrap the batched parallel kernel as a C-callback."""
        I_arr = nb.carray(I_ptr, n_pts)
        V_arr = nb.carray(V_ptr, (n_pts, n_cols))
        dV_arr = nb.carray(dV_ptr, (n_pts, n_cols))
        ddV_arr = nb.carray(ddV_ptr, (n_pts, n_cols))
        params_arr = nb.carray(params_ptr, param_dim)
        rhs_arr = nb.carray(rhs_out_ptr, (n_pts, n_cols))
        batch_kernel(I_arr, V_arr, dV_arr, ddV_arr, params_arr, rhs_arr)

    # Warm up the parallel runtime while the GIL is still held.
    batch_kernel(
        np.full(1, 0.1), np.full((1, 1), 0.1), np.full((1, 1), 0.1),
        np.full((1, 1), 0.1), np.full(param_dim, 0.1), np.zeros((1, 1)),
    )
    return batch_callback

def _build_callbacks(flowrhs_func, inv_dim, param_dim):
    """
    Build (and cache) the Numba C-callbacks bridging C++ to the flow equation.

    Parameters
    ----------
    flowrhs_func : callable
        The user-defined flow right-hand side ``(I, V, dV, ddV, params)``.
    inv_dim : int
        The invariant dimension; ``1`` enables the batched callback.
    param_dim : int
        The number of flow parameters.

    Returns
    -------
    (scalar_address, batch_address) : tuple of int
        Addresses of the scalar and batched C-callbacks. ``batch_address`` is 0
        when the batched path is unavailable (``inv_dim != 1`` or the kernel
        failed to compile).

    Raises
    ------
    TypeError
        If ``flowrhs_func`` has the wrong signature or cannot be JIT-compiled by Numba.
    """
    cache_key = (flowrhs_func, inv_dim, param_dim)
    cached = _CALLBACK_CACHE.get(cache_key)
    if cached is not None:
        return cached

    _validate_flow_signature(flowrhs_func)
    jitted_func = _make_jitted(flowrhs_func)

    try:
        scalar_address = _make_scalar_callback(jitted_func, inv_dim, param_dim).address
    except Exception as exc:
        raise TypeError(
            f"flowrhs_func '{getattr(flowrhs_func, '__name__', flowrhs_func)}' could not be "
            "JIT-compiled by Numba. It must be a nopython-compatible function with the "
            "signature (I, V, dV, ddV, params)."
        ) from exc

    batch_address = 0
    if inv_dim == 1:
        try:
            batch_address = _make_batch_callback(jitted_func, param_dim).address
        except Exception as exc:
            warnings.warn(
                f"Batched Jacobian kernel unavailable ({type(exc).__name__}: {exc}); "
                "falling back to the scalar Jacobian path.",
                RuntimeWarning,
                stacklevel=2,
            )
            batch_address = 0

    result = (scalar_address, batch_address)
    _CALLBACK_CACHE[cache_key] = result
    return result

class LPACollSolver(_lpa_cpp.LPACollSolver_cpp):
    """
    Fixed-point solver for LPA flow equations using collocation methods.

    Compiles a flow equation with Numba (fastmath) and executes
    zero-allocation, O(N) optimized Newton-Gauss loops in C++.

    Parameters
    ----------
    coll_grid : np.ndarray
        A 2D array of shape `(N_points, inv_dim)` containing the collocation grid.
    ansatz : frgfp.Ansatz
        The basis function expansion object defining the functional space.
    flowrhs_func : callable
        A user-defined function that computes the right-hand side of the flow
        equation. It should accept arguments `(I, V, dV, ddV, params)`.
    inv_dim : int, optional
        The invariant dimension of the problem (default is 1).
    param_dim : int, optional
        The number of flow parameters (default is 2).

    Attributes
    ----------
    coll_grid : np.ndarray
        The collocation grid stored as a C-contiguous float64 array.
    ansatz : frgfp.Ansatz
        The basis function expansion object.
    inv_dim : int
        The invariant dimension.
    param_dim : int
        The number of flow parameters.
    """

    def __init__(self, coll_grid, ansatz, flowrhs_func, inv_dim=1, param_dim=2):
        """
        Initialize the solver with a collocation grid, ansatz, and flow function.

        Parameters
        ----------
        coll_grid : np.ndarray
            A 2D array of shape `(N_points, inv_dim)` containing the collocation grid.
        ansatz : frgfp.Ansatz
            The basis function expansion object defining the functional space.
        flowrhs_func : callable
            A user-defined function that computes the right-hand side of the flow
            equation. It should accept arguments `(I, V, dV, ddV, params)`.
        inv_dim : int, optional
            The invariant dimension of the problem (default is 1).
        param_dim : int, optional
            The number of flow parameters (default is 2).
        """
        if not isinstance(ansatz, _core_cpp.Ansatz):
            raise TypeError(
                "The 'ansatz' parameter must be a valid instantiated Ansatz object "
                "(e.g. PolyAnsatz or ChebyshevAnsatz)."
            )
        if not isinstance(inv_dim, (int, np.integer)) or isinstance(inv_dim, bool):
            raise TypeError(
                f"The 'inv_dim' parameter must be an integer. Received: {type(inv_dim).__name__}"
            )
        if inv_dim < 1:
            raise ValueError(
                f"The 'inv_dim' must be a strictly positive integer. Received: {inv_dim}"
            )
        if not isinstance(param_dim, (int, np.integer)) or isinstance(param_dim, bool):
            raise TypeError(
                f"The 'param_dim' parameter must be an integer. Received: {type(param_dim).__name__}"
            )
        if param_dim < 1:
            raise ValueError(
                f"The 'param_dim' must be a strictly positive integer. Received: {param_dim}"
            )

        self.coll_grid = np.asarray(coll_grid, dtype=np.float64, order='C')
        if self.coll_grid.ndim not in (1, 2):
            raise ValueError(
                "The collocation grid must be a 1D or 2D array; "
                f"received an array with {self.coll_grid.ndim} dimensions."
            )
        if self.coll_grid.ndim == 2 and self.coll_grid.shape[1] != inv_dim:
            raise ValueError(
                f"The collocation grid must have shape (N_points, {inv_dim}); "
                f"received shape {tuple(self.coll_grid.shape)}."
            )
        if inv_dim > 1 and self.coll_grid.ndim != 2:
            raise ValueError(
                f"For inv_dim > 1 the collocation grid must be 2D with shape (N_points, {inv_dim})."
            )

        self.ansatz = ansatz
        self.inv_dim = inv_dim
        self.param_dim = param_dim

        # Compile (or reuse cached) Numba C-callbacks for this flow function.
        self._cfunc_address, self._batch_cfunc_address = _build_callbacks(
            flowrhs_func, self.inv_dim, self.param_dim)

        super().__init__(
            self.coll_grid, self.ansatz, self._cfunc_address, self._batch_cfunc_address,
            self.inv_dim, self.param_dim
        )

    def local_optimize(self, init_coeffs, maxiter=1000, tol=1e-6, flow_params=(3.95, 3), multithread=False):
        """
        Perform a local optimization of the expansion coefficients.

        Parameters
        ----------
        init_coeffs : list of float or np.ndarray
            Initial guess for the expansion coefficients.
        maxiter : int, optional
            Maximum number of Newton iterations (default is 1000).
        tol : float, optional
            Convergence tolerance for the Euclidean norm of the residual vector
            (default is 1e-6).
        flow_params : list or tuple or np.ndarray, optional
            Parameters required by the flow function (default is (3.95, 3)).
        multithread : bool, optional
            Whether to use OpenMP multithreading for the Jacobian evaluation
            (default is False).

        Returns
        -------
        frgfp.FuncFromAnsatz or None
            A `FuncFromAnsatz` object with the optimized coefficients if the
            solver converged, otherwise `None`.
        """
        init_coeffs = np.asarray(init_coeffs, dtype=np.float64, order='C')
        flow_params = np.asarray(flow_params, dtype=np.float64, order='C')
        
        result = super().local_optimize(init_coeffs, maxiter, tol, flow_params, multithread)
        
        if result.success:
            return FuncFromAnsatz(ansatz=self.ansatz, coeffs=result.coeffs)
        return None

    def local_optimize_fixed_iterations(self, init_coeffs, num_iter=1, flow_params=(3.95, 3), multithread=False):
        """
        Perform a fixed number of optimization steps without a tolerance-based
        stopping criterion.

        Unlike :meth:`local_optimize`, the iteration always runs exactly
        ``num_iter`` steps; convergence is not checked. The residual norm of the
        final iterate is available on the underlying ``OptimizeResult``.

        Parameters
        ----------
        init_coeffs : list of float or np.ndarray
            Initial guess for the expansion coefficients.
        num_iter : int, optional
            Exact number of iteration steps to perform (default is 1).
        flow_params : list or tuple or np.ndarray, optional
            Parameters required by the flow function (default is (3.95, 3)).
        multithread : bool, optional
            Whether to use OpenMP multithreading for the Jacobian evaluation
            (default is False).

        Returns
        -------
        frgfp.FuncFromAnsatz or None
            A `FuncFromAnsatz` object with the resulting coefficients, or
            ``None`` if the residual norm became non-finite.
        """
        init_coeffs = np.asarray(init_coeffs, dtype=np.float64, order='C')
        flow_params = np.asarray(flow_params, dtype=np.float64, order='C')

        result = super().local_optimize_fixed_iterations(
            init_coeffs, num_iter, flow_params, multithread)

        if result.success:
            return FuncFromAnsatz(ansatz=self.ansatz, coeffs=result.coeffs)
        return None

    def param_path_following(self, init_coeffs, flow_params_path, maxiter=1000, tol=1e-6, multithread=True):
        """
        Follow a path of flow parameters, optimizing at each step.

        Parameters
        ----------
        init_coeffs : list of float or np.ndarray
            Initial guess for the expansion coefficients.
        flow_params_path : list of list or tuple or np.ndarray
            A sequence of parameter vectors, each of length `param_dim`.
        maxiter : int, optional
            Maximum number of Newton iterations per step (default is 1000).
        tol : float, optional
            Convergence tolerance for the Euclidean norm of the residual vector
            (default is 1e-6).
        multithread : bool, optional
            Whether to use OpenMP multithreading for the Jacobian evaluation
            (default is True).

        Returns
        -------
        list of frgfp.FuncFromAnsatz or None
            A list where each element is a `FuncFromAnsatz` object if the
            corresponding step converged, otherwise `None`.
        """
        init_coeffs = np.asarray(init_coeffs, dtype=np.float64, order='C')
        path_arrays = [np.asarray(p, dtype=np.float64, order='C') for p in flow_params_path]
        
        cpp_results = super().param_path_following(init_coeffs, maxiter, tol, path_arrays, multithread)
        
        return [FuncFromAnsatz(ansatz=self.ansatz, coeffs=res.coeffs) if res.success else None for res in cpp_results]

    def multistart_optimize(self, init_coeffs_list, flow_params=(3.95, 3), maxiter=1000, tol=1e-6):
        """
        Perform a multistart optimization from multiple initial guesses.

        Parameters
        ----------
        init_coeffs_list : list of list or np.ndarray
            A list of initial coefficient vectors.
        flow_params : list or tuple or np.ndarray, optional
            Parameters required by the flow function (default is (3.95, 3)).
        maxiter : int, optional
            Maximum number of Newton iterations per start (default is 1000).
        tol : float, optional
            Convergence tolerance for the Euclidean norm of the residual vector
            (default is 1e-6).

        Returns
        -------
        list of frgfp.FuncFromAnsatz or None
            A list where each element is a `FuncFromAnsatz` object if the
            corresponding start converged, otherwise `None`.
        """
        flow_params = np.asarray(flow_params, dtype=np.float64, order='C')
        init_list_arrays = [np.asarray(ic, dtype=np.float64, order='C') for ic in init_coeffs_list]
        
        cpp_results = super().multistart_optimize(init_list_arrays, maxiter, tol, flow_params)
        
        return [FuncFromAnsatz(ansatz=self.ansatz, coeffs=res.coeffs) if res.success else None for res in cpp_results]
