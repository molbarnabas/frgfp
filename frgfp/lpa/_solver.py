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

def _bind_threading_layer_to_openmp():
    """
    Bind Numba's parallel layer to OpenMP before it is initialised.

    The C++ backend links OpenMP and relies on ``omp_in_parallel()`` to detect
    nested parallel regions (e.g. ``multistart_optimize`` runs an OpenMP loop
    around solver calls that must not spawn a nested parallel team).  Those
    guards only work if Numba also uses the OpenMP layer, so relying on Numba's
    default layer selection is not safe -- with a sufficiently new TBB installed
    Numba would pick TBB and the guards would always report "not in a parallel
    region".

    Selecting ``omp`` explicitly also avoids a spurious warning: Numba probes for
    TBB by loading ``libtbb.so.12``, which on systems shipping a TBB older than
    2021 update 6 (e.g. Ubuntu 22.04's TBB 2021.5, interface version 12050)
    resolves to the system library and is then rejected with a NumbaWarning
    before falling back to OpenMP anyway.

    The user's own choice always wins: an explicitly configured layer, or a layer
    that has already been initialised, is left untouched.
    """
    if nb.config.THREADING_LAYER != "default":
        return  # An explicit NUMBA_THREADING_LAYER / config setting wins.
    try:
        nb.threading_layer()
    except ValueError:
        # Not initialised yet, so it is still safe to choose the layer.
        nb.config.THREADING_LAYER = "omp"

_bind_threading_layer_to_openmp()

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

# Sensitivity callback (inv_dim == 1):
# (I, V, dV, ddV, params, fV_out, fdV_out, fddV_out, n_points)
sens_sig = types.void(
    types.CPointer(types.float64), types.CPointer(types.float64),
    types.CPointer(types.float64), types.CPointer(types.float64),
    types.CPointer(types.float64),
    types.CPointer(types.float64), types.CPointer(types.float64),
    types.CPointer(types.float64),
    types.intc
)

# Compiled callbacks are cached per (flow function, inv_dim, param_dim,
# build_batch, build_sensitivity, warm_parallel_sensitivity) so that
# constructing many solvers does not recompile the expensive parallel kernel.
_CALLBACK_CACHE = {}

# Mirrors kBatchThreshold in frgfp/lpa/src/_solver.cpp. Below this many expansion
# coefficients the C++ solver never selects the batched finite-difference
# Jacobian, so its (non-disk-cacheable) Numba kernel is not compiled. The gate is
# on the *total* number of coefficients, which already accounts for the tensor
# structure: a 2D ansatz of order (n, n) has (n + 1)**2 coefficients, so it
# crosses the same threshold at a much lower per-dimension order than a 1D
# ansatz.
_BATCH_THRESHOLD = 64

# Mirrors kSensitivityThreshold in frgfp/lpa/src/_solver.cpp. Below this many
# expansion coefficients the C++ solver never selects the rank-1 sensitivity
# Jacobian, so the corresponding (expensive and non-disk-cacheable) Numba
# kernels must not be compiled eagerly: on tiny problems that only adds
# start-up latency for code that can never run. As above, the gate uses the
# total coefficient count, independently of the invariant dimension.
_SENSITIVITY_THRESHOLD = 16

# Mirrors kSensitivityParallelMinPoints in frgfp/lpa/src/_solver.cpp. Below this
# many collocation points the C++ solver never selects the parallel sensitivity
# kernel, so its unconditional warm-up (which spins up the OpenMP pool) can be
# skipped. The warm-up is only needed to initialise Numba's threading layer
# while the GIL is held, which matters exclusively for the parallel variant.
_SENSITIVITY_PARALLEL_MIN_POINTS = 512

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

def _make_jitted(flowrhs_func, inline=False):
    """
    Numba-compile the flow function, falling back to no disk cache when needed.

    Parameters
    ----------
    flowrhs_func : callable
        The user-supplied flow right-hand side.
    inline : bool, optional
        When True, request ``inline='always'`` so that the hot batch and
        sensitivity kernels do not pay a call boundary per point (this also lets
        LLVM hoist loop-invariant work such as ``math.gamma`` out of the loops).

    Returns
    -------
    numba.core.registry.CPUDispatcher
        The lazily compiled flow function.
    """
    extra = {"inline": "always"} if inline else {}
    try:
        return nb.njit(flowrhs_func, fastmath=True, cache=True, **extra)
    except RuntimeError as exc:
        warnings.warn(
            f"flowrhs_func '{getattr(flowrhs_func, '__name__', flowrhs_func)}' cannot be "
            f"cached by Numba ({exc}). Compiling without disk caching instead; define the "
            "flow function at module level to enable caching.",
            RuntimeWarning,
            stacklevel=2,
        )
        return nb.njit(flowrhs_func, fastmath=True, **extra)

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

def _make_sensitivity_callbacks(jitted_func, param_dim, inv_dim, warm_parallel=True):
    """
    Compile the rank-1 Jacobian sensitivity kernels (serial and parallel).

    The flow RHS is pointwise in ``(I, V, dV, ddV)``, so its Jacobian with
    respect to the expansion coefficients is a sum of rank-1 (diagonal scaling)
    terms::

        J = diag(f_V) * M_val
          + sum_a     diag(f_dV_a)   * M_grad_a
          + sum_{a,b} diag(f_ddV_ab) * M_hess_ab

    The scalar partial derivatives are obtained by central differences, which
    needs ``2 * (1 + inv_dim + inv_dim**2)`` flow evaluations per collocation
    point instead of the ``2 * n_coeffs`` evaluations of the finite-difference
    Jacobian.  For ``inv_dim == 1`` that is 6 evaluations per point.

    Parameters
    ----------
    jitted_func : numba.core.registry.CPUDispatcher
        The Numba-compiled flow function.
    param_dim : int
        The number of flow parameters.
    inv_dim : int
        The invariant dimension.  ``1`` selects the flat scalar fast path; any
        larger value uses the general tensor path.
    warm_parallel : bool, optional
        Whether to run the parallel kernel once with the GIL held.  The C++ side
        only selects the parallel variant from ``_SENSITIVITY_PARALLEL_MIN_POINTS``
        collocation points upwards, so the caller disables this for smaller
        grids to avoid needlessly initialising Numba's threading layer.

    Returns
    -------
    (serial_address, parallel_address) : tuple of int
        Addresses of the serial and parallel sensitivity C-callbacks.
    """
    if inv_dim != 1:
        return _make_sensitivity_callbacks_nd(jitted_func, param_dim, inv_dim, warm_parallel)

    return _make_sensitivity_callbacks_1d(jitted_func, param_dim, warm_parallel)


def _make_sensitivity_callbacks_1d(jitted_func, param_dim, warm_parallel=True):
    """
    Rank-1 sensitivity kernels for ``inv_dim == 1`` (flat scalar fast path).

    Parameters
    ----------
    jitted_func : numba.core.registry.CPUDispatcher
        The Numba-compiled flow function.
    param_dim : int
        The number of flow parameters.
    warm_parallel : bool, optional
        Run the parallel kernel once with the GIL held (see
        ``_make_sensitivity_callbacks``).

    Returns
    -------
    (serial_address, parallel_address) : tuple of int
        Addresses of the serial and parallel sensitivity C-callbacks.
    """
    @nb.njit(fastmath=True)
    def sens_kernel_serial(I_arr, V_arr, dV_arr, ddV_arr, params_arr, fV, fdV, fddV):
        """Evaluate the flow sensitivities point-by-point (single-threaded)."""
        n_pts = V_arr.shape[0]
        for i in range(n_pts):
            v = V_arr[i]
            dv = dV_arr[i]
            ddv = ddV_arr[i]
            ii = I_arr[i]
            hv = 1e-6 * max(1.0, abs(v))
            hd = 1e-6 * max(1.0, abs(dv))
            hh = 1e-6 * max(1.0, abs(ddv))
            fV[i] = (jitted_func(ii, v + hv, dv, ddv, params_arr)
                     - jitted_func(ii, v - hv, dv, ddv, params_arr)) / (2.0 * hv)
            fdV[i] = (jitted_func(ii, v, dv + hd, ddv, params_arr)
                      - jitted_func(ii, v, dv - hd, ddv, params_arr)) / (2.0 * hd)
            fddV[i] = (jitted_func(ii, v, dv, ddv + hh, params_arr)
                       - jitted_func(ii, v, dv, ddv - hh, params_arr)) / (2.0 * hh)

    @nb.njit(parallel=True, fastmath=True)
    def sens_kernel_parallel(I_arr, V_arr, dV_arr, ddV_arr, params_arr, fV, fdV, fddV):
        """Evaluate the flow sensitivities across points in parallel."""
        n_pts = V_arr.shape[0]
        for i in nb.prange(n_pts):
            v = V_arr[i]
            dv = dV_arr[i]
            ddv = ddV_arr[i]
            ii = I_arr[i]
            hv = 1e-6 * max(1.0, abs(v))
            hd = 1e-6 * max(1.0, abs(dv))
            hh = 1e-6 * max(1.0, abs(ddv))
            fV[i] = (jitted_func(ii, v + hv, dv, ddv, params_arr)
                     - jitted_func(ii, v - hv, dv, ddv, params_arr)) / (2.0 * hv)
            fdV[i] = (jitted_func(ii, v, dv + hd, ddv, params_arr)
                      - jitted_func(ii, v, dv - hd, ddv, params_arr)) / (2.0 * hd)
            fddV[i] = (jitted_func(ii, v, dv, ddv + hh, params_arr)
                       - jitted_func(ii, v, dv, ddv - hh, params_arr)) / (2.0 * hh)

    # Force compilation of the serial kernel now (raises on a typing failure).
    sens_kernel_serial(
        np.full(1, 0.1), np.full(1, 0.1), np.full(1, 0.1), np.full(1, 0.1),
        np.full(param_dim, 0.1), np.zeros(1), np.zeros(1), np.zeros(1),
    )

    @cfunc(sens_sig)
    def serial_callback(I_ptr, V_ptr, dV_ptr, ddV_ptr, params_ptr,
                        fV_ptr, fdV_ptr, fddV_ptr, n_pts):
        """Wrap the serial sensitivity kernel as a C-callback."""
        sens_kernel_serial(
            nb.carray(I_ptr, n_pts), nb.carray(V_ptr, n_pts),
            nb.carray(dV_ptr, n_pts), nb.carray(ddV_ptr, n_pts),
            nb.carray(params_ptr, param_dim),
            nb.carray(fV_ptr, n_pts), nb.carray(fdV_ptr, n_pts),
            nb.carray(fddV_ptr, n_pts),
        )

    @cfunc(sens_sig)
    def parallel_callback(I_ptr, V_ptr, dV_ptr, ddV_ptr, params_ptr,
                          fV_ptr, fdV_ptr, fddV_ptr, n_pts):
        """Wrap the parallel sensitivity kernel as a C-callback."""
        sens_kernel_parallel(
            nb.carray(I_ptr, n_pts), nb.carray(V_ptr, n_pts),
            nb.carray(dV_ptr, n_pts), nb.carray(ddV_ptr, n_pts),
            nb.carray(params_ptr, param_dim),
            nb.carray(fV_ptr, n_pts), nb.carray(fdV_ptr, n_pts),
            nb.carray(fddV_ptr, n_pts),
        )

    # Warm up the parallel runtime while the GIL is still held. This must happen
    # here, at callback-construction time: the kernels are later invoked from C++
    # with the GIL released (py::call_guard<py::gil_scoped_release>), and letting
    # Numba's threading layer initialise in that context can deadlock. The C++
    # solver only selects the parallel kernel from
    # _SENSITIVITY_PARALLEL_MIN_POINTS collocation points upwards, so the caller
    # skips the warm-up for smaller grids, where it would only spin up an idle
    # OpenMP pool.
    if warm_parallel:
        sens_kernel_parallel(
            np.full(1, 0.1), np.full(1, 0.1), np.full(1, 0.1), np.full(1, 0.1),
            np.full(param_dim, 0.1), np.zeros(1), np.zeros(1), np.zeros(1),
        )

    return serial_callback.address, parallel_callback.address

def _make_sensitivity_callbacks_nd(jitted_func, param_dim, inv_dim, warm_parallel=True):
    """
    Rank-1 sensitivity kernels for ``inv_dim > 1`` (general tensor path).

    ``dV`` is a length-``inv_dim`` vector and ``ddV`` an ``inv_dim`` x
    ``inv_dim`` matrix at every collocation point, so the derivative set is
    ``f_V`` (scalar), ``f_dV`` (vector) and ``f_ddV`` (matrix).  Every component
    is finite-differenced independently, costing
    ``2 * (1 + inv_dim + inv_dim**2)`` flow evaluations per point -- 14 for two
    invariants, 26 for three -- against ``2 * n_coeffs`` for the
    coefficient-space finite-difference Jacobian.

    The scratch used to build the perturbed inputs is indexed by
    ``numba.get_thread_id()`` in the parallel kernel, so nothing is allocated
    inside the parallel loop and the buffers cannot race.

    Parameters
    ----------
    jitted_func : numba.core.registry.CPUDispatcher
        The Numba-compiled flow function.
    param_dim : int
        The number of flow parameters.
    inv_dim : int
        The invariant dimension (must be greater than one).
    warm_parallel : bool, optional
        Run the parallel kernel once with the GIL held (see the 1D path).

    Returns
    -------
    (serial_address, parallel_address) : tuple of int
        Addresses of the serial and parallel sensitivity C-callbacks.
    """
    @nb.njit(fastmath=True, inline="always")
    def sens_point(i, I_arr, V_arr, dV_arr, ddV_arr, params_arr, fV, fdV, fddV,
                   dv_p, dv_m, ddv_p, ddv_m):
        """Evaluate every flow sensitivity at a single collocation point."""
        v = V_arr[i]
        ii = I_arr[i]

        hv = 1e-6 * max(1.0, abs(v))
        fV[i] = (jitted_func(ii, v + hv, dV_arr[i], ddV_arr[i], params_arr)
                 - jitted_func(ii, v - hv, dV_arr[i], ddV_arr[i], params_arr)) / (2.0 * hv)

        for a in range(inv_dim):
            hd = 1e-6 * max(1.0, abs(dV_arr[i, a]))
            for b in range(inv_dim):
                dv_p[b] = dV_arr[i, b]
                dv_m[b] = dV_arr[i, b]
            dv_p[a] += hd
            dv_m[a] -= hd
            fdV[i, a] = (jitted_func(ii, v, dv_p, ddV_arr[i], params_arr)
                         - jitted_func(ii, v, dv_m, ddV_arr[i], params_arr)) / (2.0 * hd)

        for a in range(inv_dim):
            for b in range(inv_dim):
                hh = 1e-6 * max(1.0, abs(ddV_arr[i, a, b]))
                for c in range(inv_dim):
                    for e in range(inv_dim):
                        ddv_p[c, e] = ddV_arr[i, c, e]
                        ddv_m[c, e] = ddV_arr[i, c, e]
                ddv_p[a, b] += hh
                ddv_m[a, b] -= hh
                fddV[i, a, b] = (jitted_func(ii, v, dV_arr[i], ddv_p, params_arr)
                                 - jitted_func(ii, v, dV_arr[i], ddv_m, params_arr)) / (2.0 * hh)

    @nb.njit(fastmath=True)
    def sens_kernel_serial(I_arr, V_arr, dV_arr, ddV_arr, params_arr, fV, fdV, fddV):
        """Evaluate the flow sensitivities point-by-point (single-threaded)."""
        n_pts = V_arr.shape[0]
        dv_p = np.empty(inv_dim)
        dv_m = np.empty(inv_dim)
        ddv_p = np.empty((inv_dim, inv_dim))
        ddv_m = np.empty((inv_dim, inv_dim))
        for i in range(n_pts):
            sens_point(i, I_arr, V_arr, dV_arr, ddV_arr, params_arr,
                       fV, fdV, fddV, dv_p, dv_m, ddv_p, ddv_m)

    @nb.njit(parallel=True, fastmath=True)
    def sens_kernel_parallel(I_arr, V_arr, dV_arr, ddV_arr, params_arr, fV, fdV, fddV):
        """Evaluate the flow sensitivities across points in parallel."""
        n_pts = V_arr.shape[0]
        n_threads = nb.get_num_threads()
        dv_p = np.empty((n_threads, inv_dim))
        dv_m = np.empty((n_threads, inv_dim))
        ddv_p = np.empty((n_threads, inv_dim, inv_dim))
        ddv_m = np.empty((n_threads, inv_dim, inv_dim))
        for i in nb.prange(n_pts):
            tid = nb.get_thread_id()
            sens_point(i, I_arr, V_arr, dV_arr, ddV_arr, params_arr,
                       fV, fdV, fddV, dv_p[tid], dv_m[tid], ddv_p[tid], ddv_m[tid])

    # Force compilation of the serial kernel now (raises on a typing failure).
    sens_kernel_serial(
        np.full((1, inv_dim), 0.1), np.full(1, 0.1), np.full((1, inv_dim), 0.1),
        np.full((1, inv_dim, inv_dim), 0.1), np.full(param_dim, 0.1),
        np.zeros(1), np.zeros((1, inv_dim)), np.zeros((1, inv_dim, inv_dim)),
    )

    @cfunc(sens_sig)
    def serial_callback(I_ptr, V_ptr, dV_ptr, ddV_ptr, params_ptr,
                        fV_ptr, fdV_ptr, fddV_ptr, n_pts):
        """Wrap the serial sensitivity kernel as a C-callback."""
        sens_kernel_serial(
            nb.carray(I_ptr, (n_pts, inv_dim)),
            nb.carray(V_ptr, n_pts),
            nb.carray(dV_ptr, (n_pts, inv_dim)),
            nb.carray(ddV_ptr, (n_pts, inv_dim, inv_dim)),
            nb.carray(params_ptr, param_dim),
            nb.carray(fV_ptr, n_pts),
            nb.carray(fdV_ptr, (n_pts, inv_dim)),
            nb.carray(fddV_ptr, (n_pts, inv_dim, inv_dim)),
        )

    @cfunc(sens_sig)
    def parallel_callback(I_ptr, V_ptr, dV_ptr, ddV_ptr, params_ptr,
                          fV_ptr, fdV_ptr, fddV_ptr, n_pts):
        """Wrap the parallel sensitivity kernel as a C-callback."""
        sens_kernel_parallel(
            nb.carray(I_ptr, (n_pts, inv_dim)),
            nb.carray(V_ptr, n_pts),
            nb.carray(dV_ptr, (n_pts, inv_dim)),
            nb.carray(ddV_ptr, (n_pts, inv_dim, inv_dim)),
            nb.carray(params_ptr, param_dim),
            nb.carray(fV_ptr, n_pts),
            nb.carray(fdV_ptr, (n_pts, inv_dim)),
            nb.carray(fddV_ptr, (n_pts, inv_dim, inv_dim)),
        )

    # Warm up the parallel runtime while the GIL is still held (see the 1D path).
    if warm_parallel:
        sens_kernel_parallel(
            np.full((1, inv_dim), 0.1), np.full(1, 0.1), np.full((1, inv_dim), 0.1),
            np.full((1, inv_dim, inv_dim), 0.1), np.full(param_dim, 0.1),
            np.zeros(1), np.zeros((1, inv_dim)), np.zeros((1, inv_dim, inv_dim)),
        )

    return serial_callback.address, parallel_callback.address

def _build_callbacks(flowrhs_func, inv_dim, param_dim,
                     build_batch=True, build_sensitivity=True,
                     warm_parallel_sensitivity=True):
    """
    Build (and cache) the Numba C-callbacks bridging C++ to the flow equation.

    Parameters
    ----------
    flowrhs_func : callable
        The user-defined flow right-hand side ``(I, V, dV, ddV, params)``.
    inv_dim : int
        The invariant dimension; ``1`` additionally enables the batched
        finite-difference callback.
    param_dim : int
        The number of flow parameters.
    build_batch : bool, optional
        Whether to compile the batched finite-difference kernel.  The C++ solver
        only selects it for ``inv_dim == 1`` with
        ``n_coeffs >= _BATCH_THRESHOLD``, so smaller problems pass ``False`` to
        skip the (non-disk-cacheable) compilation.
    build_sensitivity : bool, optional
        Whether to compile the rank-1 sensitivity kernels.  The C++ solver only
        selects them for ``n_coeffs >= _SENSITIVITY_THRESHOLD``, so smaller
        problems pass ``False`` to skip the (non-disk-cacheable) compilation.
    warm_parallel_sensitivity : bool, optional
        Whether to warm up the parallel sensitivity kernel with the GIL held.
        Only needed when the C++ solver can actually select it (a large enough
        grid); ignored when ``build_sensitivity`` is false.

    Returns
    -------
    (scalar_address, batch_address, sens_address, sens_par_address) : tuple of int
        Addresses of the scalar, batched and (serial/parallel) rank-1 sensitivity
        C-callbacks. ``batch_address`` is 0 unless ``inv_dim == 1`` and
        ``build_batch`` is true; the sensitivity addresses are 0 unless
        ``build_sensitivity`` is true.

    Raises
    ------
    TypeError
        If ``flowrhs_func`` has the wrong signature or cannot be JIT-compiled by Numba.
    """
    cache_key = (flowrhs_func, inv_dim, param_dim,
                 build_batch, build_sensitivity, warm_parallel_sensitivity)
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

    # An aggressively inlined variant removes the per-point call boundary in the
    # hot kernels; fall back to the plain dispatcher when inlining is not possible.
    try:
        jitted_inline = _make_jitted(flowrhs_func, inline=True)
    except Exception:
        jitted_inline = None

    def _first_working(builder, label):
        """Run `builder` on each dispatcher candidate; 0 + warning on total failure."""
        last_exc = None
        for candidate in (jitted_inline, jitted_func):
            if candidate is None:
                continue
            try:
                return builder(candidate)
            except Exception as exc:  # pragma: no cover - defensive fallback
                last_exc = exc
        warnings.warn(
            f"{label} unavailable ({type(last_exc).__name__}: {last_exc}); "
            "falling back to the scalar finite-difference Jacobian path.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0

    batch_address = 0
    sens_address = 0
    sens_par_address = 0
    if build_batch and inv_dim == 1:
        # The batched finite-difference kernel assumes scalar dV/ddV layouts and
        # therefore only exists for a single invariant. It is only compiled when
        # the problem is large enough for the C++ solver to select it, so that
        # small problems do not pay its (non-disk-cacheable) compilation cost.
        batch_address = _first_working(
            lambda jf: _make_batch_callback(jf, param_dim).address,
            "Batched Jacobian kernel",
        )

    # The rank-1 sensitivity kernels work for any invariant dimension. They are
    # only compiled when the problem is large enough for the C++ solver to select
    # them (n_coeffs >= kSensitivityThreshold): for tiny problems doing so only
    # adds start-up latency, because the kernels are closures and therefore
    # cannot be disk-cached by Numba. The C++ side treats the resulting null
    # callback pointers as "rank-1 path unavailable".
    if build_sensitivity:
        sens_pack = _first_working(
            lambda jf: _make_sensitivity_callbacks(
                jf, param_dim, inv_dim,
                warm_parallel=warm_parallel_sensitivity),
            "Rank-1 sensitivity kernel",
        )
        if sens_pack:
            sens_address, sens_par_address = sens_pack

    result = (scalar_address, batch_address, sens_address, sens_par_address)
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

        # The C++ solver only selects the auxiliary Jacobian kernels above fixed
        # coefficient-count thresholds (kBatchThreshold, kSensitivityThreshold).
        # Those gates are counted on the *total* number of coefficients, which
        # already encodes the tensor structure -- a 2D ansatz of order (n, n) has
        # (n + 1)**2 coefficients, so it reaches a threshold at a much lower
        # per-dimension order than a 1D ansatz. Compiling (and warming up) a
        # kernel that can never run only adds start-up latency, so the decision
        # is taken here from ansatz.num_coeffs.
        num_coeffs = ansatz.num_coeffs
        build_batch = num_coeffs >= _BATCH_THRESHOLD
        build_sensitivity = num_coeffs >= _SENSITIVITY_THRESHOLD
        warm_parallel_sensitivity = (
            build_sensitivity
            and self.coll_grid.shape[0] >= _SENSITIVITY_PARALLEL_MIN_POINTS
        )

        # Compile (or reuse cached) Numba C-callbacks for this flow function.
        (self._cfunc_address, self._batch_cfunc_address,
         self._sens_cfunc_address, self._sens_par_cfunc_address) = _build_callbacks(
            flowrhs_func, self.inv_dim, self.param_dim,
            build_batch=build_batch,
            build_sensitivity=build_sensitivity,
            warm_parallel_sensitivity=warm_parallel_sensitivity)

        super().__init__(
            self.coll_grid, self.ansatz,
            self._cfunc_address, self._batch_cfunc_address,
            self._sens_cfunc_address, self._sens_par_cfunc_address,
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
            Whether to parallelise the compute-bound flow-evaluation kernels
            (the rank-1 sensitivity kernel, or the finite-difference callbacks
            when the rank-1 path is unavailable). When False the solver runs
            strictly single-threaded (default is False).

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
            Whether to parallelise the compute-bound flow-evaluation kernels
            (the rank-1 sensitivity kernel, or the finite-difference callbacks
            when the rank-1 path is unavailable). When False the solver runs
            strictly single-threaded (default is False).

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
            Whether to parallelise the compute-bound flow-evaluation kernels
            (the rank-1 sensitivity kernel, or the finite-difference callbacks
            when the rank-1 path is unavailable). When False the solver runs
            strictly single-threaded (default is True).

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
