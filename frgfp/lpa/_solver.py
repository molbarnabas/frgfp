"""
Internal implementation of the LPA collocation solver and Numba C-callback bridges.
"""

import numpy as np
import numba as nb
from numba import types, cfunc
from frgfp._core import FuncFromAnsatz
from . import _lpa_cpp

c_sig = types.void(
    types.CPointer(types.float64), types.CPointer(types.float64),
    types.CPointer(types.float64), types.CPointer(types.float64),
    types.CPointer(types.float64), types.CPointer(types.float64),
    types.intc, types.intc
)

class LPACollSolver(_lpa_cpp.LPACollSolver_cpp):
    """
    Fixed-point solver for LPA flow equations using collocation methods.
    
    Compiles a flow equation with Numba (fastmath) and executes 
    zero-allocation, O(N) optimized Newton-Gauss loops in C++.
    """
    def __init__(self, coll_grid, ansatz, flowrhs_func, inv_dim=1, param_dim=2):
        self.coll_grid = np.asarray(coll_grid, dtype=np.float64, order='C')
        self.ansatz = ansatz
        self.inv_dim = inv_dim
        self.param_dim = param_dim
        
        n_points = self.coll_grid.shape[0] if self.coll_grid.ndim > 1 or inv_dim == 1 else len(self.coll_grid)

        # SIMD vectorization
        jitted_func = nb.njit(flowrhs_func, fastmath=True, cache=True)

        if self.inv_dim == 1:
            @cfunc(c_sig)
            def flowrhs_c_callback(I_ptr, V_ptr, dV_ptr, ddV_ptr, params_ptr, rhs_out_ptr, n_pts, i_dim):
                I_arr = nb.carray(I_ptr, n_pts)
                V_arr = nb.carray(V_ptr, n_pts)
                dV_arr = nb.carray(dV_ptr, n_pts)
                ddV_arr = nb.carray(ddV_ptr, n_pts)
                params_arr = nb.carray(params_ptr, param_dim)
                rhs_out = nb.carray(rhs_out_ptr, n_pts)
                for i in range(n_pts):
                    rhs_out[i] = jitted_func(I_arr[i], V_arr[i], dV_arr[i], ddV_arr[i], params_arr)
            self._cfunc_address = flowrhs_c_callback.address
        else:
            @cfunc(c_sig)
            def flowrhs_c_callback(I_ptr, V_ptr, dV_ptr, ddV_ptr, params_ptr, rhs_out_ptr, n_pts, i_dim):
                I_arr = nb.carray(I_ptr, (n_pts, i_dim))
                V_arr = nb.carray(V_ptr, n_pts)
                # The interleaved C++ memory perfectly maps to these 2D/3D shapes!
                dV_arr = nb.carray(dV_ptr, (n_pts, i_dim))
                ddV_arr = nb.carray(ddV_ptr, (n_pts, i_dim, i_dim))
                params_arr = nb.carray(params_ptr, param_dim)
                rhs_out = nb.carray(rhs_out_ptr, n_pts)
                for i in range(n_pts):
                    rhs_out[i] = jitted_func(I_arr[i], V_arr[i], dV_arr[i], ddV_arr[i], params_arr)
            self._cfunc_address = flowrhs_c_callback.address

        super().__init__(
            self.coll_grid, self.ansatz, self._cfunc_address, 
            self.inv_dim, self.param_dim
        )

    def local_optimize(self, init_coeffs, maxiter=1000, tol=1e-6, flow_params=(3.95, 3), multithread=False):
        init_coeffs = np.asarray(init_coeffs, dtype=np.float64, order='C')
        flow_params = np.asarray(flow_params, dtype=np.float64, order='C')
        
        result = super().local_optimize(init_coeffs, maxiter, tol, flow_params, multithread)
        
        if result.success:
            return FuncFromAnsatz(ansatz=self.ansatz, coeffs=result.coeffs)
        return None

    def param_path_following(self, init_coeffs, flow_params_path, maxiter=1000, tol=1e-6, multithread=True):
        init_coeffs = np.asarray(init_coeffs, dtype=np.float64, order='C')
        path_arrays = [np.asarray(p, dtype=np.float64, order='C') for p in flow_params_path]
        
        cpp_results = super().param_path_following(init_coeffs, maxiter, tol, path_arrays, multithread)
        
        return [FuncFromAnsatz(ansatz=self.ansatz, coeffs=res.coeffs) if res.success else None for res in cpp_results]

    def multistart_optimize(self, init_coeffs_list, flow_params=(3.95, 3), maxiter=1000, tol=1e-6):
        flow_params = np.asarray(flow_params, dtype=np.float64, order='C')
        init_list_arrays = [np.asarray(ic, dtype=np.float64, order='C') for ic in init_coeffs_list]
        
        cpp_results = super().multistart_optimize(init_list_arrays, maxiter, tol, flow_params)
        
        return [FuncFromAnsatz(ansatz=self.ansatz, coeffs=res.coeffs) if res.success else None for res in cpp_results]