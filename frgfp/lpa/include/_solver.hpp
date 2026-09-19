#pragma once
#include <Eigen/Dense>
#include <memory>
#include <vector>
#include "frgfp/_core/include/_ansatz.hpp"

namespace frgfp {
namespace lpa {
    
struct OptimizeResult {
    Eigen::VectorXd coeffs;
    bool success;
    int iterations;
    double error;
};

typedef void (*FlowRhsFunc)(
    const double* I, const double* V, const double* dV, const double* ddV, 
    const double* params, double* rhs_out, int n_points, int inv_dim
);

// Batched variant: evaluates the flow for every (point, column) pair at once.
// All matrices are row-major: V/dV/ddV *out* are (n_points, n_cols); for
// inv_dim == 1 the derivative buffers are also (n_points, n_cols).
typedef void (*BatchRhsFunc)(
    const double* I, const double* V, const double* dV, const double* ddV,
    const double* params, double* rhs_out, int n_points, int n_cols, int inv_dim
);

// Sensitivity variant (inv_dim == 1 only): evaluates the partial derivatives of
// the (pointwise) scalar flow RHS with respect to V, dV and ddV at every
// collocation point. Because the flow is pointwise in (I, V, dV, ddV), the
// Jacobian is the rank-1 (diagonal-scaling) combination
//     J = diag(f_V) * M_val + diag(f_dV) * M_grad + diag(f_ddV) * M_hess,
// so only 6 flow evaluations per point are needed instead of 2*n_coeffs.
typedef void (*SensRhsFunc)(
    const double* I, const double* V, const double* dV, const double* ddV,
    const double* params, double* fv_out, double* fdv_out, double* fddv_out,
    int n_points
);

class LPACollSolver_cpp {
public:
    LPACollSolver_cpp(const Eigen::MatrixXd& coll_grid,
                      std::shared_ptr<core::Ansatz> ansatz,
                      size_t cfunc_ptr, size_t batch_cfunc_ptr,
                      size_t sens_cfunc_ptr, size_t sens_par_cfunc_ptr,
                      int inv_dim, int param_dim);

    OptimizeResult local_optimize(const Eigen::VectorXd& init_coeffs,
                                  int maxiter, double tol,
                                  const Eigen::VectorXd& flow_params,
                                  bool multithread = false);

    // Performs exactly num_iter iteration steps, without any tolerance-based
    // stopping criterion. The residual norm of the final iterate is reported
    // in OptimizeResult::error.
    OptimizeResult local_optimize_fixed_iterations(const Eigen::VectorXd& init_coeffs,
                                                   int num_iter,
                                                   const Eigen::VectorXd& flow_params,
                                                   bool multithread = false);

    std::vector<OptimizeResult> param_path_following(
        const Eigen::VectorXd& init_coeffs,
        int maxiter, double tol,
        const std::vector<Eigen::VectorXd>& flow_params_path,
        bool multithread = true);

    std::vector<OptimizeResult> multistart_optimize(
        const std::vector<Eigen::VectorXd>& init_coeffs_list,
        int maxiter, double tol,
        const Eigen::VectorXd& flow_params);

private:
    using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    Eigen::MatrixXd coll_grid_; 
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> coll_grid_rm_;
    
    std::shared_ptr<core::Ansatz> ansatz_;
    FlowRhsFunc flowrhs_c_func_;
    BatchRhsFunc flowrhs_batch_c_func_;
    SensRhsFunc flowrhs_sens_c_func_;
    SensRhsFunc flowrhs_sens_par_c_func_;
    int inv_dim_;
    int param_dim_;
    int n_points_;
    int n_coeffs_;
    Eigen::MatrixXd M_val_;
    Eigen::MatrixXd M_grad_;
    Eigen::MatrixXd M_hess_;

    // Per-call iteration scratch. Deliberately NOT a member of the solver:
    // multistart_optimize() runs many local_optimize() calls concurrently on the
    // same object, so all mutable iteration state must stay call-local to remain
    // reentrant.
    struct IterationWorkspace {
        int max_threads;
        Eigen::VectorXd V_base;
        Eigen::VectorXd dV_base;
        Eigen::VectorXd ddV_base;
        Eigen::VectorXd r_base;
        // Per-thread finite-difference scratch. Allocated lazily by
        // ensure_fd_scratch(): the rank-1 sensitivity path needs none of it, so
        // allocating omp_get_max_threads() copies on every solver call would be
        // pure (and, for large grids, very expensive) dead weight.
        std::vector<Eigen::VectorXd> V;
        std::vector<Eigen::VectorXd> dV;
        std::vector<Eigen::VectorXd> ddV;
        std::vector<Eigen::VectorXd> rhs;
        std::vector<Eigen::VectorXd> r_plus;
        std::vector<Eigen::VectorXd> r_minus;
        Eigen::MatrixXd J;

        // Rank-1 Jacobian scratch: partial derivatives of the scalar flow RHS
        // with respect to (V, dV, ddV) at every collocation point.
        Eigen::VectorXd fV;
        Eigen::VectorXd fdV;
        Eigen::VectorXd fddV;

        // Batched finite-difference scratch (used only when inv_dim == 1).
        // Lazily sized in compute_jacobian_batch().
        RowMajorMatrix V_batch;
        RowMajorMatrix dV_batch;
        RowMajorMatrix ddV_batch;
        RowMajorMatrix rhs_batch;
        Eigen::VectorXd eps_batch;

        IterationWorkspace(int n_points, int inv_dim, int n_coeffs, int threads);

        // Materialises the per-thread finite-difference scratch on first use.
        void ensure_fd_scratch(int n_points, int inv_dim, int threads);
    };

    // Evaluates the base residual r(x) into ws.r_base (thread 0 buffers).
    void evaluate_residual(const Eigen::VectorXd& x,
                           IterationWorkspace& ws,
                           const Eigen::VectorXd& flow_params) const;

    // Assembles the finite-difference Jacobian into ws.J.
    void compute_jacobian(const Eigen::VectorXd& x,
                          IterationWorkspace& ws,
                          const Eigen::VectorXd& flow_params,
                          bool multithread) const;

    // True when the batched prange Jacobian path can be used.
    bool can_use_batch(bool multithread) const;

    // True when the rank-1 (sensitivity) Jacobian path can be used.
    bool can_use_sensitivity(int n_coeffs) const;

    // Rank-1 Jacobian (inv_dim == 1): the pointwise flow sensitivity wrt
    // (V, dV, ddV) turns the Jacobian into three diagonal scalings of the
    // precomputed basis matrices, requiring only 6 flow evaluations per point.
    // The base residual must already have been evaluated into the workspace.
    void compute_jacobian_sensitivity(const Eigen::VectorXd& x,
                                      IterationWorkspace& ws,
                                      const Eigen::VectorXd& flow_params,
                                      bool multithread) const;

    // Batched finite-difference Jacobian (inv_dim == 1): one callback call
    // computes both the +eps and -eps perturbations for all coefficients.
    void compute_jacobian_batch(const Eigen::VectorXd& x,
                                IterationWorkspace& ws,
                                const Eigen::VectorXd& flow_params) const;

    // Solves A * dx = b. Square A uses the rank-revealing FullPivLU,
    // non-square A uses the column-pivoting QR (least squares).
    Eigen::VectorXd solve_linear_system(const Eigen::MatrixXd& A,
                                        const Eigen::VectorXd& b) const;

    // The Newton iteration step: assemble the Jacobian J and solve for dx.
    // Reads only immutable solver state and writes into ws. Returns dx.
    Eigen::VectorXd compute_step(const Eigen::VectorXd& x,
                                 IterationWorkspace& ws,
                                 const Eigen::VectorXd& flow_params,
                                 bool multithread) const;
};

} // namespace lpa
} // namespace frgfp