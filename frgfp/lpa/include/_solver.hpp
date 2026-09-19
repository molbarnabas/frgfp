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
typedef void (*BatchRhsFunc)(
    const double* I, const double* V, const double* dV, const double* ddV,
    const double* params, double* rhs_out, int n_points, int n_cols, int inv_dim
);

// Sensitivity variant: evaluates the pointwise flow partials (V, dV, ddV) at every point.
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

    // Performs exactly num_iter steps; the final residual norm is in OptimizeResult::error.
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

    // Per-call iteration scratch: kept call-local so concurrent multistart calls stay reentrant.
    struct IterationWorkspace {
        int max_threads;
        Eigen::VectorXd V_base;
        Eigen::VectorXd dV_base;
        Eigen::VectorXd ddV_base;
        Eigen::VectorXd r_base;
        // Per-thread finite-difference scratch, lazily allocated by ensure_fd_scratch().
        std::vector<Eigen::VectorXd> V;
        std::vector<Eigen::VectorXd> dV;
        std::vector<Eigen::VectorXd> ddV;
        std::vector<Eigen::VectorXd> rhs;
        std::vector<Eigen::VectorXd> r_plus;
        std::vector<Eigen::VectorXd> r_minus;
        Eigen::MatrixXd J;

        // Rank-1 Jacobian scratch: flow partials wrt (V, dV, ddV) per point.
        Eigen::VectorXd fV;
        Eigen::VectorXd fdV;
        Eigen::VectorXd fddV;

        // Batched finite-difference scratch (inv_dim == 1), lazily sized in compute_jacobian_batch().
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

    // Rank-1 Jacobian via diagonal scalings of the basis matrices; base residual must be current.
    void compute_jacobian_sensitivity(const Eigen::VectorXd& x,
                                      IterationWorkspace& ws,
                                      const Eigen::VectorXd& flow_params,
                                      bool multithread) const;

    // Batched finite-difference Jacobian (inv_dim == 1): one callback for all +/-eps columns.
    void compute_jacobian_batch(const Eigen::VectorXd& x,
                                IterationWorkspace& ws,
                                const Eigen::VectorXd& flow_params) const;

    // Solves A * dx = b via FullPivLU (square) or column-pivoting QR (least squares).
    Eigen::VectorXd solve_linear_system(const Eigen::MatrixXd& A,
                                        const Eigen::VectorXd& b) const;

    // The Newton iteration step: assemble J, solve for dx and return it.
    Eigen::VectorXd compute_step(const Eigen::VectorXd& x,
                                 IterationWorkspace& ws,
                                 const Eigen::VectorXd& flow_params,
                                 bool multithread) const;
};

} // namespace lpa
} // namespace frgfp