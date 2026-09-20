#include "_solver.hpp"
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace {

// Minimum n_coeffs before the scalar finite-difference loop is parallelized.
constexpr int kParallelThreshold = 256;
// Minimum n_coeffs before the batched Jacobian path is used.
constexpr int kBatchThreshold = 64;
// Minimum n_coeffs before the rank-1 (sensitivity) Jacobian path is used.
constexpr int kSensitivityThreshold = 16;
// Minimum collocation points before the parallel sensitivity kernel pays off.
constexpr int kSensitivityParallelMinPoints = 512;
// Upper bound on n_points * 2*n_coeffs for the batched path (caps peak memory).
constexpr long long kBatchMaxElements = 500000;

// Ties Eigen's global (GEMM) thread count to `multithread` outside parallel regions.
struct EigenThreadGuard {
    int prev;
    explicit EigenThreadGuard(bool multithread) {
        if (omp_in_parallel()) { prev = -1; return; }
        prev = Eigen::nbThreads();
        Eigen::setNbThreads(multithread ? 0 : 1);
    }
    ~EigenThreadGuard() {
        if (prev >= 0) Eigen::setNbThreads(prev);
    }
    EigenThreadGuard(const EigenThreadGuard&) = delete;
    EigenThreadGuard& operator=(const EigenThreadGuard&) = delete;
};

} // anonymous namespace

namespace frgfp {
namespace lpa {

LPACollSolver_cpp::LPACollSolver_cpp(const Eigen::MatrixXd& coll_grid,
                                     std::shared_ptr<core::Ansatz> ansatz,
                                     size_t cfunc_ptr, size_t batch_cfunc_ptr,
                                     size_t sens_cfunc_ptr, size_t sens_par_cfunc_ptr,
                                     int inv_dim, int param_dim)
    : coll_grid_(coll_grid), ansatz_(ansatz), inv_dim_(inv_dim), param_dim_(param_dim) 
{
    coll_grid_rm_ = coll_grid; 
    
    n_points_ = coll_grid_.rows();
    flowrhs_c_func_ = reinterpret_cast<FlowRhsFunc>(cfunc_ptr);
    flowrhs_batch_c_func_ = reinterpret_cast<BatchRhsFunc>(batch_cfunc_ptr);
    flowrhs_sens_c_func_ = reinterpret_cast<SensRhsFunc>(sens_cfunc_ptr);
    flowrhs_sens_par_c_func_ = reinterpret_cast<SensRhsFunc>(sens_par_cfunc_ptr);

    n_coeffs_ = ansatz_->get_num_coeffs();

    if (n_points_ < n_coeffs_) {
        throw std::invalid_argument(
            "The number of collocation points (" + std::to_string(n_points_) + 
            ") is smaller than the number of coefficients (" + std::to_string(n_coeffs_) + 
            ")! The system is underdetermined."
        );
    }

    // Fused evaluation of basis, gradient and Hessian from a single set of 1D caches.
    std::vector<Eigen::MatrixXd> grad_vec;
    std::vector<Eigen::MatrixXd> hess_vec;
    ansatz_->evaluate_all(coll_grid_, M_val_, grad_vec, hess_vec);

    M_grad_.resize(n_points_ * inv_dim_, n_coeffs_);
    for (int pt = 0; pt < n_points_; ++pt) {
        for (int d = 0; d < inv_dim_; ++d) {
            M_grad_.row(pt * inv_dim_ + d) = grad_vec[d].row(pt);
        }
    }

    M_hess_.resize(n_points_ * inv_dim_ * inv_dim_, n_coeffs_);
    for (int pt = 0; pt < n_points_; ++pt) {
        for (int d1 = 0; d1 < inv_dim_; ++d1) {
            for (int d2 = 0; d2 < inv_dim_; ++d2) {
                int flat_d = d1 * inv_dim_ + d2;
                int row_idx = pt * (inv_dim_ * inv_dim_) + flat_d;
                M_hess_.row(row_idx) = hess_vec[flat_d].row(pt);
            }
        }
    }
}

LPACollSolver_cpp::IterationWorkspace::IterationWorkspace(int n_points, int inv_dim, int n_coeffs, int threads)
    : max_threads(threads),
      V_base(n_points),
      dV_base(n_points * inv_dim),
      ddV_base(n_points * inv_dim * inv_dim),
      r_base(n_points),
      J(n_points, n_coeffs),
      fV(n_points),
      fdV(n_points * inv_dim),
      fddV(n_points * inv_dim * inv_dim) {}

void LPACollSolver_cpp::IterationWorkspace::ensure_fd_scratch(int n_points, int inv_dim, int threads)
{
    if (static_cast<int>(V.size()) >= threads) return;
    V.assign(threads, Eigen::VectorXd(n_points));
    dV.assign(threads, Eigen::VectorXd(n_points * inv_dim));
    ddV.assign(threads, Eigen::VectorXd(n_points * inv_dim * inv_dim));
    rhs.assign(threads, Eigen::VectorXd(n_points));
    r_plus.assign(threads, Eigen::VectorXd(n_points));
    r_minus.assign(threads, Eigen::VectorXd(n_points));
}

void LPACollSolver_cpp::evaluate_residual(const Eigen::VectorXd& x,
                                          IterationWorkspace& ws,
                                          const Eigen::VectorXd& flow_params) const
{
    ws.V_base.noalias()   = M_val_ * x;
    ws.dV_base.noalias()  = M_grad_ * x;
    ws.ddV_base.noalias() = M_hess_ * x;

    // The flow callback writes the base residual directly into r_base.
    flowrhs_c_func_(
        coll_grid_rm_.data(), ws.V_base.data(), ws.dV_base.data(), ws.ddV_base.data(),
        flow_params.data(), ws.r_base.data(), n_points_, inv_dim_
    );
}

void LPACollSolver_cpp::compute_jacobian(const Eigen::VectorXd& x,
                                         IterationWorkspace& ws,
                                         const Eigen::VectorXd& flow_params,
                                         bool multithread) const
{
    if (can_use_sensitivity(x.size())) {
        compute_jacobian_sensitivity(x, ws, flow_params, multithread);
        return;
    }

    if (can_use_batch(multithread)) {
        compute_jacobian_batch(x, ws, flow_params);
        return;
    }

    int n_coeffs = x.size();
    bool run_parallel = multithread && (n_coeffs >= kParallelThreshold);

    // The scalar finite-difference loop is the only consumer of the per-thread scratch.
    ws.ensure_fd_scratch(n_points_, inv_dim_, run_parallel ? ws.max_threads : 1);

    #pragma omp parallel for if(run_parallel)
    for (int j = 0; j < n_coeffs; ++j) {
        int tid = run_parallel ? omp_get_thread_num() : 0;
        double eps = 1e-6 * std::max(1.0, std::abs(x(j)));

        ws.V[tid].noalias()   = ws.V_base + eps * M_val_.col(j);
        ws.dV[tid].noalias()  = ws.dV_base + eps * M_grad_.col(j);
        ws.ddV[tid].noalias() = ws.ddV_base + eps * M_hess_.col(j);

        flowrhs_c_func_(
            coll_grid_rm_.data(), ws.V[tid].data(), ws.dV[tid].data(), ws.ddV[tid].data(),
            flow_params.data(), ws.rhs[tid].data(), n_points_, inv_dim_
        );

        ws.r_plus[tid].noalias() = ws.rhs[tid];

        ws.V[tid].noalias()   = ws.V_base - eps * M_val_.col(j);
        ws.dV[tid].noalias()  = ws.dV_base - eps * M_grad_.col(j);
        ws.ddV[tid].noalias() = ws.ddV_base - eps * M_hess_.col(j);

        flowrhs_c_func_(
            coll_grid_rm_.data(), ws.V[tid].data(), ws.dV[tid].data(), ws.ddV[tid].data(),
            flow_params.data(), ws.rhs[tid].data(), n_points_, inv_dim_
        );

        ws.r_minus[tid].noalias() = ws.rhs[tid];

        ws.J.col(j).noalias() = (ws.r_plus[tid] - ws.r_minus[tid]) / (2.0 * eps);
    }
}

bool LPACollSolver_cpp::can_use_batch(bool multithread) const
{
    return multithread
        && flowrhs_batch_c_func_ != nullptr
        && inv_dim_ == 1
        && n_coeffs_ >= kBatchThreshold
        && static_cast<long long>(n_points_) * (2LL * n_coeffs_) <= kBatchMaxElements
        && !omp_in_parallel();
}

bool LPACollSolver_cpp::can_use_sensitivity(int n_coeffs) const
{
    // Works for any invariant dimension: sums the inv_dim and inv_dim^2 basis-matrix blocks.
    return (flowrhs_sens_c_func_ != nullptr || flowrhs_sens_par_c_func_ != nullptr)
        && n_coeffs >= kSensitivityThreshold;
}

void LPACollSolver_cpp::compute_jacobian_sensitivity(const Eigen::VectorXd& x,
                                                     IterationWorkspace& ws,
                                                     const Eigen::VectorXd& flow_params,
                                                     bool multithread) const
{
    (void)x;  // V_base/dV_base/ddV_base already reflect x (caller contract).

    // Pick the parallel kernel only when it is wanted, large enough and not nested.
    SensRhsFunc probe;
    if (multithread && !omp_in_parallel() && n_points_ >= kSensitivityParallelMinPoints
        && flowrhs_sens_par_c_func_ != nullptr) {
        probe = flowrhs_sens_par_c_func_;
    } else if (flowrhs_sens_c_func_ != nullptr) {
        probe = flowrhs_sens_c_func_;
    } else {
        probe = flowrhs_sens_par_c_func_;
    }

    probe(
        coll_grid_rm_.data(), ws.V_base.data(), ws.dV_base.data(), ws.ddV_base.data(),
        flow_params.data(), ws.fV.data(), ws.fdV.data(), ws.fddV.data(), n_points_
    );

    // J = diag(fV)*M_val + sum_a diag(fdV_a)*M_grad_a + sum_{a,b} diag(fddV_ab)*M_hess_ab, assembled serially (bandwidth-bound).
    const int n = n_points_;
    const int m = n_coeffs_;
    const int d = inv_dim_;
    const int dd = d * d;
    if (d == 1) {
        // Flat fast path: a single invariant gives all three matrices the same shape.
        for (int j = 0; j < m; ++j) {
            for (int i = 0; i < n; ++i) {
                ws.J(i, j) = ws.fV(i) * M_val_(i, j)
                           + ws.fdV(i) * M_grad_(i, j)
                           + ws.fddV(i) * M_hess_(i, j);
            }
        }
        return;
    }

    // General path: accumulate the inv_dim gradient and inv_dim^2 Hessian blocks per point.
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < n; ++i) {
            double val = ws.fV(i) * M_val_(i, j);

            const double* f_dv = &ws.fdV(i * d);
            const double* grad_row = &M_grad_(i * d, j);
            for (int a = 0; a < d; ++a) {
                val += f_dv[a] * grad_row[a];
            }

            const double* f_ddv = &ws.fddV(i * dd);
            const double* hess_row = &M_hess_(i * dd, j);
            for (int k = 0; k < dd; ++k) {
                val += f_ddv[k] * hess_row[k];
            }

            ws.J(i, j) = val;
        }
    }
}

void LPACollSolver_cpp::compute_jacobian_batch(const Eigen::VectorXd& x,
                                               IterationWorkspace& ws,
                                               const Eigen::VectorXd& flow_params) const
{
    const int n_coeffs = x.size();
    const int two_n = 2 * n_coeffs;

    ws.eps_batch.resize(n_coeffs);
    for (int j = 0; j < n_coeffs; ++j) {
        ws.eps_batch(j) = 1e-6 * std::max(1.0, std::abs(x(j)));
    }

    // Columns [0, n) hold the +eps perturbations, columns [n, 2n) the -eps ones.
    ws.V_batch.resize(n_points_, two_n);
    ws.dV_batch.resize(n_points_, two_n);
    ws.ddV_batch.resize(n_points_, two_n);

    for (int j = 0; j < n_coeffs; ++j) {
        double eps = ws.eps_batch(j);
        ws.V_batch.col(j)             = ws.V_base + eps * M_val_.col(j);
        ws.V_batch.col(n_coeffs + j)  = ws.V_base - eps * M_val_.col(j);
        ws.dV_batch.col(j)            = ws.dV_base + eps * M_grad_.col(j);
        ws.dV_batch.col(n_coeffs + j) = ws.dV_base - eps * M_grad_.col(j);
        ws.ddV_batch.col(j)             = ws.ddV_base + eps * M_hess_.col(j);
        ws.ddV_batch.col(n_coeffs + j)  = ws.ddV_base - eps * M_hess_.col(j);
    }

    // One batched callback call (Numba prange) evaluates all 2*n_coeffs columns.
    ws.rhs_batch.resize(n_points_, two_n);
    flowrhs_batch_c_func_(
        coll_grid_rm_.data(), ws.V_batch.data(), ws.dV_batch.data(), ws.ddV_batch.data(),
        flow_params.data(), ws.rhs_batch.data(), n_points_, two_n, inv_dim_
    );

    for (int j = 0; j < n_coeffs; ++j) {
        ws.J.col(j).noalias() =
            (ws.rhs_batch.col(j) - ws.rhs_batch.col(n_coeffs + j)) / (2.0 * ws.eps_batch(j));
    }
}

Eigen::VectorXd LPACollSolver_cpp::solve_linear_system(const Eigen::MatrixXd& A,
                                                       const Eigen::VectorXd& b) const
{
    if (A.rows() == A.cols()) {
        // Full pivoting's rank detection regularises exactly rank-deficient systems.
        return A.fullPivLu().solve(b);
    }
    return A.colPivHouseholderQr().solve(b);
}

Eigen::VectorXd LPACollSolver_cpp::compute_step(const Eigen::VectorXd& x,
                                                IterationWorkspace& ws,
                                                const Eigen::VectorXd& flow_params,
                                                bool multithread) const
{
    compute_jacobian(x, ws, flow_params, multithread);
    return solve_linear_system(ws.J, -ws.r_base);
}

OptimizeResult LPACollSolver_cpp::local_optimize(const Eigen::VectorXd& init_coeffs,
                                                 int maxiter, double tol,
                                                 const Eigen::VectorXd& flow_params,
                                                 bool multithread) 
{
    EigenThreadGuard eigen_guard(multithread);

    Eigen::VectorXd x = init_coeffs;
    
    int max_threads = multithread ? omp_get_max_threads() : 1;
    IterationWorkspace ws(n_points_, inv_dim_, x.size(), max_threads);
    
    int iter = 0;
    double err = 1e9;
    
    while (iter < maxiter) {
        evaluate_residual(x, ws, flow_params);
        
        err = ws.r_base.norm();
        if (err <= tol) break;
        
        Eigen::VectorXd dx = compute_step(x, ws, flow_params, multithread);
        x += dx;
        iter++;
    }
    
    OptimizeResult result;
    result.coeffs = x;
    result.success = (err <= tol);
    result.iterations = iter;
    result.error = err;
    return result;
}

OptimizeResult LPACollSolver_cpp::local_optimize_fixed_iterations(
    const Eigen::VectorXd& init_coeffs,
    int num_iter,
    const Eigen::VectorXd& flow_params,
    bool multithread)
{
    EigenThreadGuard eigen_guard(multithread);

    Eigen::VectorXd x = init_coeffs;

    int max_threads = multithread ? omp_get_max_threads() : 1;
    IterationWorkspace ws(n_points_, inv_dim_, x.size(), max_threads);

    int iter = 0;
    double err = 0.0;

    // No tolerance-based stop: perform exactly num_iter iteration steps.
    while (iter < num_iter) {
        evaluate_residual(x, ws, flow_params);
        err = ws.r_base.norm();

        Eigen::VectorXd dx = compute_step(x, ws, flow_params, multithread);
        x += dx;
        iter++;
    }

    // Report the residual norm of the final iterate.
    evaluate_residual(x, ws, flow_params);
    err = ws.r_base.norm();

    OptimizeResult result;
    result.coeffs = x;
    result.success = std::isfinite(err);
    result.iterations = iter;
    result.error = err;
    return result;
}

std::vector<OptimizeResult> LPACollSolver_cpp::param_path_following(
    const Eigen::VectorXd& init_coeffs,
    int maxiter, double tol,
    const std::vector<Eigen::VectorXd>& flow_params_path,
    bool multithread) 
{
    std::vector<OptimizeResult> results;
    results.reserve(flow_params_path.size());
    
    Eigen::VectorXd current_coeffs = init_coeffs;
    
    for (const auto& params : flow_params_path) {
        auto res = local_optimize(current_coeffs, maxiter, tol, params, multithread);
        if (res.success) {
            current_coeffs = res.coeffs; 
        }
        results.push_back(res);
    }
    
    return results;
}

std::vector<OptimizeResult> LPACollSolver_cpp::multistart_optimize(
    const std::vector<Eigen::VectorXd>& init_coeffs_list,
    int maxiter, double tol,
    const Eigen::VectorXd& flow_params) 
{
    // The inner local_optimize() calls run in parallel and are single-threaded.
    EigenThreadGuard eigen_guard(false);

    int n_starts = init_coeffs_list.size();
    std::vector<OptimizeResult> results(n_starts);

    #pragma omp parallel for
    for (int i = 0; i < n_starts; ++i) {
        // Nested threading disabled for the outer parallelization.
        results[i] = local_optimize(init_coeffs_list[i], maxiter, tol, flow_params, false);
    }
    
    return results;
}

} // namespace lpa
} // namespace frgfp