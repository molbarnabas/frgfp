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
// Upper bound on n_points * 2*n_coeffs for the batched path. Each workspace
// holds four such row-major matrices, and multistart keeps up to
// omp_get_max_threads() workspaces alive at once, so this caps peak memory.
constexpr long long kBatchMaxElements = 500000;

// Ties Eigen's internal (GEMM) thread count to the caller's `multithread`
// flag: 1 thread when disabled, the OpenMP maximum (0) when enabled.
// Eigen's setting is a single global, so it must never be touched from inside
// an OpenMP parallel region (e.g. multistart's concurrent inner solves).
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
                                     int inv_dim, int param_dim)
    : coll_grid_(coll_grid), ansatz_(ansatz), inv_dim_(inv_dim), param_dim_(param_dim) 
{
    coll_grid_rm_ = coll_grid; 
    
    n_points_ = coll_grid_.rows();
    flowrhs_c_func_ = reinterpret_cast<FlowRhsFunc>(cfunc_ptr);
    flowrhs_batch_c_func_ = reinterpret_cast<BatchRhsFunc>(batch_cfunc_ptr);

    M_val_ = ansatz_->evaluate_basis(coll_grid_);
    n_coeffs_ = M_val_.cols();

    if (n_points_ < n_coeffs_) {
        throw std::invalid_argument(
            "The number of collocation points (" + std::to_string(n_points_) + 
            ") is smaller than the number of coefficients (" + std::to_string(n_coeffs_) + 
            ")! The system is underdetermined."
        );
    }

    auto grad_vec = ansatz_->evaluate_basis_grad(coll_grid_);
    M_grad_.resize(n_points_ * inv_dim_, n_coeffs_);
    for (int pt = 0; pt < n_points_; ++pt) {
        for (int d = 0; d < inv_dim_; ++d) {
            M_grad_.row(pt * inv_dim_ + d) = grad_vec[d].row(pt);
        }
    }

    auto hess_vec = ansatz_->evaluate_basis_hess(coll_grid_);
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
      V(threads, Eigen::VectorXd(n_points)),
      dV(threads, Eigen::VectorXd(n_points * inv_dim)),
      ddV(threads, Eigen::VectorXd(n_points * inv_dim * inv_dim)),
      rhs(threads, Eigen::VectorXd(n_points)),
      r_plus(threads, Eigen::VectorXd(n_points)),
      r_minus(threads, Eigen::VectorXd(n_points)),
      J(n_points, n_coeffs) {}

void LPACollSolver_cpp::evaluate_residual(const Eigen::VectorXd& x,
                                          IterationWorkspace& ws,
                                          const Eigen::VectorXd& flow_params) const
{
    ws.V_base.noalias()   = M_val_ * x;
    ws.dV_base.noalias()  = M_grad_ * x;
    ws.ddV_base.noalias() = M_hess_ * x;

    flowrhs_c_func_(
        coll_grid_rm_.data(), ws.V_base.data(), ws.dV_base.data(), ws.ddV_base.data(),
        flow_params.data(), ws.rhs[0].data(), n_points_, inv_dim_
    );

    ws.r_base.noalias() = ws.rhs[0];
}

void LPACollSolver_cpp::compute_jacobian(const Eigen::VectorXd& x,
                                         IterationWorkspace& ws,
                                         const Eigen::VectorXd& flow_params,
                                         bool multithread) const
{
    if (can_use_batch(multithread)) {
        compute_jacobian_batch(x, ws, flow_params);
        return;
    }

    int n_coeffs = x.size();
    bool run_parallel = multithread && (n_coeffs >= kParallelThreshold);

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