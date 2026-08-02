#include "_solver.hpp"
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace frgfp {
namespace lpa {

LPACollSolver_cpp::LPACollSolver_cpp(const Eigen::MatrixXd& coll_grid,
                                     std::shared_ptr<core::Ansatz> ansatz,
                                     size_t cfunc_ptr, int inv_dim, int param_dim)
    : coll_grid_(coll_grid), ansatz_(ansatz), inv_dim_(inv_dim), param_dim_(param_dim) 
{
    coll_grid_rm_ = coll_grid; 
    
    n_points_ = coll_grid_.rows();
    flowrhs_c_func_ = reinterpret_cast<FlowRhsFunc>(cfunc_ptr);

    M_val_ = ansatz_->evaluate_basis(coll_grid_);
    int n_coeffs = M_val_.cols();

    if (n_points_ < n_coeffs) {
        throw std::invalid_argument(
            "Hiba: A kollokacios pontok szama (" + std::to_string(n_points_) + 
            ") kisebb, mint az egyutthatok szama (" + std::to_string(n_coeffs) + 
            ")! A rendszer alulhatarozott."
        );
    }

    auto grad_vec = ansatz_->evaluate_basis_grad(coll_grid_);
    M_grad_.resize(n_points_ * inv_dim_, n_coeffs);
    for (int pt = 0; pt < n_points_; ++pt) {
        for (int d = 0; d < inv_dim_; ++d) {
            M_grad_.row(pt * inv_dim_ + d) = grad_vec[d].row(pt);
        }
    }

    auto hess_vec = ansatz_->evaluate_basis_hess(coll_grid_);
    M_hess_.resize(n_points_ * inv_dim_ * inv_dim_, n_coeffs);
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

OptimizeResult LPACollSolver_cpp::local_optimize(const Eigen::VectorXd& init_coeffs,
                                                 int maxiter, double tol,
                                                 const Eigen::VectorXd& flow_params,
                                                 bool multithread) 
{
    Eigen::VectorXd x = init_coeffs;
    int n_coeffs = x.size();
    
    int max_threads = multithread ? omp_get_max_threads() : 1;
    
    Eigen::VectorXd V_base(n_points_);
    Eigen::VectorXd dV_base(n_points_ * inv_dim_);
    Eigen::VectorXd ddV_base(n_points_ * inv_dim_ * inv_dim_);
    Eigen::VectorXd r_base(n_points_);
    
    std::vector<Eigen::VectorXd> V_ws(max_threads, Eigen::VectorXd(n_points_));
    std::vector<Eigen::VectorXd> dV_ws(max_threads, Eigen::VectorXd(n_points_ * inv_dim_));
    std::vector<Eigen::VectorXd> ddV_ws(max_threads, Eigen::VectorXd(n_points_ * inv_dim_ * inv_dim_));
    std::vector<Eigen::VectorXd> rhs_ws(max_threads, Eigen::VectorXd(n_points_));
    std::vector<Eigen::VectorXd> r_plus_ws(max_threads, Eigen::VectorXd(n_points_));
    std::vector<Eigen::VectorXd> r_minus_ws(max_threads, Eigen::VectorXd(n_points_));
    
    Eigen::MatrixXd J(n_points_, n_coeffs);
    
    int iter = 0;
    double err = 1e9;
    
    while (iter < maxiter) {
        V_base.noalias()   = M_val_ * x;
        dV_base.noalias()  = M_grad_ * x;
        ddV_base.noalias() = M_hess_ * x;

        flowrhs_c_func_(
            coll_grid_rm_.data(), V_base.data(), dV_base.data(), ddV_base.data(),
            flow_params.data(), rhs_ws[0].data(), n_points_, inv_dim_
        );
        
        r_base.noalias() = rhs_ws[0]; 
        
        err = r_base.norm();
        if (err <= tol) break;
        
        #pragma omp parallel for if(multithread)
        for (int j = 0; j < n_coeffs; ++j) {
            int tid = multithread ? omp_get_thread_num() : 0;
            double eps = 1e-6 * std::max(1.0, std::abs(x(j))); 

            V_ws[tid].noalias()   = V_base + eps * M_val_.col(j);
            dV_ws[tid].noalias()  = dV_base + eps * M_grad_.col(j);
            ddV_ws[tid].noalias() = ddV_base + eps * M_hess_.col(j);

            flowrhs_c_func_(
                coll_grid_rm_.data(), V_ws[tid].data(), dV_ws[tid].data(), ddV_ws[tid].data(),
                flow_params.data(), rhs_ws[tid].data(), n_points_, inv_dim_
            );
            
            r_plus_ws[tid].noalias() = rhs_ws[tid];

            V_ws[tid].noalias()   = V_base - eps * M_val_.col(j);
            dV_ws[tid].noalias()  = dV_base - eps * M_grad_.col(j);
            ddV_ws[tid].noalias() = ddV_base - eps * M_hess_.col(j);

            flowrhs_c_func_(
                coll_grid_rm_.data(), V_ws[tid].data(), dV_ws[tid].data(), ddV_ws[tid].data(),
                flow_params.data(), rhs_ws[tid].data(), n_points_, inv_dim_
            );
            
            r_minus_ws[tid].noalias() = rhs_ws[tid];

            J.col(j).noalias() = (r_plus_ws[tid] - r_minus_ws[tid]) / (2.0 * eps);
        }
        
        Eigen::VectorXd dx = J.bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>().solve(-r_base);
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
    int n_starts = init_coeffs_list.size();
    std::vector<OptimizeResult> results(n_starts);

    #pragma omp parallel for
    for (int i = 0; i < n_starts; ++i) {
        // Nested threading kikapcsolva a külső párhuzamosításhoz
        results[i] = local_optimize(init_coeffs_list[i], maxiter, tol, flow_params, false);
    }
    
    return results;
}

} // namespace lpa
} // namespace frgfp