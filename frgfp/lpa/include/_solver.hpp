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

class LPACollSolver_cpp {
public:
    LPACollSolver_cpp(const Eigen::MatrixXd& coll_grid,
                      std::shared_ptr<core::Ansatz> ansatz,
                      size_t cfunc_ptr, int inv_dim, int param_dim);

    OptimizeResult local_optimize(const Eigen::VectorXd& init_coeffs,
                                  int maxiter, double tol,
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
    Eigen::MatrixXd coll_grid_;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> coll_grid_rm_;
    std::shared_ptr<core::Ansatz> ansatz_;
    FlowRhsFunc flowrhs_c_func_;
    int inv_dim_;
    int param_dim_;
    int n_points_;
    Eigen::MatrixXd M_val_;
    Eigen::MatrixXd M_grad_;
    Eigen::MatrixXd M_hess_;
};

} // namespace lpa
} // namespace frgfp