#include "_func.hpp"
#include <stdexcept>

namespace frgfp {
namespace core {

FuncFromAnsatz::FuncFromAnsatz(std::shared_ptr<Ansatz> ansatz, const Eigen::VectorXd& coeffs)
    : m_ansatz(ansatz), m_coeffs(coeffs) {
    if (!ansatz) {
        throw std::invalid_argument("Ansatz object cannot be null.");
    }
    if (coeffs.size() != ansatz->get_num_coeffs()) {
        throw std::invalid_argument("Coefficient vector size must exactly match the number of basis functions in the ansatz.");
    }
}

void FuncFromAnsatz::set_coeffs(const Eigen::VectorXd& coeffs) {
    if (coeffs.size() != m_ansatz->get_num_coeffs()) {
        throw std::invalid_argument("Coefficient vector size must exactly match the number of basis functions in the ansatz.");
    }
    m_coeffs = coeffs;
}

Eigen::VectorXd FuncFromAnsatz::evaluate(const Eigen::MatrixXd& grid) const {
    return m_ansatz->evaluate_basis(grid) * m_coeffs;
}

std::vector<Eigen::VectorXd> FuncFromAnsatz::evaluate_grad(const Eigen::MatrixXd& grid) const {
    auto grads_mats = m_ansatz->evaluate_basis_grad(grid);
    std::vector<Eigen::VectorXd> result(grads_mats.size());
    for (size_t i = 0; i < grads_mats.size(); ++i) {
        result[i] = grads_mats[i] * m_coeffs;
    }
    return result;
}

std::vector<Eigen::VectorXd> FuncFromAnsatz::evaluate_hess(const Eigen::MatrixXd& grid) const {
    auto hess_mats = m_ansatz->evaluate_basis_hess(grid);
    std::vector<Eigen::VectorXd> result(hess_mats.size());
    for (size_t i = 0; i < hess_mats.size(); ++i) {
        result[i] = hess_mats[i] * m_coeffs;
    }
    return result;
}

Eigen::VectorXd fit_coeffs_from_grid(std::shared_ptr<Ansatz> ansatz, const Eigen::MatrixXd& grid, const Eigen::VectorXd& vals) {
    Eigen::MatrixXd M = ansatz->evaluate_basis(grid);
    
    return M.bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>().solve(vals);
}

} // namespace core
} // namespace frgfp