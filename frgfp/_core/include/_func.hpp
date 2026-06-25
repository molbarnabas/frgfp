#pragma once
#include <Eigen/Dense>
#include <memory>
#include <vector>
#include "_ansatz.hpp"

namespace frgfp {
namespace core {

class FuncFromAnsatz {
private:
    std::shared_ptr<Ansatz> m_ansatz;
    Eigen::VectorXd m_coeffs;

public:
    FuncFromAnsatz(std::shared_ptr<Ansatz> ansatz, const Eigen::VectorXd& coeffs);

    std::shared_ptr<Ansatz> get_ansatz() const { return m_ansatz; }
    Eigen::VectorXd get_coeffs() const { return m_coeffs; }
    void set_coeffs(const Eigen::VectorXd& coeffs);
    int get_num_coeffs() const { return m_ansatz->get_num_coeffs(); }

    Eigen::VectorXd evaluate(const Eigen::MatrixXd& grid) const;
    std::vector<Eigen::VectorXd> evaluate_grad(const Eigen::MatrixXd& grid) const;
    std::vector<Eigen::VectorXd> evaluate_hess(const Eigen::MatrixXd& grid) const;
};

Eigen::VectorXd fit_coeffs_from_grid(std::shared_ptr<Ansatz> ansatz, const Eigen::MatrixXd& grid, const Eigen::VectorXd& vals);

} // namespace core
} // namespace frgfp