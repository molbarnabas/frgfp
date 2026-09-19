#pragma once
#include <Eigen/Dense>
#include <vector>

namespace frgfp {
namespace core {

class Ansatz {
protected:
    int m_inv_dim;
    int m_num_coeffs;

public:
    Ansatz(int inv_dim, int num_coeffs) : m_inv_dim(inv_dim), m_num_coeffs(num_coeffs) {}
    virtual ~Ansatz() = default;

    int get_inv_dim() const { return m_inv_dim; }
    int get_num_coeffs() const { return m_num_coeffs; }

    virtual Eigen::MatrixXd evaluate_basis(const Eigen::MatrixXd& grid) const = 0;
    virtual std::vector<Eigen::MatrixXd> evaluate_basis_grad(const Eigen::MatrixXd& grid) const = 0;
    virtual std::vector<Eigen::MatrixXd> evaluate_basis_hess(const Eigen::MatrixXd& grid) const = 0;

    // Fused basis/gradient/Hessian evaluation from a single set of 1D caches.
    virtual void evaluate_all(const Eigen::MatrixXd& grid,
                              Eigen::MatrixXd& M,
                              std::vector<Eigen::MatrixXd>& grads,
                              std::vector<Eigen::MatrixXd>& hesss) const;
};

class ChebyshevAnsatz : public Ansatz {
private:
    Eigen::VectorXi m_order;
    Eigen::MatrixXd m_interval;

public:
    ChebyshevAnsatz(int inv_dim, const Eigen::VectorXi& order, const Eigen::MatrixXd& interval);
    Eigen::MatrixXd evaluate_basis(const Eigen::MatrixXd& grid) const override;
    std::vector<Eigen::MatrixXd> evaluate_basis_grad(const Eigen::MatrixXd& grid) const override;
    std::vector<Eigen::MatrixXd> evaluate_basis_hess(const Eigen::MatrixXd& grid) const override;
    void evaluate_all(const Eigen::MatrixXd& grid,
                      Eigen::MatrixXd& M,
                      std::vector<Eigen::MatrixXd>& grads,
                      std::vector<Eigen::MatrixXd>& hesss) const override;
};

class PolyAnsatz : public Ansatz {
private:
    Eigen::VectorXi m_order;
    Eigen::VectorXd m_center;

public:
    PolyAnsatz(int inv_dim, const Eigen::VectorXi& order, const Eigen::VectorXd& center);
    Eigen::MatrixXd evaluate_basis(const Eigen::MatrixXd& grid) const override;
    std::vector<Eigen::MatrixXd> evaluate_basis_grad(const Eigen::MatrixXd& grid) const override;
    std::vector<Eigen::MatrixXd> evaluate_basis_hess(const Eigen::MatrixXd& grid) const override;
    void evaluate_all(const Eigen::MatrixXd& grid,
                      Eigen::MatrixXd& M,
                      std::vector<Eigen::MatrixXd>& grads,
                      std::vector<Eigen::MatrixXd>& hesss) const override;
};

} // namespace core
} // namespace frgfp