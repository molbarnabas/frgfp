#include "_ansatz.hpp"
#include <vector>

namespace {
    struct Caches1D {
        std::vector<Eigen::MatrixXd> v0;
        std::vector<Eigen::MatrixXd> v1;
        std::vector<Eigen::MatrixXd> v2;
    };

    Caches1D build_poly_caches(int inv_dim, const Eigen::VectorXi& order, const Eigen::VectorXd& center, const Eigen::MatrixXd& grid) {
        int n_points = grid.rows();
        Caches1D c;
        c.v0.resize(inv_dim); c.v1.resize(inv_dim); c.v2.resize(inv_dim);
        
        for (int d = 0; d < inv_dim; ++d) {
            int ord = order(d);
            c.v0[d] = Eigen::MatrixXd::Zero(n_points, ord + 1);
            c.v1[d] = Eigen::MatrixXd::Zero(n_points, ord + 1);
            c.v2[d] = Eigen::MatrixXd::Zero(n_points, ord + 1);
            
            for (int p = 0; p < n_points; ++p) {
                double x_c = grid(p, d) - center(d);
                c.v0[d](p, 0) = 1.0;
                
                if (ord >= 1) {
                    c.v0[d](p, 1) = x_c;
                    c.v1[d](p, 1) = 1.0;
                }
                if (ord >= 2) {
                    c.v0[d](p, 2) = x_c * x_c;
                    c.v1[d](p, 2) = 2.0 * x_c;
                    c.v2[d](p, 2) = 2.0;
                }
                for (int k = 3; k <= ord; ++k) {
                    c.v0[d](p, k) = c.v0[d](p, k - 1) * x_c;
                    c.v1[d](p, k) = k * c.v0[d](p, k - 1);
                    c.v2[d](p, k) = k * (k - 1) * c.v0[d](p, k - 2);
                }
            }
        }
        return c;
    }

    Caches1D build_cheb_caches(int inv_dim, const Eigen::VectorXi& order, const Eigen::MatrixXd& interval, const Eigen::MatrixXd& grid) {
        int n_points = grid.rows();
        Caches1D c;
        c.v0.resize(inv_dim); c.v1.resize(inv_dim); c.v2.resize(inv_dim);
        
        for (int d = 0; d < inv_dim; ++d) {
            int ord = order(d);
            c.v0[d] = Eigen::MatrixXd::Zero(n_points, ord + 1);
            c.v1[d] = Eigen::MatrixXd::Zero(n_points, ord + 1);
            c.v2[d] = Eigen::MatrixXd::Zero(n_points, ord + 1);
            
            double min_val = interval(d, 0);
            double max_val = interval(d, 1);
            double scale = 2.0 / (max_val - min_val);

            for (int p = 0; p < n_points; ++p) {
                double y = scale * (grid(p, d) - min_val) - 1.0;

                c.v0[d](p, 0) = 1.0;
                if (ord >= 1) {
                    c.v0[d](p, 1) = y;
                    c.v1[d](p, 1) = scale;
                }
                for (int k = 2; k <= ord; ++k) {
                    c.v0[d](p, k) = 2.0 * y * c.v0[d](p, k - 1) - c.v0[d](p, k - 2);
                    c.v1[d](p, k) = 2.0 * scale * c.v0[d](p, k - 1) + 2.0 * y * c.v1[d](p, k - 1) - c.v1[d](p, k - 2);
                    c.v2[d](p, k) = 4.0 * scale * c.v1[d](p, k - 1) + 2.0 * y * c.v2[d](p, k - 1) - c.v2[d](p, k - 2);
                }
            }
        }
        return c;
    }
}

namespace frgfp {
namespace core {

// --- ChebyshevAnsatz ---

ChebyshevAnsatz::ChebyshevAnsatz(int inv_dim, const Eigen::VectorXi& order, const Eigen::MatrixXd& interval)
    : Ansatz(inv_dim, 0), m_order(order), m_interval(interval) {
    int total = 1;
    for (int i = 0; i < m_order.size(); ++i) total *= (m_order(i) + 1);
    this->m_num_coeffs = total;
}

Eigen::MatrixXd ChebyshevAnsatz::evaluate_basis(const Eigen::MatrixXd& grid) const {
    int n_points = grid.rows();
    Caches1D caches = build_cheb_caches(m_inv_dim, m_order, m_interval, grid);
    Eigen::MatrixXd M = Eigen::MatrixXd::Ones(n_points, m_num_coeffs);

    for (int p = 0; p < n_points; ++p) {
        for (int i = 0; i < m_num_coeffs; ++i) {
            int current_flat = i;
            double val = 1.0;
            for (int d = m_inv_dim - 1; d >= 0; --d) {
                int i_d = current_flat % (m_order(d) + 1);
                current_flat /= (m_order(d) + 1);
                val *= caches.v0[d](p, i_d);
            }
            M(p, i) = val;
        }
    }
    return M;
}

std::vector<Eigen::MatrixXd> ChebyshevAnsatz::evaluate_basis_grad(const Eigen::MatrixXd& grid) const {
    int n_points = grid.rows();
    Caches1D caches = build_cheb_caches(m_inv_dim, m_order, m_interval, grid);
    std::vector<Eigen::MatrixXd> grads(m_inv_dim, Eigen::MatrixXd::Zero(n_points, m_num_coeffs));

    for (int p = 0; p < n_points; ++p) {
        for (int i = 0; i < m_num_coeffs; ++i) {
            std::vector<int> multi_idx(m_inv_dim);
            int current_flat = i;
            for (int d = m_inv_dim - 1; d >= 0; --d) {
                multi_idx[d] = current_flat % (m_order(d) + 1);
                current_flat /= (m_order(d) + 1);
            }

            for (int target_d = 0; target_d < m_inv_dim; ++target_d) {
                double val = 1.0;
                for (int d = 0; d < m_inv_dim; ++d) {
                    if (d == target_d) val *= caches.v1[d](p, multi_idx[d]);
                    else val *= caches.v0[d](p, multi_idx[d]);
                }
                grads[target_d](p, i) = val;
            }
        }
    }
    return grads;
}

std::vector<Eigen::MatrixXd> ChebyshevAnsatz::evaluate_basis_hess(const Eigen::MatrixXd& grid) const {
    int n_points = grid.rows();
    Caches1D caches = build_cheb_caches(m_inv_dim, m_order, m_interval, grid);
    int hess_dim = m_inv_dim * m_inv_dim;
    std::vector<Eigen::MatrixXd> hesss(hess_dim, Eigen::MatrixXd::Zero(n_points, m_num_coeffs));

    for (int p = 0; p < n_points; ++p) {
        for (int i = 0; i < m_num_coeffs; ++i) {
            std::vector<int> multi_idx(m_inv_dim);
            int current_flat = i;
            for (int d = m_inv_dim - 1; d >= 0; --d) {
                multi_idx[d] = current_flat % (m_order(d) + 1);
                current_flat /= (m_order(d) + 1);
            }

            for (int d1 = 0; d1 < m_inv_dim; ++d1) {
                for (int d2 = 0; d2 < m_inv_dim; ++d2) {
                    double val = 1.0;
                    for (int d = 0; d < m_inv_dim; ++d) {
                        if (d == d1 && d == d2) val *= caches.v2[d](p, multi_idx[d]);
                        else if (d == d1 || d == d2) val *= caches.v1[d](p, multi_idx[d]);
                        else val *= caches.v0[d](p, multi_idx[d]);
                    }
                    hesss[d1 * m_inv_dim + d2](p, i) = val;
                }
            }
        }
    }
    return hesss;
}

// --- PolyAnsatz ---

PolyAnsatz::PolyAnsatz(int inv_dim, const Eigen::VectorXi& order, const Eigen::VectorXd& center)
    : Ansatz(inv_dim, 0), m_order(order), m_center(center) {
    int total = 1;
    for (int i = 0; i < m_order.size(); ++i) total *= (m_order(i) + 1);
    this->m_num_coeffs = total;
}

Eigen::MatrixXd PolyAnsatz::evaluate_basis(const Eigen::MatrixXd& grid) const {
    int n_points = grid.rows();
    Caches1D caches = build_poly_caches(m_inv_dim, m_order, m_center, grid);
    Eigen::MatrixXd M = Eigen::MatrixXd::Ones(n_points, m_num_coeffs);

    for (int p = 0; p < n_points; ++p) {
        for (int i = 0; i < m_num_coeffs; ++i) {
            int current_flat = i;
            double val = 1.0;
            for (int d = m_inv_dim - 1; d >= 0; --d) {
                int i_d = current_flat % (m_order(d) + 1);
                current_flat /= (m_order(d) + 1);
                val *= caches.v0[d](p, i_d);
            }
            M(p, i) = val;
        }
    }
    return M;
}

std::vector<Eigen::MatrixXd> PolyAnsatz::evaluate_basis_grad(const Eigen::MatrixXd& grid) const {
    int n_points = grid.rows();
    Caches1D caches = build_poly_caches(m_inv_dim, m_order, m_center, grid);
    std::vector<Eigen::MatrixXd> grads(m_inv_dim, Eigen::MatrixXd::Zero(n_points, m_num_coeffs));

    for (int p = 0; p < n_points; ++p) {
        for (int i = 0; i < m_num_coeffs; ++i) {
            std::vector<int> multi_idx(m_inv_dim);
            int current_flat = i;
            for (int d = m_inv_dim - 1; d >= 0; --d) {
                multi_idx[d] = current_flat % (m_order(d) + 1);
                current_flat /= (m_order(d) + 1);
            }

            for (int target_d = 0; target_d < m_inv_dim; ++target_d) {
                double val = 1.0;
                for (int d = 0; d < m_inv_dim; ++d) {
                    if (d == target_d) val *= caches.v1[d](p, multi_idx[d]);
                    else val *= caches.v0[d](p, multi_idx[d]);
                }
                grads[target_d](p, i) = val;
            }
        }
    }
    return grads;
}

std::vector<Eigen::MatrixXd> PolyAnsatz::evaluate_basis_hess(const Eigen::MatrixXd& grid) const {
    int n_points = grid.rows();
    Caches1D caches = build_poly_caches(m_inv_dim, m_order, m_center, grid);
    int hess_dim = m_inv_dim * m_inv_dim;
    std::vector<Eigen::MatrixXd> hesss(hess_dim, Eigen::MatrixXd::Zero(n_points, m_num_coeffs));

    for (int p = 0; p < n_points; ++p) {
        for (int i = 0; i < m_num_coeffs; ++i) {
            std::vector<int> multi_idx(m_inv_dim);
            int current_flat = i;
            for (int d = m_inv_dim - 1; d >= 0; --d) {
                multi_idx[d] = current_flat % (m_order(d) + 1);
                current_flat /= (m_order(d) + 1);
            }

            for (int d1 = 0; d1 < m_inv_dim; ++d1) {
                for (int d2 = 0; d2 < m_inv_dim; ++d2) {
                    double val = 1.0;
                    for (int d = 0; d < m_inv_dim; ++d) {
                        if (d == d1 && d == d2) val *= caches.v2[d](p, multi_idx[d]);
                        else if (d == d1 || d == d2) val *= caches.v1[d](p, multi_idx[d]);
                        else val *= caches.v0[d](p, multi_idx[d]);
                    }
                    hesss[d1 * m_inv_dim + d2](p, i) = val;
                }
            }
        }
    }
    return hesss;
}

} // namespace core
} // namespace frgfp