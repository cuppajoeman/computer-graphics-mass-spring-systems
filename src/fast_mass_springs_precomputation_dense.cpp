#include "fast_mass_springs_precomputation_dense.h"
#include "signed_incidence_matrix_dense.h"
#include <Eigen/Dense>

bool fast_mass_springs_precomputation_dense(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & E,
  const double k,
  const Eigen::VectorXd & m,
  const Eigen::VectorXi & b,
  const double delta_t,
  Eigen::VectorXd & r,
  Eigen::MatrixXd & M,
  Eigen::MatrixXd & A,
  Eigen::MatrixXd & C,
  Eigen::LLT<Eigen::MatrixXd> & prefactorization)
{
  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(V.rows(),V.rows());

  // here we're computing the rest lenghts which is simply the euclidean distance 
  // between the connected vertices, its the rest length because this is the distance the
  // springs want to maintain
  r.resize(E.rows()); 

  for (int edge_index = 0; edge_index < E.rows(); edge_index++){
    auto diff = V.row(E(edge_index, 0)) - V.row(E(edge_index, 1));
    r(edge_index) = (diff).norm();
  }

  // zero out the matrix
  M = Eigen::MatrixXd::Zero(V.rows(), V.rows());

  // here we are constructing a diagonal mass matrix
  for (int mass_mat_index = 0; mass_mat_index < M.rows(); mass_mat_index++){
    M(mass_mat_index, mass_mat_index) = m(mass_mat_index);
  }

  A = Eigen::MatrixXd::Zero(E.rows(), V.rows());
  signed_incidence_matrix_dense(V.rows(), E, A);

  // here we construct a matrix such that each row 
  // of the matrix corresponds to vertex which is pinned
  // we specify which vertex is pinned by setting it equal to one in that row.
  C = Eigen::MatrixXd::Zero(b.rows(), V.rows()); 
  for (int idx = 0; idx < b.rows(); idx++){
    C(idx, b(idx)) = 1;
  }

  // recall that Q was defined via the following equation:
  // k A^T A + 1 / (del_t^2) M in R ^ {n x n}
  Q = k * A.transpose() * A + 1 / (pow(delta_t, 2)) * M;

  // as suggested in the handout we set:
  double w = 1e10; 
  Q += w * C.transpose() * C; // here we're adding the quadratic coef onto Q

  prefactorization.compute(Q);
  return prefactorization.info() != Eigen::NumericalIssue;
}
