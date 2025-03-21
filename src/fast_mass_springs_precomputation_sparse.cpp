#include "fast_mass_springs_precomputation_sparse.h"
#include "signed_incidence_matrix_sparse.h"
#include <vector>

bool fast_mass_springs_precomputation_sparse(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & E,
  const double k,
  const Eigen::VectorXd & m,
  const Eigen::VectorXi & b,
  const double delta_t,
  Eigen::VectorXd & r,
  Eigen::SparseMatrix<double>  & M,
  Eigen::SparseMatrix<double>  & A,
  Eigen::SparseMatrix<double>  & C,
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > & prefactorization)
{
  /////////////////////////////////////////////////////////////////////////////
  // Replace with your code
  const int n = V.rows();
  Eigen::SparseMatrix<double> Q(n,n);


  // here we're computing the rest lenghts which is simply the euclidean distance 
  // between the connected vertices, its the rest length because this is the distance the
  // springs want to maintain
  r.resize(E.rows());
  for (int edge_idx = 0; edge_idx < E.rows(); edge_idx++){
    auto diff = V.row(E(edge_idx, 0)) - V.row(E(edge_idx, 1));
    r(edge_idx) = diff.norm();
  }

  // here we are setting up a diagonal mass matrix
  M.resize(n, n);
  std::vector<Eigen::Triplet<double>> ijv_mass_mat;
  for (int mass_mat_idx = 0; mass_mat_idx < M.rows(); mass_mat_idx++){
  	ijv_mass_mat.emplace_back(mass_mat_idx, mass_mat_idx, m(mass_mat_idx));
  }
  M.setFromTriplets(ijv_mass_mat.begin(),ijv_mass_mat.end());

  // load it up into A
  signed_incidence_matrix_sparse(V.rows(), E, A);

  // here we construct a matrix such that each row 
  // of the matrix corresponds to vertex which is pinned
  // we specify which vertex is pinned by setting it equal to one in that row.
  C.resize(b.rows(), n);
  std::vector<Eigen::Triplet<double>> ijv_pinned_vertex_mat;
  for (int pinned_mat_idx = 0; pinned_mat_idx < C.rows(); pinned_mat_idx++){
  	ijv_pinned_vertex_mat.emplace_back(pinned_mat_idx, b(pinned_mat_idx) ,1);
  }
  C.setFromTriplets(ijv_pinned_vertex_mat.begin(),ijv_pinned_vertex_mat.end());

  // recall that Q was defined via the following equation:
  // k A^T A + 1 / (del_t^2) M in R ^ {n x n}
  Q = k * A.transpose() * A + 1 / (pow(delta_t, 2)) * M;

  // as suggested in the handout we set:
  double w = 1e10; 
  Q += w * C.transpose() * C; // here we're adding the quadratic coef onto Q
  
  prefactorization.compute(Q);
  return prefactorization.info() != Eigen::NumericalIssue;
}
