#include "signed_incidence_matrix_sparse.h"
#include <vector>

void signed_incidence_matrix_sparse(
  const int n,
  const Eigen::MatrixXi & E,
  Eigen::SparseMatrix<double>  & A)
{
  std::vector<Eigen::Triplet<double>> ijv;
  for (int edge_idx = 0; edge_idx < E.rows(); edge_idx++) {
    // doing triplets for each non-zero and add it to the matrix (per handout spec)
    // same logic as dense matrix approach
    ijv.emplace_back(Eigen::Triplet<double>(edge_idx, E(edge_idx, 0), 1));
    ijv.emplace_back(Eigen::Triplet<double>(edge_idx, E(edge_idx, 1), -1));
  }
  A.resize(E.rows(), n);
  A.setFromTriplets(ijv.begin(), ijv.end());
}
