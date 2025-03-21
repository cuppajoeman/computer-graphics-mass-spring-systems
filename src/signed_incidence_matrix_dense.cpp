#include "signed_incidence_matrix_dense.h"

void signed_incidence_matrix_dense(
  const int n,
  const Eigen::MatrixXi & E,
  Eigen::MatrixXd & A)
{
  A = Eigen::MatrixXd::Zero(E.rows(), n);
  for (int edge_idx = 0; edge_idx < E.rows(); edge_idx++) {
    // here we're creating a row in the matrix where exactly one element in the row
    // is one and the other is negative one, with the rest being zeros in this case 
    // the one represents the starting point and the -1 is the end point, because we 
    // are operating on a directed graph
    A(edge_idx, E(edge_idx, 0)) = 1;
    A(edge_idx, E(edge_idx, 1)) = -1;
  }
}
