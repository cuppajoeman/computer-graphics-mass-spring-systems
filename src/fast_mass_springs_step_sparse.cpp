#include "fast_mass_springs_step_sparse.h"
#include <igl/matlab_format.h>

void fast_mass_springs_step_sparse(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & E,
  const double k,
  const Eigen::VectorXi & b,
  const double delta_t,
  const Eigen::MatrixXd & fext,
  const Eigen::VectorXd & r,
  const Eigen::SparseMatrix<double>  & M,
  const Eigen::SparseMatrix<double>  & A,
  const Eigen::SparseMatrix<double>  & C,
  const Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > & prefactorization,
  const Eigen::MatrixXd & Uprev,
  const Eigen::MatrixXd & Ucur,
  Eigen::MatrixXd & Unext)
{
  Eigen::MatrixXd d = Eigen::MatrixXd::Zero(E.rows(), 3);
  Eigen::MatrixXd Ucur_copy = Ucur;
  int total_iterations = 50;
  double w = 1e10; 

  for (int num_iterations = 0; num_iterations < total_iterations; num_iterations++) {

    // our goal here is to vectorize the equations specified in the handout
    // right here we are constructing d_ij in matrix from
    for (int d_idx = 0; d_idx < d.rows(); d_idx++) {
      auto edge_dir = (Ucur_copy.row(E(d_idx,0)) - Ucur_copy.row(E(d_idx, 1))).normalized();
      d.row(d_idx) = r(d_idx) * edge_dir;
    }
      
    // using the equation specified in the handout:
    Eigen::MatrixXd y = 1 / (pow(delta_t, 2)) * M * (2 * Ucur - Uprev) + fext; // in R ^ { n x 3}
    
    // note that in this equation what we're doing is implementing the matrix version (vectorized)
    // version of the original equations. Note that V is the matrix of rest vertex positions 
    // and that the final added term is the linear coefficient as specified in the handout.
    const Eigen::MatrixXd l = k * A.transpose() * d + y + w * C.transpose() * C * V;
    Ucur_copy = prefactorization.solve(l);
  }
  Unext = Ucur_copy;
}
