#pragma once

#include <cmath>  // pow function
#include <tuple>  // c++ tuples
#include <algorithm>  // std::min_element function
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;
using std::vector;

// helper functions for convex MwPGP algorithm
std::tuple<double, double, double> projection_L2_balls(double x1, double x2, double x3, double m_maxima);
std::tuple<double, double, double> phi_MwPGP(double x1, double x2, double x3, double g1, double g2, double g3, double m_maxima);
std::tuple<double, double, double> beta_tilde(double x1, double x2, double x3, double g1, double g2, double g3, double alpha, double m_maxima);
std::tuple<double, double, double> g_reduced_gradient(double x1, double x2, double x3, double g1, double g2, double g3, double alpha, double m_maxima);
std::tuple<double, double, double> g_reduced_projected_gradient(double x1, double x2, double x3, double g1, double g2, double g3, double alpha, double m_maxima);
double find_max_alphaf(double x1, double x2, double x3, double p1, double p2, double p3, double m_maxima);
void print_MwPGP(Array& A_obj, Array& b_obj, Array& x_k1, Array& m_proxy, Array& m_maxima, Array& m_history, Array& objective_history, Array& R2_history, int print_iter, int k, double nu, double reg_l0, double reg_l1, double reg_l2, double reg_l2_shift);

// the hyperparameters all have default values if they are left unspecified -- see python.cpp
std::tuple<Array, Array, Array, Array> MwPGP_algorithm(Array& A_obj, Array& b_obj, Array& ATb, Array& m_proxy, Array& m0, Array& m_maxima, double alpha, double nu=1.0e100, double delta=0.5, double epsilon=1.0e-4, double reg_l0=0.0, double reg_l1=0.0, double reg_l2=0.0, double reg_l2_shifted=0.0, int max_iter=500, double min_fb=1.0e-20, bool verbose=false);

// helper functions for projected L-BFGS (can be used for convex solve in relax-and-split instead of MWPGP)
Array f_PQN(Array& A_obj, Array& b_obj, Array& xk, Array& m_proxy, Array& m_maxima, double reg_l2, double reg_l2_shift, double nu);
Array df_PQN(Array& A_obj, Array& b_obj, Array& ATb_rs, Array& xk, double reg_l2, double reg_l2_shift, double nu);

// right now, only solving QPQCs so f = q and these functions are repetitive. But this allows for
// situation in which f is not a QP, and q is the QP approximation (Taylor expansion) of f.
Array q_PQN(Array& A_obj, Array& b_obj, Array& xk, Array& m_proxy, Array& m_maxima, double reg_l2, double reg_l2_shift, double nu);
Array qf_PQN(Array& A_obj, Array& b_obj, Array& ATb_rs, Array& xk, double reg_l2, double reg_l2_shift, double nu);

// Solves quadratic programs with quadratic constraints (QPQCs) so for relax-and-split SPG and PQN should
// perform basically the same (there is no approximation to assume that we have a quadratic program)
std::tuple<Array, Array, Array, Array> SPG(Array& A_obj, Array& b_obj, Array& ATb, Array& m_proxy, Array& m0, Array& m_maxima, double alpha_min, double alpha_max, double alpha_bb, int h, double reg_l2, double reg_l2_shifted, double nu)

// the hyperparameters all have default values if they are left unspecified -- see python.cpp
std::tuple<Array, Array, Array, Array> PQN_algorithm(Array& A_obj, Array& b_obj, Array& ATb, Array& m_proxy, Array& m0, Array& m_maxima, int max_iter, double epsilon, bool verbose, double reg_l0, double reg_l1, double reg_l2, double reg_l2_shift, double nu);

// helper functions for nonconvex, binary matching pursuit algorithm (no relax-and-split here)
void print_BMP(Array& A_obj, Array& b_obj, Array& x_k1, Array& m_history, Array& objective_history, Array& R2_history, int print_iter, int k, double reg_l2, double reg_l2_shift);

// the hyperparameters all have default values if they are left unspecified -- see python.cpp
std::tuple<Array, Array, Array, Array> BMP_algorithm(Array& A_obj, Array& b_obj, Array& ATb, int K, double reg_l2, double reg_l2_shift, bool verbose);
