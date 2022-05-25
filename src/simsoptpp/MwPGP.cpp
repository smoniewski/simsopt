#include "MwPGP.h"
#include <Eigen/Dense>

// Project a 3-vector onto the L2 ball with radius m_maxima
std::tuple<double, double, double> projection_L2_balls(double x1, double x2, double x3, double m_maxima) {
    // project a 3-vector on the unit ball
    double denom = std::max(1.0, sqrt(x1 * x1 + x2 * x2 + x3 * x3) / m_maxima); 
    return std::make_tuple(x1 / denom, x2 / denom, x3 / denom);
}

// Takes a vector and zeros if is very close to the L2 ball surface
std::tuple<double, double, double> phi_MwPGP(double x1, double x2, double x3, double g1, double g2, double g3, double m_maxima)
{
    // phi(x_i, g_i) = g_i(x_i) if x_i is not on the L2 ball,
    // otherwise set phi to zero
    double xmag2 = x1 * x1 + x2 * x2 + x3 * x3;
    if (abs(xmag2 - m_maxima * m_maxima) > 1.0e-8 + 1.0e-5 * m_maxima * m_maxima) {
        return std::make_tuple(g1, g2, g3);
    }
    else {
        // if triplet is in the active set (on the L2 unit ball)
        // then zero out those three indices
        return std::make_tuple(0.0, 0.0, 0.0);
    }
}

// Takes a vector and zeros it if it is NOT on the L2 ball surface
std::tuple<double, double, double> beta_tilde(double x1, double x2, double x3, double g1, double g2, double g3, double alpha, double m_maxima)
{
    // beta_tilde(x_i, g_i, alpha) = 0_i if the ith triplet
    // is not on the L2 ball, otherwise is equal to different
    // values depending on the orientation of g.
    double ng, mmax2, dist;
    dist = x1 * x1 + x2 * x2 + x3 * x3;
    mmax2 = m_maxima * m_maxima;
    if (abs(dist - mmax2) < (1.0e-8 + 1.0e-5 * mmax2)) {
        ng = (x1 * g1 + x2 * g2 + x3 * g3) / sqrt(dist);
        if (ng > 0) { 
            return std::make_tuple(g1, g2, g3);
        }
        else {
            return g_reduced_gradient(x1, x2, x3, g1, g2, g3, alpha, m_maxima);
        }
    }
    else {
        // if triplet is NOT in the active set (on the L2 unit ball)
        // then zero out those three indices
        return std::make_tuple(0.0, 0.0, 0.0);
    }
}

// The reduced gradient of G is simply the
// gradient step in the L2-ball-projected direction.
std::tuple<double, double, double> g_reduced_gradient(double x1, double x2, double x3, double g1, double g2, double g3, double alpha, double m_maxima) 
{
    double proj_L2x, proj_L2y, proj_L2z; 
    std::tie(proj_L2x, proj_L2y, proj_L2z) = projection_L2_balls(x1 - alpha * g1, x2 - alpha * g2, x3 - alpha * g3, m_maxima);
    return std::make_tuple((x1 - proj_L2x) / alpha, (x2 - proj_L2y) / alpha, (x3 - proj_L2z) / alpha);
}

// This function is just phi + beta_tilde 
std::tuple<double, double, double> g_reduced_projected_gradient(double x1, double x2, double x3, double g1, double g2, double g3, double alpha, double m_maxima) {
    double psum1, psum2, psum3, bsum1, bsum2, bsum3;
    std::tie(psum1, psum2, psum3) = phi_MwPGP(x1, x2, x3, g1, g2, g3, m_maxima);
    std::tie(bsum1, bsum2, bsum3) = beta_tilde(x1, x2, x3, g1, g2, g3, alpha, m_maxima);
    return std::make_tuple(psum1 + bsum1, psum2 + bsum2, psum3 + bsum3);
}

// Solve a quadratic equation to determine the largest
// step size alphaf such that the entirety of x - alpha * p
// lives in the L2 ball with radius m_maxima 
double find_max_alphaf(double x1, double x2, double x3, double p1, double p2, double p3, double m_maxima) {
    double a, b, c, alphaf_plus;
    double tol = 1e-20;
    a = p1 * p1 + p2 * p2 + p3 * p3;
    c = x1 * x1 + x2 * x2 + x3 * x3 - m_maxima * m_maxima;
    b = - 2 * (x1 * p1 + x2 * p2 + x3 * p3);
    if (a > tol) { 
        // c is always negative and a is always positive
        // so alphaf_plus >= 0 always
        alphaf_plus = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
    }
    else {
        alphaf_plus = 1e100; // ignore the value
    }
    return alphaf_plus;
}

// print out all the possible loss terms in the objective function
// and record histories of the dipole moments, objective values, etc.
void print_verbose(Array& A_obj, Array& b_obj, Array& x_k1, Array& m_proxy, Array& m_maxima, Array& m_history, Array& objective_history, Array& R2_history, int print_iter, int k, double nu, double reg_l0, double reg_l1, double reg_l2, double reg_l2_shift)
{
    int Ngrid = A_obj.shape(0);
    int N = m_maxima.shape(0);
    double R2 = 0.0;
    double N2 = 0.0;
    double L2 = 0.0;
    double L2_shift = 0.0;
    double L1 = 0.0;
    double L0 = 0.0;
    double cost = 0.0;
    double l0_tol = 1e-20;
    Array R2_temp = xt::zeros<double>({Ngrid});
#pragma omp parallel for reduction(+: N2, L2, L2_shift, L1, L0)
    for(int i = 0; i < N; ++i) {
	for(int ii = 0; ii < 3; ++ii) {
	    m_history(i, ii, print_iter) = x_k1(i, ii);
	    N2 += (x_k1(i, ii) - m_proxy(i, ii)) * (x_k1(i, ii) - m_proxy(i, ii));
	    L2 += x_k1(i, ii) * x_k1(i, ii);
	    L2_shift += (x_k1(i, ii) - m_maxima(i)) * (x_k1(i, ii) - m_maxima(i));
	    L1 += abs(x_k1(i, ii));
	    L0 += ((abs(m_proxy(i, ii)) < l0_tol) ? 1.0 : 0.0);
	}
    }

    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_mat(const_cast<double*>(A_obj.data()), Ngrid, 3*N);
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_v(const_cast<double*>(x_k1.data()), 3*N, 1);
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_res(const_cast<double*>(R2_temp.data()), Ngrid, 1);
    eigen_res = eigen_mat*eigen_v;
#pragma omp parallel for reduction(+: R2)
    for(int i = 0; i < Ngrid; ++i) {
	R2 += (R2_temp(i) - b_obj(i)) * (R2_temp(i) - b_obj(i));
    }
    R2 = 0.5 * R2;
    N2 = 0.5 * N2 / nu;
    L2 = reg_l2 * L2;
    L2_shift = reg_l2_shift * L2_shift;
    L1 = reg_l1 * L1;
    L0 = reg_l0 * L0;
    cost = R2 + N2 + L2 + L2_shift; // + L1 + L0;
    objective_history(print_iter) = cost;
    R2_history(print_iter) = R2;
    printf("%d ... %.2e ... %.2e ... %.2e ... %.2e ... %.2e ... %.2e ... %.2e \n", k, R2, N2, L2, L2_shift, L1, L0, cost);
}


// Run the overall algorithm for solving the convex part of
// the permanent magnet optimization problem. This algorithm has
// many optional parameters for additional loss terms.
//std::tuple<Array, Array, Array, Array> MwPGP_algorithm(Array& A_obj, Array& b_obj, Array& ATb, Array& m_proxy, Array& m0, Array& m_maxima, Array& U, Array& SV, double alpha, double nu, double delta, double epsilon, double reg_l0, double reg_l1, double reg_l2, double reg_l2_shift, int max_iter, bool verbose)
//std::tuple<Array, Array, Array, Array> MwPGP_algorithm(Array& A_obj, Array& b_obj, Array& ATA, Array& ATb, Array& m_proxy, Array& m0, Array& m_maxima, double alpha, double nu, double delta, double epsilon, double reg_l0, double reg_l1, double reg_l2, double reg_l2_shift, int max_iter, bool verbose)
std::tuple<Array, Array, Array, Array> MwPGP_algorithm(Array& A_obj, Array& b_obj, Array& ATb, Array& m_proxy, Array& m0, Array& m_maxima, double alpha, double nu, double delta, double epsilon, double reg_l0, double reg_l1, double reg_l2, double reg_l2_shift, int max_iter, bool verbose)
{
    // Needs ATb in shape (N, 3)
    //int rank = U.shape(1);
    int npoints = A_obj.shape(0);
    int N = ATb.shape(0);
    int print_iter = 0;
    double x_sum;
    Array g = xt::zeros<double>({N, 3});
    Array p = xt::zeros<double>({N, 3});
    Array ATAp = xt::zeros<double>({N, 3});

    // define bunch of doubles, mostly for setting the std::tuples correctly
    double norm_g_alpha_p, norm_phi_temp, gamma, gp, pATAp;
    double g_alpha_p1, g_alpha_p2, g_alpha_p3, phi_temp1, phi_temp2, phi_temp3;
    double phig1, phig2, phig3, p_temp1, p_temp2, p_temp3;
    double alpha_cg, alpha_f;
    vector<double> alpha_fs(N);
    Array x_k1 = m0;
    Array x_k_prev;

    // record the history of the algorithm iterations
    Array m_history = xt::zeros<double>({N, 3, 21});
    Array objective_history = xt::zeros<double>({21});
    Array R2_history = xt::zeros<double>({21});

    // Add contribution from relax-and-split term
    Array ATb_rs = ATb + m_proxy / nu;

    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_mat(const_cast<double*>(A_obj.data()), npoints, 3*N);
    //Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_matT(const_cast<double*>(eigen_mat.transpose().data()), 3*N, npoints);
    //Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_mat(const_cast<double*>(ATA.data()), 3*N, 3*N);
    //Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_u(const_cast<double*>(U.data()), 3*N, rank);
    //Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_sv(const_cast<double*>(SV.data()), rank, 3*N);

    // Set up initial g and p Arrays 
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_v(const_cast<double*>(m0.data()), 1, 3*N);
    Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_res(const_cast<double*>(g.data()), 1, 3*N);
    // if using A instead of ATA, need to include contributions from L2 
    //eigen_res = eigen_v*eigen_mat;
    eigen_res = eigen_v*eigen_mat.transpose()*eigen_mat + 2 * eigen_v * (reg_l2 + reg_l2_shift + 1.0 / (2.0 * nu));  
    
    g -= ATb_rs;  // subtract off relax-and-split contribution
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        std::tie(p(i, 0), p(i, 1), p(i, 2)) = phi_MwPGP(m0(i, 0), m0(i, 1), m0(i, 2), g(i, 0), g(i, 1), g(i, 2), m_maxima(i));
    }

    if (verbose)
        printf("Iteration ... |Am - b|^2 ... |m-w|^2/v ...   a|m|^2 ...  b|m-1|^2 ...   c|m|_1 ...   d|m|_0 ... Total Error:\n");

    // Main loop over the optimization iterations
    for (int k = 0; k < max_iter; ++k) {
	
	x_k_prev = x_k1;
       
	// fairly convoluted way to print every ~ max_iter / 10 iterations 
        if (verbose && ((k % ((int)(max_iter / 20)) == 0) || k == 0 || k == max_iter - 1)) {
	    print_verbose(A_obj, b_obj, x_k1, m_proxy, m_maxima, m_history, objective_history, R2_history, print_iter, k, nu, reg_l0, reg_l1, reg_l2, reg_l2_shift);
            print_iter += 1;
	}    

        // compute L2 norm of reduced g and L2 norm of phi(x, g)
        // as well as some dot products needed for the algorithm
        norm_g_alpha_p = 0.0;
        norm_phi_temp = 0.0;
        gp = 0.0;
        pATAp = 0.0;
        ATAp = xt::zeros<double>({N, 3});
        Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_v(const_cast<double*>(p.data()), 1, 3*N);
        Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_res(const_cast<double*>(ATAp.data()), 1, 3*N);
        eigen_res = eigen_v*eigen_mat.transpose()*eigen_mat + 2 * eigen_v * (reg_l2 + reg_l2_shift + 1.0 / (2.0 * nu));
        //eigen_res = eigen_v*eigen_mat;
        //eigen_res = eigen_v*eigen_u*eigen_sv;

#pragma omp parallel for reduction(+: norm_g_alpha_p, norm_phi_temp, gp, pATAp) private(phi_temp1, phi_temp2, phi_temp3, g_alpha_p1, g_alpha_p2, g_alpha_p3)
        for(int i = 0; i < N; ++i) {
            std::tie(g_alpha_p1, g_alpha_p2, g_alpha_p3) = g_reduced_projected_gradient(x_k1(i, 0), x_k1(i, 1), x_k1(i, 2), g(i, 0), g(i, 1), g(i, 2), alpha, m_maxima(i));
            std::tie(phi_temp1, phi_temp2, phi_temp3) = phi_MwPGP(x_k1(i, 0), x_k1(i, 1), x_k1(i, 2), g(i, 0), g(i, 1), g(i, 2), m_maxima(i));
	    norm_g_alpha_p += g_alpha_p1 * g_alpha_p1 + g_alpha_p2 * g_alpha_p2 + g_alpha_p3 * g_alpha_p3;
            norm_phi_temp += phi_temp1 * phi_temp1 + phi_temp2 * phi_temp2 + phi_temp3 * phi_temp3;
	    gp += g(i, 0) * p(i, 0) + g(i, 1) * p(i, 1) + g(i, 2) * p(i, 2);
            alpha_fs[i] = find_max_alphaf(x_k1(i, 0), x_k1(i, 1), x_k1(i, 2), p(i, 0), p(i, 1), p(i, 2), m_maxima(i));
            pATAp += p(i, 0) * ATAp(i, 0) + p(i, 1) * ATAp(i, 1) + p(i, 2) * ATAp(i, 2);
        }

        auto max_i = std::min_element(alpha_fs.begin(), alpha_fs.end()); 
        alpha_f = *max_i;
        alpha_cg = gp / pATAp;

        // based on these norms, decide what kind of a descent step to take
        if (2 * delta * norm_g_alpha_p <= norm_phi_temp) {
            if (alpha_cg < alpha_f) {
                // Take a conjugate gradient step
#pragma omp parallel for
                for (int i = 0; i < N; ++i) {  
                    for (int ii = 0; ii < 3; ++ii) {  
                        x_k1(i, ii) += - alpha_cg * p(i, ii);
                        g(i, ii) += - alpha_cg * ATAp(i, ii);
                    }
                }
                // compute gamma step size
                gamma = 0.0;
#pragma omp parallel for reduction(+: gamma) private(phig1, phig2, phig3)
                for (int i = 0; i < N; ++i) {     
                    std::tie(phig1, phig2, phig3) = phi_MwPGP(x_k1(i, 0), x_k1(i, 1), x_k1(i, 2), g(i, 0), g(i, 1), g(i, 2), m_maxima(i));
                    gamma += (phig1 * ATAp(i, 0) + phig2 * ATAp(i, 1) + phig3 * ATAp(i, 2)); 
                }    
                gamma = gamma / pATAp;
                // update p
#pragma omp parallel for private(p_temp1, p_temp2, p_temp3)
                for (int i = 0; i < N; ++i) {     
                    std::tie(p_temp1, p_temp2, p_temp3) = phi_MwPGP(x_k1(i, 0), x_k1(i, 1), x_k1(i, 2), g(i, 0), g(i, 1), g(i, 2), m_maxima(i));
                    p(i, 0) = p_temp1 - gamma * p(i, 0);
                    p(i, 1) = p_temp2 - gamma * p(i, 1);
                    p(i, 2) = p_temp3 - gamma * p(i, 2);
                }
            }
            else {
                // Take a mixed projected gradient step
#pragma omp parallel for
                for (int i = 0; i < N; ++i) {     
                    std::tie(x_k1(i, 0), x_k1(i, 1), x_k1(i, 2)) = projection_L2_balls((x_k1(i, 0) - alpha_f * p(i, 0)) - alpha * (g(i, 0) - alpha_f * ATAp(i, 0)), (x_k1(i, 1) - alpha_f * p(i, 1)) - alpha * (g(i, 1) - alpha_f * ATAp(i, 1)), (x_k1(i, 2) - alpha_f * p(i, 2)) - alpha * (g(i, 2) - alpha_f * ATAp(i, 2)), m_maxima(i));
                }
                // update g and p 
                // g = xt::zeros<double>({N, 3});
                Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_v(const_cast<double*>(x_k1.data()), 1, 3*N);
                Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_res(const_cast<double*>(g.data()), 1, 3*N);
                //eigen_res = eigen_v*eigen_mat;
                eigen_res = eigen_v*eigen_mat.transpose()*eigen_mat + 2 * eigen_v * (reg_l2 + reg_l2_shift + 1.0 / (2.0 * nu));

#pragma omp parallel for
                for (int i = 0; i < N; ++i) {     
                    for (int jj = 0; jj < 3; ++jj) {
                        g(i, jj) += - ATb_rs(i, jj);
                    }
                    std::tie(p(i, 0), p(i, 1), p(i, 2)) = phi_MwPGP(x_k1(i, 0), x_k1(i, 1), x_k1(i, 2), g(i, 0), g(i, 1), g(i, 2), m_maxima(i));
                }
            }
        }
        else {
            // projected gradient descent method
#pragma omp parallel for
            for (int i = 0; i < N; ++i) {
                std::tie(x_k1(i, 0), x_k1(i, 1), x_k1(i, 2)) = projection_L2_balls(x_k1(i, 0) - alpha * g(i, 0), x_k1(i, 1) - alpha * g(i, 1), x_k1(i, 2) - alpha * g(i, 2), m_maxima(i));
            }
            // update g and p
            // g = xt::zeros<double>({N, 3});
            Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_v(const_cast<double*>(x_k1.data()), 1, 3*N);
            Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>> eigen_res(const_cast<double*>(g.data()), 1, 3*N);
            //eigen_res = eigen_v*eigen_mat;
            eigen_res = eigen_v*eigen_mat.transpose()*eigen_mat + 2 * eigen_v * (reg_l2 + reg_l2_shift + 1.0 / (2.0 * nu));

#pragma omp parallel for
            for (int i = 0; i < N; ++i) {     
                for (int jj = 0; jj < 3; ++jj) {
                    g(i, jj) += - ATb_rs(i, jj);
                }
                std::tie(p(i, 0), p(i, 1), p(i, 2)) = phi_MwPGP(x_k1(i, 0), x_k1(i, 1), x_k1(i, 2), g(i, 0), g(i, 1), g(i, 2), m_maxima(i));
            }
        }
	// check if converged
	x_sum = 0; 
#pragma omp parallel for
        for (int i = 0; i < N; ++i) {     
            for (int ii = 0; ii < 3; ++ii) {
                x_sum += abs(x_k1(i, ii) - x_k_prev(i, ii));
	    }
	}
	if (x_sum < epsilon) {
            printf("MwPGP algorithm ended early, at iteration %d\n", k);
	    break;
	}
    }
    return std::make_tuple(objective_history, R2_history, m_history, x_k1);
}
