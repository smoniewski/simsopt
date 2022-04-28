#include "dipole_field.h"
#include "simdhelpers.h"
#include "vec3dsimd.h"

// Calculate the B field at a set of evaluation points from N dipoles
// points: where to evaluate the field
// m_points: where the dipoles are located
// m: dipole moments ('orientation')
// everything in xyz coordinates
Array dipole_field_B(Array& points, Array& m_points, Array& m) {
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(m_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("m_points needs to be in row-major storage order");
    if(m.layout() != xt::layout_type::row_major)
          throw std::runtime_error("m needs to be in row-major storage order");

    int num_points = points.shape(0);
    int num_dipoles = m_points.shape(0);
    constexpr int simd_size = xsimd::simd_type<double>::size;
    Array B = xt::zeros<double>({points.shape(0), points.shape(1)});
   
    // initialize pointers to the beginning of m and the dipole grid
    double* m_points_ptr = &(m_points(0, 0));
    double* m_ptr = &(m(0, 0));
    double fak = 1e-7;  // mu0 divided by 4 * pi factor

    // Loop through the evaluation points by chunks of simd_size
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < num_points; i += simd_size) {
        auto point_i = Vec3dSimd();
        auto B_i = Vec3dSimd();
        // check that i + k isn't bigger than num_points
        int klimit = std::min(simd_size, num_points - i);
        for(int k = 0; k < klimit; k++){
            for (int d = 0; d < 3; ++d) {
                point_i[d][k] = points(i + k, d);
            }
        }
        for (int j = 0; j < num_dipoles; ++j) {
            Vec3dSimd m_j = Vec3dSimd(m_ptr[3 * j + 0], m_ptr[3 * j + 1], m_ptr[3 * j + 2]);
            Vec3dSimd mp_j = Vec3dSimd(m_points_ptr[3 * j + 0], m_points_ptr[3 * j + 1], m_points_ptr[3 * j + 2]);
            Vec3dSimd r = point_i - mp_j;
            simd_t rmag_2     = normsq(r);
            simd_t rmag_inv   = rsqrt(rmag_2);
            simd_t rmag_inv_3 = rmag_inv * (rmag_inv * rmag_inv);
            simd_t rmag_inv_5 = rmag_inv_3 * (rmag_inv * rmag_inv);
            simd_t rdotm = inner(r, m_j);
            B_i.x += 3.0 * rdotm * r.x * rmag_inv_5 - m_j.x * rmag_inv_3;
            B_i.y += 3.0 * rdotm * r.y * rmag_inv_5 - m_j.y * rmag_inv_3;
            B_i.z += 3.0 * rdotm * r.z * rmag_inv_5 - m_j.z * rmag_inv_3;
        } 
        for(int k = 0; k < klimit; k++){
            B(i + k, 0) = fak * B_i.x[k];
            B(i + k, 1) = fak * B_i.y[k];
            B(i + k, 2) = fak * B_i.z[k];
        }
    }
    return B;
}

Array dipole_field_dB(Array& points, Array& m_points, Array& m) {
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(m_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("m_points needs to be in row-major storage order");
    if(m.layout() != xt::layout_type::row_major)
          throw std::runtime_error("m needs to be in row-major storage order");

    int num_points = points.shape(0);
    int num_dipoles = m_points.shape(0);
    constexpr int simd_size = xsimd::simd_type<double>::size;
    Array dB = xt::zeros<double>({points.shape(0), points.shape(1), points.shape(1)});
    double* m_points_ptr = &(m_points(0, 0));
    double* m_ptr = &(m(0, 0));
    double fak = 1e-7;
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < num_points; i += simd_size) {
        auto point_i = Vec3dSimd();
        auto dB_i1   = Vec3dSimd();
        auto dB_i2   = Vec3dSimd();
        int klimit = std::min(simd_size, num_points - i);
        for(int k = 0; k < klimit; k++){
            for (int d = 0; d < 3; ++d) {
                point_i[d][k] = points(i + k, d);
            }
        }
        for (int j = 0; j < num_dipoles; ++j) {
            Vec3dSimd m_j = Vec3dSimd(m_ptr[3 * j], m_ptr[3 * j + 1], m_ptr[3 * j + 2]);
            Vec3dSimd mp_j = Vec3dSimd(m_points_ptr[3 * j], m_points_ptr[3 * j + 1], m_points_ptr[3 * j + 2]);
            Vec3dSimd r = point_i - mp_j;
            simd_t rmag_2     = normsq(r);
            simd_t rmag_inv   = rsqrt(rmag_2);
	    simd_t rmag_inv_2 = rmag_inv * rmag_inv;
            simd_t rmag_inv_3 = rmag_inv * rmag_inv_2;
            simd_t rmag_inv_5 = rmag_inv_3 * rmag_inv_2; 
            simd_t rdotm = inner(r, m_j);
            dB_i1.x += 3.0 * rmag_inv_5 * ((2.0 * m_j.x * r.x + rdotm) - 5.0 * rdotm * r.x * r.x * rmag_inv_2);
            dB_i1.y += 3.0 * rmag_inv_5 * ((m_j.x * r.y + m_j.y * r.x) - 5.0 * rdotm * r.x * r.y * rmag_inv_2);
            dB_i1.z += 3.0 * rmag_inv_5 * ((m_j.x * r.z + m_j.z * r.x) - 5.0 * rdotm * r.x * r.z * rmag_inv_2);
            dB_i2.x += 3.0 * rmag_inv_5 * ((2.0 * m_j.y * r.y + rdotm) - 5.0 * rdotm * r.y * r.y * rmag_inv_2);
            dB_i2.y += 3.0 * rmag_inv_5 * ((m_j.y * r.z + m_j.z * r.y) - 5.0 * rdotm * r.y * r.z * rmag_inv_2);
            dB_i2.z += 3.0 * rmag_inv_5 * ((2.0 * m_j.z * r.z + rdotm) - 5.0 * rdotm * r.z * r.z * rmag_inv_2);
        } 
        for(int k = 0; k < klimit; k++){
            dB(i + k, 0, 0) = fak * dB_i1.x[k];
            dB(i + k, 0, 1) = fak * dB_i1.y[k];
            dB(i + k, 0, 2) = fak * dB_i1.z[k];
            dB(i + k, 1, 1) = fak * dB_i2.x[k];
            dB(i + k, 1, 2) = fak * dB_i2.y[k];
            dB(i + k, 2, 2) = fak * dB_i2.z[k];
	    dB(i + k, 1, 0) = dB(i + k, 0, 1);
	    dB(i + k, 2, 0) = dB(i + k, 0, 2);
	    dB(i + k, 2, 1) = dB(i + k, 1, 2);
	}
    }
    return dB;
}

Array dipole_field_Bn(Array& points, Array& m_points, Array& unitnormal, int nfp, int stellsym) {
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(m_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("m_points needs to be in row-major storage order");
    if(unitnormal.layout() != xt::layout_type::row_major)
          throw std::runtime_error("unit normal needs to be in row-major storage order");
    
    int nsym = nfp * (stellsym + 1);
    int num_points = points.shape(0);
    int num_dipoles = m_points.shape(0);
    constexpr int simd_size = xsimd::simd_type<double>::size;
    Array geo_factor = xt::zeros<double>({num_points, num_dipoles, 3, nsym});
   
    // initialize pointers to the beginning of m and the dipole grid
    double* m_points_ptr = &(m_points(0, 0));
    // double* n_ptr = &(unitnormal(0, 0));
    double fak = 1e-7;  // mu0 divided by 4 * pi factor

    // Loop through the evaluation points by chunks of simd_size
#pragma omp parallel for schedule(static)
    for(int i = 0; i < num_points; i += simd_size) {
        auto point_i = Vec3dSimd();
        auto n_i = Vec3dSimd();
        // check that i + k isn't bigger than num_points
        int klimit = std::min(simd_size, num_points - i);
        for(int k = 0; k < klimit; k++){
            for (int d = 0; d < 3; ++d) {
                point_i[d][k] = points(i + k, d);
                n_i[d][k] = unitnormal(i + k, d);
            }
        }
	simd_t phi = atan2(point_i.y, point_i.x);
	//Vec3dSimd n_i = Vec3dSimd(n_ptr[3 * i + 0], n_ptr[3 * i + 1], n_ptr[3 * i + 2]);
	for(int fp = 0; fp < nfp; fp++) {
	    simd_t phi0 = (2 * M_PI / ((simd_t) nfp)) * fp;
	    simd_t phi_sym = phi + phi0;
	    simd_t sphi = sin(phi_sym); 
	    simd_t sphi0 = sin(phi0);
	    simd_t cphi = cos(phi_sym); 
	    simd_t cphi0 = cos(phi0);
            for (int j = 0; j < num_dipoles; ++j) {
                auto G_i = Vec3dSimd();
                auto H_i = Vec3dSimd();
                Vec3dSimd mp_j = Vec3dSimd(m_points_ptr[3 * j + 0], m_points_ptr[3 * j + 1], m_points_ptr[3 * j + 2]);
                simd_t mmag = sqrt(xsimd::fma(mp_j.x, mp_j.x, mp_j.y * mp_j.y));
		simd_t mp_x_new = mmag * cphi;
		simd_t mp_y_new = mmag * sphi;
		Vec3dSimd mp_j_new = Vec3dSimd(mp_x_new, mp_y_new, mp_j.z);
		Vec3dSimd r = point_i - mp_j_new;
                simd_t rmag_2 = normsq(r);
                simd_t rmag_inv   = rsqrt(rmag_2);
                simd_t rmag_inv_3 = rmag_inv * (rmag_inv * rmag_inv);
                simd_t rmag_inv_5 = rmag_inv_3 * (rmag_inv * rmag_inv);
                simd_t rdotn = inner(r, n_i);
                G_i.x = 3.0 * rdotn * r.x * rmag_inv_5 - n_i.x * rmag_inv_3;
                G_i.y = 3.0 * rdotn * r.y * rmag_inv_5 - n_i.y * rmag_inv_3;
                G_i.z = 3.0 * rdotn * r.z * rmag_inv_5 - n_i.z * rmag_inv_3;
	        // stellarator symmetry means dipole grid -> (x, -y, -z)
		// and geo_factor_x gets a minus sign
		Vec3dSimd mp_j_stell = Vec3dSimd(mp_x_new, -mp_y_new, -mp_j.z);
		r = point_i - mp_j_stell;
                rmag_2 = normsq(r);
                rmag_inv   = rsqrt(rmag_2);
                rmag_inv_3 = rmag_inv * (rmag_inv * rmag_inv);
                rmag_inv_5 = rmag_inv_3 * (rmag_inv * rmag_inv);
                rdotn = inner(r, n_i);
		H_i.x = 3.0 * rdotn * r.x * rmag_inv_5 - n_i.x * rmag_inv_3;
                H_i.y = 3.0 * rdotn * r.y * rmag_inv_5 - n_i.y * rmag_inv_3;
                H_i.z = 3.0 * rdotn * r.z * rmag_inv_5 - n_i.z * rmag_inv_3;
		for(int k = 0; k < klimit; k++){
		    geo_factor(i + k, j, 0, fp) = fak * (G_i.x[k] * cphi0[k] - G_i.y[k] * sphi0[k]);
		    geo_factor(i + k, j, 1, fp) = fak * (G_i.x[k] * sphi0[k] + G_i.y[k] * cphi0[k]);
		    geo_factor(i + k, j, 2, fp) = fak * G_i.z[k];
		    geo_factor(i + k, j, 0, nfp + fp) = - fak * (H_i.x[k] * cphi0[k] - H_i.y[k] * sphi0[k]);
		    geo_factor(i + k, j, 1, nfp + fp) = fak * (H_i.x[k] * sphi0[k] + H_i.y[k] * cphi0[k]);
		    geo_factor(i + k, j, 2, nfp + fp) = fak * H_i.z[k];
		}
	    }
	}
    }
    return geo_factor;
}

