from dataclasses import dataclass

import numpy as np
from sympy import Symbol, lambdify, exp

from .._core.json import GSONable
from .._core.util import RealArray

import simsoptpp as sopp
from simsopt.geo.curve import Curve

__all__ = ['GaussianSampler', 'GaussianSamplerInformed', 'PerturbationSample', 'CurvePerturbed', 
           'RotationSample', 'AssembledCurve', 'CurvePerturbedMix', 'GaussianSampler1D', 'CurveFromRegcoil']


@dataclass
class GaussianSampler(GSONable):
    r"""
    Generate a periodic gaussian process on the interval [0, 1] on a given list of quadrature points.
    The process has standard deviation ``sigma`` a correlation length scale ``length_scale``.
    Large values of ``length_scale`` correspond to smooth processes, small values result in highly oscillatory
    functions.
    Also has the ability to sample the derivatives of the function.

    We consider the kernel

    .. math::

        \kappa(d) = \sigma^2 \exp(-d^2/l^2)

    and then consider a Gaussian process with covariance

    .. math::

        Cov(X(s), X(t)) = \sum_{i=-\infty}^\infty \sigma^2 \exp(-(s-t+i)^2/l^2)

    the sum is used to make the kernel periodic and in practice the infinite sum is truncated.

    Args:
        points: the quadrature points along which the perturbation should be computed.
        sigma: standard deviation of the underlying gaussian process
               (measure for the magnitude of the perturbation).
        length_scale: length scale of the underlying gaussian process
                      (measure for the smoothness of the perturbation).
        n_derivs: number of derivatives of the gaussian process to sample.
    """

    points: RealArray
    sigma: float
    length_scale: float
    n_derivs: int = 1

    def __post_init__(self):
        xs = self.points
        n = len(xs)
        cov_mat = np.zeros((n*(self.n_derivs+1), n*(self.n_derivs+1)))

        def kernel(x, y):
            return sum((self.sigma**2)*exp(-(x-y+i)**2/(self.length_scale)**2) for i in range(-5, 6))

        XX, YY = np.meshgrid(xs, xs, indexing='ij')
        x = Symbol("x")
        y = Symbol("y")
        f = kernel(x, y)
        # print("Type f:", type(f))
        for ii in range(self.n_derivs+1):
            for jj in range(self.n_derivs+1): 
                if ii + jj == 0:
                    lam = lambdify((x, y), f, "numpy")
                else:
                    lam = lambdify((x, y), f.diff(*(ii * [x] + jj * [y])), "numpy")
                cov_mat[(ii*n):((ii+1)*n), (jj*n):((jj+1)*n)] = lam(XX, YY)
        self.cov_mat = cov_mat
        # we need to compute the sqrt of the covariance matrix. we used to do this using scipy.linalg.sqrtm,
        # but it seems sometime  scipy 1.11.1 and 1.11.2 that function broke/changed behaviour.
        # So we use a LDLT decomposition instead. See als https://github.com/hiddenSymmetries/simsopt/issues/349
        # from scipy.linalg import sqrtm, ldl
        # self.L = np.real(sqrtm(cov_mat))
        from scipy.linalg import ldl
        # self.L = cholesky(cov_mat)
        lu, d, _ = ldl(cov_mat)
        self.d = d
        self.lu = lu
        self.L = lu @ np.sqrt(np.maximum(d, 0))

    def draw_sample(self, randomgen=None):
        """
        Returns a list of ``n_derivs+1`` arrays of size ``(len(points), 3)``, containing the
        perturbation and the derivatives.
        """
        n = len(self.points)
        n_derivs = self.n_derivs
        if randomgen is None:
            randomgen = np.random
        z = randomgen.standard_normal(size=(n*(n_derivs+1), 3))
        curve_and_derivs = self.L@z
        return [curve_and_derivs[(i*n):((i+1)*n), :] for i in range(n_derivs+1)]
    
    def get_spectrum(self):
        n = len(self.points)
        sqrt_d =np.sqrt(np.maximum(self.d, 0))
        return(self.d, self.lu, self.cov_mat)
    
    
@dataclass
class GaussianSamplerInformed(GSONable):
    r"""
    Generate a periodic gaussian process on the interval [0, 1] on a given list of quadrature points.
    The process has standard deviation ``sigma`` a correlation length scale ``length_scale``.
    Large values of ``length_scale`` correspond to smooth processes, small values result in highly oscillatory
    functions.
    Also has the ability to sample the derivatives of the function.

    We consider the kernel

    .. math::

        \kappa(d) = \sigma^2 \exp(-d^2/l^2)

    and then consider a Gaussian process with covariance

    .. math::

        Cov(X(s), X(t)) = \sum_{i=-\infty}^\infty \sigma^2 \exp(-(s-t+i)^2/l^2)

    the sum is used to make the kernel periodic and in practice the infinite sum is truncated.

    Args:
        points: the quadrature points along which the perturbation should be computed.
        sigma: standard deviation of the underlying gaussian process
               (measure for the magnitude of the perturbation).
        length_scale: length scale of the underlying gaussian process
                      (measure for the smoothness of the perturbation).
        n_derivs: number of derivatives of the gaussian process to sample.
    """

    points: RealArray
    signal: RealArray
    signal_weight: float
    product: bool
    sigma: float
    length_scale: float
    n_derivs: int = 1

    def __post_init__(self):
        xs = self.points
        n = len(xs)
        cov_mat = np.zeros((n*(self.n_derivs+1), n*(self.n_derivs+1)))

        def kernel(x, y):
            return sum((self.sigma**2)*exp(-(x-y+i)**2/(self.length_scale**2)) for i in range(-5, 6))

        XX, YY = np.meshgrid(xs, xs, indexing='ij')
        sx_part = np.ones((n,n))*self.signal
        sx = self.signal_weight*self.sigma*sx_part*sx_part.T/(np.max(sx_part)**2)
        sx_dash = np.ones((n,n))*np.gradient(self.signal)
        sx_dashdash = np.ones((n,n))*np.gradient(sx_dash[0])
        x = Symbol("x")
        y = Symbol("y")
        f = kernel(x, y)
        for ii in range(self.n_derivs+1):
            for jj in range(self.n_derivs+1):
                lam = lambdify((x, y), f, "numpy")
                if ii+jj == 0:
                    if self.product == False:
                        cov_mat[(ii*n):((ii+1)*n), (jj*n):((jj+1)*n)] = sx+lam(XX, YY)
                    else:
                        cov_mat[(ii*n):((ii+1)*n), (jj*n):((jj+1)*n)] = sx*lam(XX, YY)
                else:
                    lam = lambdify((x, y), f.diff(*(ii * [x] + jj * [y])), "numpy")
                    cov_mat[(ii*n):((ii+1)*n), (jj*n):((jj+1)*n)] = lam(XX,YY)
        self.cov_mat = cov_mat
        # we need to compute the sqrt of the covariance matrix. we used to do this using scipy.linalg.sqrtm,
        # but it seems sometime  scipy 1.11.1 and 1.11.2 that function broke/changed behaviour.
        # So we use a LDLT decomposition instead. See als https://github.com/hiddenSymmetries/simsopt/issues/349
        # from scipy.linalg import sqrtm, ldl
        # self.L = np.real(sqrtm(cov_mat))
        from scipy.linalg import ldl
        # self.L = cholesky(cov_mat)
        lu, d, _ = ldl(cov_mat)
        self.d = d
        self.lu = lu
        self.L = lu @ np.sqrt(np.maximum(d, 0))

    def draw_sample(self, randomgen=None):
        """
        Returns a list of ``n_derivs+1`` arrays of size ``(len(points), 3)``, containing the
        perturbation and the derivatives.
        """
        n = len(self.points)
        n_derivs = self.n_derivs
        if randomgen is None:
            randomgen = np.random
        z = randomgen.standard_normal(size=(n*(n_derivs+1), 3))
        curve_and_derivs = self.L@z
        return [curve_and_derivs[(i*n):((i+1)*n), :] for i in range(n_derivs+1)]
    
    def get_spectrum(self):
        n = len(self.points)
        sqrt_d =np.sqrt(np.maximum(self.d, 0))
        return(self.d, self.lu, self.cov_mat)


@dataclass
class GaussianSampler1D(GSONable):
    r"""
    Generate a 1D periodic gaussian process on the interval [0, 1] on a given list of quadrature points.
    The process has standard deviation ``sigma`` a correlation length scale ``length_scale``.
    Large values of ``length_scale`` correspond to smooth processes, small values result in highly oscillatory
    functions.
    Also has the ability to sample the derivatives of the function.

    We consider the kernel

    .. math::

        \kappa(d) = \sigma^2 \exp(-d^2/l^2)

    and then consider a Gaussian process with covariance

    .. math::

        Cov(X(s), X(t)) = \sum_{i=-\infty}^\infty \sigma^2 \exp(-(s-t+i)^2/l^2)

    the sum is used to make the kernel periodic and in practice the infinite sum is truncated.

    Args:
        points: the quadrature points along which the perturbation should be computed.
        sigma: standard deviation of the underlying gaussian process
               (measure for the magnitude of the perturbation).
        length_scale: length scale of the underlying gaussian process
                      (measure for the smoothness of the perturbation).
        n_derivs: number of derivatives of the gaussian process to sample.
    """

    points: RealArray
    sigma: float
    length_scale: float
    n_derivs: int = 1

    def __post_init__(self):
        xs = self.points
        n = len(xs)
        cov_mat = np.zeros((n*(self.n_derivs+1), n*(self.n_derivs+1)))

        def kernel(x, y):
            return sum((self.sigma**2)*exp(-(x-y+i)**2/(self.length_scale**2)) for i in range(-5, 6))

        XX, YY = np.meshgrid(xs, xs, indexing='ij')
        x = Symbol("x")
        y = Symbol("y")
        f = kernel(x, y)
        for ii in range(self.n_derivs+1):
            for jj in range(self.n_derivs+1):
                if ii + jj == 0:
                    lam = lambdify((x, y), f, "numpy")
                else:
                    lam = lambdify((x, y), f.diff(*(ii * [x] + jj * [y])), "numpy")
                cov_mat[(ii*n):((ii+1)*n), (jj*n):((jj+1)*n)] = lam(XX, YY)

        # we need to compute the sqrt of the covariance matrix. we used to do this using scipy.linalg.sqrtm,
        # but it seems sometime between scipy 1.11.1 and 1.11.2 that function broke/changed behaviour.
        # So we use a LDLT decomposition instead. See als https://github.com/hiddenSymmetries/simsopt/issues/349
        # from scipy.linalg import sqrtm, ldl
        # self.L = np.real(sqrtm(cov_mat))
        from scipy.linalg import ldl
        lu, d, _ = ldl(cov_mat)
        self.L = lu @ np.sqrt(np.maximum(d, 0))

    def draw_sample(self, randomgen=None):
        """
        Returns a list of ``n_derivs+1`` arrays of size ``(len(points), 3)``, containing the
        perturbation and the derivatives.
        """
        n = len(self.points)
        n_derivs = self.n_derivs
        if randomgen is None:
            randomgen = np.random
        z = randomgen.standard_normal(size=(n*(n_derivs+1), 3))
        curve_and_derivs = self.L@z
        return [curve_and_derivs[(i*n):((i+1)*n), :] for i in range(n_derivs+1)]


class PerturbationSample(GSONable):
    """
    This class represents a single sample of a perturbation.  The point of
    having a dedicated class for this is so that we can apply the same
    perturbation to multipe curves (e.g. in the case of multifilament
    approximations to finite build coils).
    The main way to interact with this class is via the overloaded ``__getitem__``
    (i.e. ``[ ]`` indexing).
    For example::

        sample = PerturbationSample(...)
        g = sample[0] # get the values of the perturbation
        gd = sample[1] # get the first derivative of the perturbation
    """

    def __init__(self, sampler, randomgen=None, sample=None):
        self.sampler = sampler
        self.randomgen = randomgen   # If not None, most likely fail with serialization
        if sample:
            self._sample = sample
        else:
            self.resample()

    def resample(self):
        self._sample = self.sampler.draw_sample(self.randomgen)

    def __getitem__(self, deriv):
        """
        Get the perturbation (if ``deriv=0``) or its ``deriv``-th derivative.
        """
        assert isinstance(deriv, int)
        if deriv >= len(self._sample):
            raise ValueError("""
The sample on has {len(self._sample)-1} derivatives.
Adjust the `n_derivs` parameter of the sampler to access higher derivatives.
""")
        return self._sample[deriv]


class CurvePerturbed(sopp.Curve, Curve):

    """A perturbed curve."""

    def __init__(self, curve, sample):
        r"""
        Perturb a underlying :mod:`simsopt.geo.curve.Curve` object by drawing a perturbation from a
        :obj:`GaussianSampler`.

        Comment:
        Doing anything involving randomness in a reproducible way requires care.
        Even more so, when doing things in parallel.
        Let's say we have a list of :mod:`simsopt.geo.curve.Curve` objects ``curves`` that represent a stellarator,
        and now we want to consider ``N`` perturbed stellarators. Let's also say we have multiple MPI ranks.
        To avoid the same thing happening on the different MPI ranks, we could pick a different seed on each rank.
        However, then we get different results depending on the number of MPI ranks that we run on. Not ideal.
        Instead, we should pick a new seed for each :math:`1\le i\le N`. e.g.

        .. code-block:: python

            from randomgen import SeedSequence, PCG64
            import numpy as np
            curves = ...
            sigma = 0.01
            length_scale = 0.2
            sampler = GaussianSampler(curves[0].quadpoints, sigma, length_scale, n_derivs=1)
            globalseed = 1
            N = 10 # number of perturbed stellarators
            seeds = SeedSequence(globalseed).spawn(N)
            idx_start, idx_end = split_range_between_mpi_rank(N) # e.g. [0, 5) on rank 0, [5, 10) on rank 1
            perturbed_curves = [] # this will be a List[List[Curve]], with perturbed_curves[i] containing the perturbed curves for the i-th stellarator
            for i in range(idx_start, idx_end):
                rg = np.random.Generator(PCG64(seeds_sys[j], inc=0))
                stell = []
                for c in curves:
                    pert = PerturbationSample(sampler_systematic, randomgen=rg)
                    stell.append(CurvePerturbed(c, pert))
                perturbed_curves.append(stell)
        """
        self.curve = curve
        sopp.Curve.__init__(self, curve.quadpoints)
        Curve.__init__(self, depends_on=[curve])
        self.sample = sample

    def resample(self):
        self.sample.resample()
        self.recompute_bell()

    def recompute_bell(self, parent=None):
        self.invalidate_cache()

    def gamma_impl(self, gamma, quadpoints):
        assert quadpoints.shape[0] == self.curve.quadpoints.shape[0]
        assert np.linalg.norm(quadpoints - self.curve.quadpoints) < 1e-15
        gamma[:] = self.curve.gamma() + self.sample[0]

    def gammadash_impl(self, gammadash):
        gammadash[:] = self.curve.gammadash() + self.sample[1]

    def gammadashdash_impl(self, gammadashdash):
        gammadashdash[:] = self.curve.gammadashdash() + self.sample[2]

    def gammadashdashdash_impl(self, gammadashdashdash):
        gammadashdashdash[:] = self.curve.gammadashdashdash() + self.sample[3]

    def dgamma_by_dcoeff_vjp(self, v):
        return self.curve.dgamma_by_dcoeff_vjp(v)

    def dgammadash_by_dcoeff_vjp(self, v):
        return self.curve.dgammadash_by_dcoeff_vjp(v)

    def dgammadashdash_by_dcoeff_vjp(self, v):
        return self.curve.dgammadashdash_by_dcoeff_vjp(v)

    def dgammadashdashdash_by_dcoeff_vjp(self, v):
        return self.curve.dgammadashdashdash_by_dcoeff_vjp(v)
    
    
    
class CurvePerturbedMix(sopp.Curve, Curve):

    """A perturbed curve."""

    def __init__(self, curve, sample1, sample2):
        r"""
        Perturb a underlying :mod:`simsopt.geo.curve.Curve` object by drawing a perturbation from a
        :obj:`GaussianSampler`.

        Comment:
        Doing anything involving randomness in a reproducible way requires care.
        Even more so, when doing things in parallel.
        Let's say we have a list of :mod:`simsopt.geo.curve.Curve` objects ``curves`` that represent a stellarator,
        and now we want to consider ``N`` perturbed stellarators. Let's also say we have multiple MPI ranks.
        To avoid the same thing happening on the different MPI ranks, we could pick a different seed on each rank.
        However, then we get different results depending on the number of MPI ranks that we run on. Not ideal.
        Instead, we should pick a new seed for each :math:`1\le i\le N`. e.g.

        .. code-block:: python

            from randomgen import SeedSequence, PCG64
            import numpy as np
            curves = ...
            sigma = 0.01
            length_scale = 0.2
            sampler = GaussianSampler(curves[0].quadpoints, sigma, length_scale, n_derivs=1)
            globalseed = 1
            N = 10 # number of perturbed stellarators
            seeds = SeedSequence(globalseed).spawn(N)
            idx_start, idx_end = split_range_between_mpi_rank(N) # e.g. [0, 5) on rank 0, [5, 10) on rank 1
            perturbed_curves = [] # this will be a List[List[Curve]], with perturbed_curves[i] containing the perturbed curves for the i-th stellarator
            for i in range(idx_start, idx_end):
                rg = np.random.Generator(PCG64(seeds_sys[j], inc=0))
                stell = []
                for c in curves:
                    pert = PerturbationSample(sampler_systematic, randomgen=rg)
                    stell.append(CurvePerturbed(c, pert))
                perturbed_curves.append(stell)
        """

        self.curve = curve
        sopp.Curve.__init__(self, curve.quadpoints)
        Curve.__init__(self, depends_on=[curve])
        self.sample1 = sample1
        self.sample2 = sample2
        
    def resample(self):
        self.sample.resample()
        self.recompute_bell()

    def recompute_bell(self, parent=None):
        self.invalidate_cache()

    def gamma_impl(self, gamma, quadpoints):
        assert quadpoints.shape[0] == self.curve.quadpoints.shape[0]
        assert np.linalg.norm(quadpoints - self.curve.quadpoints) < 1e-15
        gamma[:] = self.curve.gamma() + self.sample1[0] + self.sample2[0]

    def gammadash_impl(self, gammadash):
        gammadash[:] = self.curve.gammadash() + self.sample1[1] + self.sample2[1]

    def gammadashdash_impl(self, gammadashdash):
        gammadashdash[:] = self.curve.gammadashdash() + self.sample1[2] + self.sample2[2]

    def gammadashdashdash_impl(self, gammadashdashdash):
        gammadashdashdash[:] = self.curve.gammadashdashdash() + self.sample1[3] + self.sample2[3]

    def dgamma_by_dcoeff_vjp(self, v):
        return self.curve.dgamma_by_dcoeff_vjp(v)

    def dgammadash_by_dcoeff_vjp(self, v):
        return self.curve.dgammadash_by_dcoeff_vjp(v)

    def dgammadashdash_by_dcoeff_vjp(self, v):
        return self.curve.dgammadashdash_by_dcoeff_vjp(v)

    def dgammadashdashdash_by_dcoeff_vjp(self, v):
        return self.curve.dgammadashdashdash_by_dcoeff_vjp(v)
    
    
    
class CurveFromRegcoil(sopp.Curve, Curve):

    """A perturbed curve."""

    def __init__(self, curve, regcoil_gamma):
        r"""
            Creates a very basic Curve object from the interpolated Regcoil gamma
        """
        self.curve = curve
        sopp.Curve.__init__(self, curve.quadpoints)
        Curve.__init__(self, depends_on=[curve])
        self.regcoil_gamma = regcoil_gamma
        
    def resample(self):
        self.sample.resample()
        self.recompute_bell()

    def recompute_bell(self, parent=None):
        self.invalidate_cache()

    def gamma_impl(self, gamma, quadpoints):
        assert quadpoints.shape[0] == self.curve.quadpoints.shape[0]
        assert np.linalg.norm(quadpoints - self.curve.quadpoints) < 1e-15
        gamma[:] = self.regcoil_gamma

    def gammadash_impl(self, gammadash):
        gammadash[:] = self.curve.gammadash()

    def gammadashdash_impl(self, gammadashdash):
        gammadashdash[:] = self.curve.gammadashdash()

    def gammadashdashdash_impl(self, gammadashdashdash):
        gammadashdashdash[:] = self.curve.gammadashdashdash()

    def dgamma_by_dcoeff_vjp(self, v):
        return self.curve.dgamma_by_dcoeff_vjp(v)

    def dgammadash_by_dcoeff_vjp(self, v):
        return self.curve.dgammadash_by_dcoeff_vjp(v)

    def dgammadashdash_by_dcoeff_vjp(self, v):
        return self.curve.dgammadashdash_by_dcoeff_vjp(v)

    def dgammadashdashdash_by_dcoeff_vjp(self, v):
        return self.curve.dgammadashdashdash_by_dcoeff_vjp(v)




def RX(theta):
	matrix = np.zeros((3,3))
	matrix[0][0] = 1
	matrix[1][1] = np.cos(theta)
	matrix[1][2] = -np.sin(theta)
	matrix[2][1] = np.sin(theta)
	matrix[2][2] = np.cos(theta)
	return(matrix)

def RY(theta):
	matrix = np.zeros((3,3))
	matrix[0][0] = np.cos(theta)
	matrix[1][1] = 1
	matrix[0][2] = np.sin(theta)
	matrix[2][0] = -np.sin(theta)
	matrix[2][2] = np.cos(theta)
	return(matrix)

def RZ(theta):
	matrix = np.zeros((3,3))
	matrix[2][2] = 1
	matrix[0][0] = np.cos(theta)
	matrix[0][1] = -np.sin(theta)
	matrix[1][0] = np.sin(theta)
	matrix[1][1] = np.cos(theta)
	return(matrix)

def offset_reset_gamma(path, angle):
    rotation1 = RZ(-angle)
    new_path = []
    for i in range(len(path)):
            new_point = np.dot(rotation1,path[i]) 
            new_path.append(new_point)
    return(np.array(new_path))


def center_of_mass(gamma):
    gamma = np.array(gamma)
    n = len(gamma)
    x_com = sum(gamma[:,0])/n
    y_com = sum(gamma[:,1])/n
    z_com = sum(gamma[:,2])/n
    CoM = np.array([x_com,y_com,z_com])
    return(CoM)

def center_of_mass_wp(curves):
    all_CoM = []
    n = len(curves)
    for c in curves:
        gamma = c.gamma()
        all_CoM.append(center_of_mass(gamma))
    all_CoM = np.array(all_CoM)
    x_com = sum(all_CoM[:,0])/n
    y_com = sum(all_CoM[:,1])/n
    z_com = sum(all_CoM[:,2])/n
    CoM_wp = np.array([x_com,y_com,z_com])
    return(CoM_wp)

class RotationSample():
    """ Generates a random 3D rotation matrix from angles given as input.
        ARGUMENTS:
            angles: (3,)-array containing the angles for each of the rotation matrixes rx, ry,rz respectively
            order: order in which the matrixes are ordered when multiplied. """
    
    def __init__(self, order = None):
        if order is None:
            all_compositions = ['XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX']
            self.matrix = all_compositions[random.randint(0,5)]
        else:
            self.matrix = order
        
    
    def __getitem__(self, angles):
        alpha , beta, gamma = angles[0], angles[1], angles[2]
        if self.matrix is None:
            self.matrix = random.randint(1,6)
        Rx, Ry, Rz = RX(alpha), RY(beta), RZ(gamma)
        if self.matrix == 'XYZ':
            rotation = np.dot(Rx, np.dot(Ry,Rz))
        elif self.matrix == 'XZY':
            rotation = np.dot(Rx, np.dot(Rz,Ry))
        elif self.matrix == 'YXZ':
            rotation = np.dot(Ry, np.dot(Rx,Rz))
        elif self.matrix == 'YZX':
            rotation = np.dot(Ry, np.dot(Rz,Rx))
        elif self.matrix == 'ZXY':
            rotation = np.dot(Rz, np.dot(Rx,Ry))
        elif self.matrix == 'ZYX':
            rotation = np.dot(Rz, np.dot(Ry,Rx))
        return(rotation)

def CurveAssembly(gamma, dr, CoM = None, rotmat = None):
    path = np.array(gamma)
    if rotmat is None:
        raise Exception("Error: Rotation matrix missing")
    else:
        rotation = rotmat
    if CoM is None:
        center = center_of_mass(gamma)
    else:
        center = CoM
    angle_offset = np.arctan(center[1]/center[0])
    path = path - center
    reset_path = offset_reset_gamma(path, angle_offset)
    new_path = []
    for i in range(len(path)):
            new_point = np.dot(rotation, reset_path[i])
            new_path.append(new_point)
    final_path = offset_reset_gamma(np.array(new_path), angle=-angle_offset)
    final_path = final_path + center + dr
    return(final_path)	


class AssembledCurve(sopp.Curve, Curve):
    def __init__(self, curve, dr, CoM = None, rotsample = None):
        self.curve = curve
        sopp.Curve.__init__(self, curve.quadpoints)
        Curve.__init__(self, depends_on=[curve])
        self.dr = dr
        self.rotation = rotsample
        if CoM is not None:
            self.com = CoM
            self.path = CurveAssembly(self.curve.gamma(), self.dr, CoM = self.com, rotmat=self.rotation)
        else:
            self.path = CurveAssembly(self.curve.gamma(), self.dr, rotmat=self.rotation)
        self.rotmatT = self.rotation.T.copy()
            
        
    def recompute_bell(self, parent=None):
        self.invalidate_cache()

    def gamma_impl(self, gamma, quadpoints):
        assert quadpoints.shape[0] == self.curve.quadpoints.shape[0]
        assert np.linalg.norm(quadpoints - self.curve.quadpoints) < 1e-15
        gamma[:] = self.path

    def gammadash_impl(self, gammadash):
        gammadash[:] = CurveAssembly(self.curve.gammadash(), self.dr, rotmat=self.rotation)

    def gammadashdash_impl(self, gammadashdash):
        gammadashdash[:] = CurveAssembly(self.curve.gammadashdash(), self.dr, rotmat=self.rotation)

    def gammadashdashdash_impl(self, gammadashdashdash):
        gammadashdashdash[:] = CurveAssembly(self.curve.gammadashdashdash(), self.dr, rotmat=self.rotation)

    def dgamma_by_dcoeff_vjp(self, v):
        v = sopp.matmult(v, self.rotation)
        return self.curve.dgamma_by_dcoeff_vjp(v)

    def dgammadash_by_dcoeff_vjp(self, v):
        v = sopp.matmult(v, self.rotation)
        return self.curve.dgammadash_by_dcoeff_vjp(v)

    def dgammadashdash_by_dcoeff_vjp(self, v):
        v = sopp.matmult(v, self.rotation)
        return self.curve.dgammadashdash_by_dcoeff_vjp(v)

    def dgammadashdashdash_by_dcoeff_vjp(self, v):
        v = sopp.matmult(v, self.rotation)
        return self.curve.dgammadashdashdash_by_dcoeff_vjp(v)
    