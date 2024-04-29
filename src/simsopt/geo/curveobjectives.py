from deprecated import deprecated

import numpy as np
from jax import grad
import jax.numpy as jnp

from .jit import jit
from .._core.types import RealArray
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec, Derivative
import simsoptpp as sopp


__all__ = ['CurveLength', 'LpCurveCurvature', 'LpCurveTorsion',
           'CurveCurveDistance', 'CurveSurfaceDistance', 'ArclengthVariation',
           'MeanSquaredCurvature', 'LinkingNumber', 'MeanMajorRadius', 'WeaveLaneMinorRadius']


@jit
def curve_length_pure(l):
    """
    This function is used in a Python+Jax implementation of the curve length formula.
    """
    return jnp.mean(l)


class CurveLength(Optimizable):
    r"""
    CurveLength is a class that computes the length of a curve, i.e.

    .. math::
        J = \int_{\text{curve}}~dl.

    """

    def __init__(self, curve):
        self.curve = curve
        self.thisgrad = jit(lambda l: grad(curve_length_pure)(l))
        super().__init__(depends_on=[curve])

    def J(self):
        """
        This returns the value of the quantity.
        """
        return curve_length_pure(self.curve.incremental_arclength())

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """

        return self.curve.dincremental_arclength_by_dcoeff_vjp(
            self.thisgrad(self.curve.incremental_arclength()))

    return_fn_map = {'J': J, 'dJ': dJ}


@jit
def Lp_curvature_pure(kappa, gammadash, p, desired_kappa):
    """
    This function is used in a Python+Jax implementation of the curvature penalty term.
    """
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    return (1./p)*jnp.mean(jnp.maximum(kappa-desired_kappa, 0)**p * arc_length)


class LpCurveCurvature(Optimizable):
    r"""
    This class computes a penalty term based on the :math:`L_p` norm
    of the curve's curvature, and penalizes where the local curve curvature exceeds a threshold

    .. math::
        J = \frac{1}{p} \int_{\text{curve}} \text{max}(\kappa - \kappa_0, 0)^p ~dl

    where :math:`\kappa_0` is a threshold curvature, given by the argument ``threshold``.
    """

    def __init__(self, curve, p, threshold=0.0):
        self.curve = curve
        self.p = p
        self.threshold = threshold
        super().__init__(depends_on=[curve])
        self.J_jax = jit(lambda kappa, gammadash: Lp_curvature_pure(kappa, gammadash, p, threshold))
        self.thisgrad0 = jit(lambda kappa, gammadash: grad(self.J_jax, argnums=0)(kappa, gammadash))
        self.thisgrad1 = jit(lambda kappa, gammadash: grad(self.J_jax, argnums=1)(kappa, gammadash))

    def J(self):
        """
        This returns the value of the quantity.
        """
        return self.J_jax(self.curve.kappa(), self.curve.gammadash())

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        grad0 = self.thisgrad0(self.curve.kappa(), self.curve.gammadash())
        grad1 = self.thisgrad1(self.curve.kappa(), self.curve.gammadash())
        return self.curve.dkappa_by_dcoeff_vjp(grad0) + self.curve.dgammadash_by_dcoeff_vjp(grad1)

    return_fn_map = {'J': J, 'dJ': dJ}


@jit
def Lp_torsion_pure(torsion, gammadash, p, threshold):
    """
    This function is used in a Python+Jax implementation of the formula for the torsion penalty term.
    """
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    return (1./p)*jnp.mean(jnp.maximum(jnp.abs(torsion)-threshold, 0)**p * arc_length)


class LpCurveTorsion(Optimizable):
    r"""
    LpCurveTorsion is a class that computes a penalty term based on the :math:`L_p` norm
    of the curve's torsion:

    .. math::
        J = \frac{1}{p} \int_{\text{curve}} \max(|\tau|-\tau_0, 0)^p ~dl.

    """

    def __init__(self, curve, p, threshold=0.0):
        self.curve = curve
        self.p = p
        self.threshold = threshold
        super().__init__(depends_on=[curve])
        self.J_jax = jit(lambda torsion, gammadash: Lp_torsion_pure(torsion, gammadash, p, threshold))
        self.thisgrad0 = jit(lambda torsion, gammadash: grad(self.J_jax, argnums=0)(torsion, gammadash))
        self.thisgrad1 = jit(lambda torsion, gammadash: grad(self.J_jax, argnums=1)(torsion, gammadash))

    def J(self):
        """
        This returns the value of the quantity.
        """
        return self.J_jax(self.curve.torsion(), self.curve.gammadash())

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        grad0 = self.thisgrad0(self.curve.torsion(), self.curve.gammadash())
        grad1 = self.thisgrad1(self.curve.torsion(), self.curve.gammadash())
        return self.curve.dtorsion_by_dcoeff_vjp(grad0) + self.curve.dgammadash_by_dcoeff_vjp(grad1)

    return_fn_map = {'J': J, 'dJ': dJ}


def cc_distance_pure(gamma1, l1, gamma2, l2, minimum_distance):
    """
    This function is used in a Python+Jax implementation of the curve-curve distance formula.
    """
    dists = jnp.sqrt(jnp.sum((gamma1[:, None, :] - gamma2[None, :, :])**2, axis=2))
    alen = jnp.linalg.norm(l1, axis=1)[:, None] * jnp.linalg.norm(l2, axis=1)[None, :]
    return jnp.sum(alen * jnp.maximum(minimum_distance-dists, 0)**2)/(gamma1.shape[0]*gamma2.shape[0])


class CurveCurveDistance(Optimizable):
    r"""
    CurveCurveDistance is a class that computes

    .. math::
        J = \sum_{i = 1}^{\text{num_coils}} \sum_{j = 1}^{i-1} d_{i,j}

    where 

    .. math::
        d_{i,j} = \int_{\text{curve}_i} \int_{\text{curve}_j} \max(0, d_{\min} - \| \mathbf{r}_i - \mathbf{r}_j \|_2)^2 ~dl_j ~dl_i\\

    and :math:`\mathbf{r}_i`, :math:`\mathbf{r}_j` are points on coils :math:`i` and :math:`j`, respectively.
    :math:`d_\min` is a desired threshold minimum intercoil distance.  This penalty term is zero when the points on coil :math:`i` and 
    coil :math:`j` lie more than :math:`d_\min` away from one another, for :math:`i, j \in \{1, \cdots, \text{num_coils}\}`

    If num_basecurves is passed, then the code only computes the distance to
    the first `num_basecurves` many curves, which is useful when the coils
    satisfy symmetries that can be exploited.

    """

    def __init__(self, curves, minimum_distance, num_basecurves=None):
        self.curves = curves
        self.minimum_distance = minimum_distance

        self.J_jax = jit(lambda gamma1, l1, gamma2, l2: cc_distance_pure(gamma1, l1, gamma2, l2, minimum_distance))
        self.thisgrad0 = jit(lambda gamma1, l1, gamma2, l2: grad(self.J_jax, argnums=0)(gamma1, l1, gamma2, l2))
        self.thisgrad1 = jit(lambda gamma1, l1, gamma2, l2: grad(self.J_jax, argnums=1)(gamma1, l1, gamma2, l2))
        self.thisgrad2 = jit(lambda gamma1, l1, gamma2, l2: grad(self.J_jax, argnums=2)(gamma1, l1, gamma2, l2))
        self.thisgrad3 = jit(lambda gamma1, l1, gamma2, l2: grad(self.J_jax, argnums=3)(gamma1, l1, gamma2, l2))
        self.candidates = None
        self.num_basecurves = num_basecurves or len(curves)
        super().__init__(depends_on=curves)

    def recompute_bell(self, parent=None):
        self.candidates = None

    def compute_candidates(self):
        if self.candidates is None:
            candidates = sopp.get_pointclouds_closer_than_threshold_within_collection(
                [c.gamma() for c in self.curves], self.minimum_distance, self.num_basecurves)
            self.candidates = candidates

    def shortest_distance_among_candidates(self):
        self.compute_candidates()
        from scipy.spatial.distance import cdist
        return min([self.minimum_distance] + [np.min(cdist(self.curves[i].gamma(), self.curves[j].gamma())) for i, j in self.candidates])

    def shortest_distance(self):
        self.compute_candidates()
        if len(self.candidates) > 0:
            return self.shortest_distance_among_candidates()
        from scipy.spatial.distance import cdist
        return min([np.min(cdist(self.curves[i].gamma(), self.curves[j].gamma())) for i in range(len(self.curves)) for j in range(i)])

    def J(self):
        """
        This returns the value of the quantity.
        """
        self.compute_candidates()
        res = 0
        for i, j in self.candidates:
            gamma1 = self.curves[i].gamma()
            l1 = self.curves[i].gammadash()
            gamma2 = self.curves[j].gamma()
            l2 = self.curves[j].gammadash()
            res += self.J_jax(gamma1, l1, gamma2, l2)

        return res

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        self.compute_candidates()
        dgamma_by_dcoeff_vjp_vecs = [np.zeros_like(c.gamma()) for c in self.curves]
        dgammadash_by_dcoeff_vjp_vecs = [np.zeros_like(c.gammadash()) for c in self.curves]

        for i, j in self.candidates:
            gamma1 = self.curves[i].gamma()
            l1 = self.curves[i].gammadash()
            gamma2 = self.curves[j].gamma()
            l2 = self.curves[j].gammadash()
            dgamma_by_dcoeff_vjp_vecs[i] += self.thisgrad0(gamma1, l1, gamma2, l2)
            dgammadash_by_dcoeff_vjp_vecs[i] += self.thisgrad1(gamma1, l1, gamma2, l2)
            dgamma_by_dcoeff_vjp_vecs[j] += self.thisgrad2(gamma1, l1, gamma2, l2)
            dgammadash_by_dcoeff_vjp_vecs[j] += self.thisgrad3(gamma1, l1, gamma2, l2)

        res = [self.curves[i].dgamma_by_dcoeff_vjp(dgamma_by_dcoeff_vjp_vecs[i]) + self.curves[i].dgammadash_by_dcoeff_vjp(dgammadash_by_dcoeff_vjp_vecs[i]) for i in range(len(self.curves))]
        return sum(res)

    return_fn_map = {'J': J, 'dJ': dJ}


def cs_distance_pure(gammac, lc, gammas, ns, minimum_distance):
    """
    This function is used in a Python+Jax implementation of the curve-surface distance
    formula.
    """
    dists = jnp.sqrt(jnp.sum(
        (gammac[:, None, :] - gammas[None, :, :])**2, axis=2))
    integralweight = jnp.linalg.norm(lc, axis=1)[:, None] \
        * jnp.linalg.norm(ns, axis=1)[None, :]
    return jnp.mean(integralweight * jnp.maximum(minimum_distance-dists, 0)**2)


class CurveSurfaceDistance(Optimizable):
    r"""
    CurveSurfaceDistance is a class that computes

    .. math::
        J = \sum_{i = 1}^{\text{num_coils}} d_{i}

    where

    .. math::
        d_{i} = \int_{\text{curve}_i} \int_{surface} \max(0, d_{\min} - \| \mathbf{r}_i - \mathbf{s} \|_2)^2 ~dl_i ~ds\\

    and :math:`\mathbf{r}_i`, :math:`\mathbf{s}` are points on coil :math:`i`
    and the surface, respectively. :math:`d_\min` is a desired threshold
    minimum coil-to-surface distance.  This penalty term is zero when the
    points on all coils :math:`i` and on the surface lie more than
    :math:`d_\min` away from one another.

    """

    def __init__(self, curves, surface, minimum_distance):
        self.curves = curves
        self.surface = surface
        self.minimum_distance = minimum_distance

        self.J_jax = jit(lambda gammac, lc, gammas, ns: cs_distance_pure(gammac, lc, gammas, ns, minimum_distance))
        self.thisgrad0 = jit(lambda gammac, lc, gammas, ns: grad(self.J_jax, argnums=0)(gammac, lc, gammas, ns))
        self.thisgrad1 = jit(lambda gammac, lc, gammas, ns: grad(self.J_jax, argnums=1)(gammac, lc, gammas, ns))
        self.candidates = None
        super().__init__(depends_on=curves)  # Bharat's comment: Shouldn't we add surface here

    def recompute_bell(self, parent=None):
        self.candidates = None

    def compute_candidates(self):
        if self.candidates is None:
            candidates = sopp.get_pointclouds_closer_than_threshold_between_two_collections(
                [c.gamma() for c in self.curves], [self.surface.gamma().reshape((-1, 3))], self.minimum_distance)
            self.candidates = candidates

    def shortest_distance_among_candidates(self):
        self.compute_candidates()
        from scipy.spatial.distance import cdist
        xyz_surf = self.surface.gamma().reshape((-1, 3))
        return min([self.minimum_distance] + [np.min(cdist(self.curves[i].gamma(), xyz_surf)) for i, _ in self.candidates])

    def shortest_distance(self):
        self.compute_candidates()
        if len(self.candidates) > 0:
            return self.shortest_distance_among_candidates()
        from scipy.spatial.distance import cdist
        xyz_surf = self.surface.gamma().reshape((-1, 3))
        return min([np.min(cdist(self.curves[i].gamma(), xyz_surf)) for i in range(len(self.curves))])

    def J(self):
        """
        This returns the value of the quantity.
        """
        self.compute_candidates()
        res = 0
        gammas = self.surface.gamma().reshape((-1, 3))
        ns = self.surface.normal().reshape((-1, 3))
        for i, _ in self.candidates:
            gammac = self.curves[i].gamma()
            lc = self.curves[i].gammadash()
            res += self.J_jax(gammac, lc, gammas, ns)
        return res

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        self.compute_candidates()
        dgamma_by_dcoeff_vjp_vecs = [np.zeros_like(c.gamma()) for c in self.curves]
        dgammadash_by_dcoeff_vjp_vecs = [np.zeros_like(c.gammadash()) for c in self.curves]
        gammas = self.surface.gamma().reshape((-1, 3))

        gammas = self.surface.gamma().reshape((-1, 3))
        ns = self.surface.normal().reshape((-1, 3))
        for i, _ in self.candidates:
            gammac = self.curves[i].gamma()
            lc = self.curves[i].gammadash()
            dgamma_by_dcoeff_vjp_vecs[i] += self.thisgrad0(gammac, lc, gammas, ns)
            dgammadash_by_dcoeff_vjp_vecs[i] += self.thisgrad1(gammac, lc, gammas, ns)
        res = [self.curves[i].dgamma_by_dcoeff_vjp(dgamma_by_dcoeff_vjp_vecs[i]) + self.curves[i].dgammadash_by_dcoeff_vjp(dgammadash_by_dcoeff_vjp_vecs[i]) for i in range(len(self.curves))]
        
        return sum(res)

    return_fn_map = {'J': J, 'dJ': dJ}


@jit
def curve_arclengthvariation_pure(l, mat):
    """
    This function is used in a Python+Jax implementation of the curve arclength variation.
    """
    return jnp.var(mat @ l)


class ArclengthVariation(Optimizable):

    def __init__(self, curve, nintervals="full"):
        r"""
        This class penalizes variation of the arclength along a curve.
        The idea of this class is to avoid ill-posedness of curve objectives due to
        non-uniqueness of the underlying parametrization. Essentially we want to
        achieve constant arclength along the curve. Since we can not expect
        perfectly constant arclength along the entire curve, this class has
        some support to relax this notion. Consider a partition of the :math:`[0, 1]`
        interval into intervals :math:`\{I_i\}_{i=1}^L`, and tenote the average incremental arclength
        on interval :math:`I_i` by :math:`\ell_i`. This objective then penalises the variance

        .. math::
            J = \mathrm{Var}(\ell_i)

        it remains to choose the number of intervals :math:`L` that :math:`[0, 1]` is split into.
        If ``nintervals="full"``, then the number of intervals :math:`L` is equal to the number of quadrature
        points of the curve. If ``nintervals="partial"``, then the argument is as follows:

        A curve in 3d space is defined uniquely by an initial point, an initial
        direction, and the arclength, curvature, and torsion along the curve. For a
        :mod:`simsopt.geo.curvexyzfourier.CurveXYZFourier`, the intuition is now as
        follows: assuming that the curve has order :math:`p`, that means we have
        :math:`3*(2p+1)` degrees of freedom in total. Assuming that three each are
        required for both the initial position and direction, :math:`6p-3` are left
        over for curvature, torsion, and arclength. We want to fix the arclength,
        so we can afford :math:`2p-1` constraints, which corresponds to
        :math:`L=2p`.

        Finally, the user can also provide an integer value for `nintervals`
        and thus specify the number of intervals directly.
        """
        super().__init__(depends_on=[curve])

        assert nintervals in ["full", "partial"] \
            or (isinstance(nintervals, int) and 0 < nintervals <= curve.gamma().shape[0])
        self.curve = curve
        nquadpoints = len(curve.quadpoints)
        if nintervals == "full":
            nintervals = curve.gamma().shape[0]
        elif nintervals == "partial":
            from simsopt.geo.curvexyzfourier import CurveXYZFourier, JaxCurveXYZFourier
            if isinstance(curve, CurveXYZFourier) or isinstance(curve, JaxCurveXYZFourier):
                nintervals = 2*curve.order
            else:
                raise RuntimeError("Please provide a value other than `partial` for `nintervals`. We only have a default for `CurveXYZFourier` and `JaxCurveXYZFourier`.")

        self.nintervals = nintervals
        indices = np.floor(np.linspace(0, nquadpoints, nintervals+1, endpoint=True)).astype(int)
        mat = np.zeros((nintervals, nquadpoints))
        for i in range(nintervals):
            mat[i, indices[i]:indices[i+1]] = 1/(indices[i+1]-indices[i])
        self.mat = mat
        self.thisgrad = jit(lambda l: grad(lambda x: curve_arclengthvariation_pure(x, mat))(l))

    def J(self):
        return float(curve_arclengthvariation_pure(self.curve.incremental_arclength(), self.mat))

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        return self.curve.dincremental_arclength_by_dcoeff_vjp(
            self.thisgrad(self.curve.incremental_arclength()))

    return_fn_map = {'J': J, 'dJ': dJ}


@jit
def curve_msc_pure(kappa, gammadash):
    """
    This function is used in a Python+Jax implementation of the mean squared curvature objective.
    """
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    return jnp.mean(kappa**2 * arc_length)/jnp.mean(arc_length)


class MeanSquaredCurvature(Optimizable):

    def __init__(self, curve):
        r"""
        Compute the mean of the squared curvature of a curve.

        .. math::
            J = (1/L) \int_{\text{curve}} \kappa^2 ~dl

        where :math:`L` is the curve length, :math:`\ell` is the incremental
        arclength, and :math:`\kappa` is the curvature.

        Args:
            curve: the curve of which the curvature should be computed.
        """
        super().__init__(depends_on=[curve])
        self.curve = curve
        self.thisgrad0 = jit(lambda kappa, gammadash: grad(curve_msc_pure, argnums=0)(kappa, gammadash))
        self.thisgrad1 = jit(lambda kappa, gammadash: grad(curve_msc_pure, argnums=1)(kappa, gammadash))

    def J(self):
        return float(curve_msc_pure(self.curve.kappa(), self.curve.gammadash()))

    @derivative_dec
    def dJ(self):
        grad0 = self.thisgrad0(self.curve.kappa(), self.curve.gammadash())
        grad1 = self.thisgrad1(self.curve.kappa(), self.curve.gammadash())
        return self.curve.dkappa_by_dcoeff_vjp(grad0) + self.curve.dgammadash_by_dcoeff_vjp(grad1)


@deprecated("`MinimumDistance` has been deprecated and will be removed. Please use `CurveCurveDistance` instead.")
class MinimumDistance(CurveCurveDistance):
    pass


class LinkingNumber(Optimizable):

    def __init__(self, curves, downsample=1):
        Optimizable.__init__(self, depends_on=curves)
        self.curves = curves
        for curve in curves:
            assert np.mod(len(curve.quadpoints), downsample) == 0, f"Downsample {downsample} does not divide the number of quadpoints {len(curve.quadpoints)}."

        self.downsample = downsample
        self.dphis = np.array([(c.quadpoints[1] - c.quadpoints[0]) * downsample for c in self.curves])

        r"""
        Compute the Gauss linking number of a set of curves, i.e. whether the curves
        are interlocked or not.

        The value is an integer, >= 1 if the curves are interlocked, 0 if not. For each pair
        of curves, the contribution to the linking number is
        
        .. math::
            Link(c_1, c_2) = \frac{1}{4\pi} \left| \oint_{c_1}\oint_{c_2}\frac{\textbf{r}_1 - \textbf{r}_2}{|\textbf{r}_1 - \textbf{r}_2|^3} (d\textbf{r}_1 \times d\textbf{r}_2) \right|
            
        where :math:`c_1` is the first curve, :math:`c_2` is the second curve,
        :math:`\textbf{r}_1` is the position vector along the first curve, and
        :math:`\textbf{r}_2` is the position vector along the second curve.

        Args:
            curves: the set of curves for which the linking number should be computed.
            downsample: integer factor by which to downsample the quadrature
                points when computing the linking number. Setting this parameter to
                a value larger than 1 will speed up the calculation, which may
                be useful if the set of coils is large, though it may introduce
                inaccuracy if ``downsample`` is set too large.
        """

    def J(self):
        return sopp.compute_linking_number(
            [c.gamma() for c in self.curves],
            [c.gammadash() for c in self.curves],
            self.dphis,
            self.downsample,
        )

    @derivative_dec
    def dJ(self):
        return Derivative({})
    
def gradfunction(x0, order, curves, n):
    grad = np.zeros(len(x0))
    c_dof_len = 3*(2*order+1)
    for i in range(len(curves)):
        denominator = (n*np.sqrt(x0[n-1+i*c_dof_len]**2 + x0[2*order+n+c_dof_len*i]**2))
        grad[n-1+i*c_dof_len] = (x0[n-1+i*c_dof_len]/denominator)
        grad[2*order+n+c_dof_len*i] = (x0[2*order+n+c_dof_len*i]/denominator)
    #print("Derivative MMR:", grad)
    return(grad)


class MeanMajorRadius(Optimizable):

    def __init__(self, curve, target = 0,
                 mode_order = 6):
        Optimizable.__init__(self, depends_on=[curve])
        self.curve = curve
        self.order = mode_order
        self.target = target
        

    def J(self):
        dofs = self.curve.full_x
        cos_radius = dofs[0]
        sin_radius = dofs[2*self.order+1]
        major_radius = np.sqrt(cos_radius**2+sin_radius**2)
        return np.abs(major_radius - self.target)
    

    @derivative_dec
    def dJ(self):
        return self.curve.gradMajorRadius(self.order, self.target) 
            

class WeaveLaneMinorRadius(Optimizable):
        
    def __init__(self, curve, target = 0,
                 mode_order = 6):
        Optimizable.__init__(self, depends_on=[curve])
        self.curve = curve
        self.order = mode_order
        self.target = target
        

    def J(self):
        dofs = self.curve.full_x
        cos_minor = dofs[1]
        sin_minor = dofs[2*self.order+2]
        minor_radius = np.sqrt(cos_minor**2+sin_minor**2)
        return np.abs(minor_radius - self.target)
    

    @derivative_dec
    def dJ(self):
        return self.curve.gradWLMinorRadius(self.order, self.target) 
            


