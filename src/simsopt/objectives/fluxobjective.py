import numpy as np

import simsoptpp as sopp
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec


__all__ = ['SquaredFlux', 'SquaredFluxMax']


class SquaredFlux(Optimizable):

    r"""
    Objective representing quadratic-flux-like quantities, useful for stage-2
    coil optimization. Several variations are available, which can be selected
    using the ``definition`` argument. For ``definition="quadratic flux"`` 
    (the default), the objective is defined as

    .. math::
        J = \frac12 \int_{S} (\mathbf{B}\cdot \mathbf{n} - B_T)^2 ds,

    where :math:`\mathbf{n}` is the surface unit normal vector and
    :math:`B_T` is an optional (zero by default) target value for the
    magnetic field. Also :math:`\int_{S} ds` indicates a surface integral.
    For ``definition="normalized"``, the objective is defined as

    .. math::
        J = \frac12 \frac{\int_{S} (\mathbf{B}\cdot \mathbf{n} - B_T)^2 ds}
                         {\int_{S} |\mathbf{B}|^2 ds}.

    For ``definition="local"``, the objective is defined as

    .. math::
        J = \frac12 \int_{S} \frac{(\mathbf{B}\cdot \mathbf{n} - B_T)^2}{|\mathbf{B}|^2} ds.

    The definition ``"quadratic flux"`` has the advantage of simplicity, and it
    is used in other contexts such as REGCOIL. However for stage-2 optimization,
    the optimizer can "cheat", lowering this objective by reducing the magnitude
    of the field. The definitions ``"normalized"`` and ``"local"`` close this loophole.

    Args:
        surface: A :obj:`simsopt.geo.surface.Surface` object on which to compute the flux
        field: A :obj:`simsopt.field.magneticfield.MagneticField` for which to compute the flux.
        target: A ``nphi x ntheta`` numpy array containing target values for the flux. Here 
          ``nphi`` and ``ntheta`` correspond to the number of quadrature points on `surface` 
          in ``phi`` and ``theta`` direction.
        definition: A string to select among the definitions above. The
          available options are ``"quadratic flux"``, ``"normalized"``, and ``"local"``.
    """

    def __init__(self, surface, field, target=None, definition="quadratic flux"):
        self.surface = surface
        if target is not None:
            self.target = np.ascontiguousarray(target)
        else:
            self.target = np.zeros(self.surface.normal().shape[:2])
        self.field = field
        xyz = self.surface.gamma()
        self.field.set_points(xyz.reshape((-1, 3)))
        if definition not in ["quadratic flux", "normalized", "local"]:
            raise ValueError("Unrecognized option for 'definition'.")
        self.definition = definition
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[field])

    def J(self):
        n = self.surface.normal()
        Bcoil = self.field.B().reshape(n.shape)
        return sopp.integral_BdotN(Bcoil, self.target, n, self.definition)

    @derivative_dec
    def dJ(self):
        n = self.surface.normal()
        absn = np.linalg.norm(n, axis=2)
        unitn = n * (1. / absn)[:, :, None]
        Bcoil = self.field.B().reshape(n.shape)
        Bcoil_n = np.sum(Bcoil * unitn, axis=2)
        if self.target is not None:
            B_n = (Bcoil_n - self.target)
        else:
            B_n = Bcoil_n

        if self.definition == "quadratic flux":
            dJdB = (B_n[..., None] * unitn * absn[..., None]) / absn.size
            dJdB = dJdB.reshape((-1, 3))

        elif self.definition == "local":
            mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
            dJdB = ((
                (B_n/mod_Bcoil)[..., None] * (
                    unitn / mod_Bcoil[..., None] - (B_n / mod_Bcoil**3)[..., None] * Bcoil
                )) * absn[..., None]) / absn.size

        elif self.definition == "normalized":
            mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
            num = np.mean(B_n**2 * absn)
            denom = np.mean(mod_Bcoil**2 * absn)

            dnum = 2 * (B_n[..., None] * unitn * absn[..., None]) / absn.size
            ddenom = 2 * (Bcoil * absn[..., None]) / absn.size
            dJdB = 0.5 * (dnum / denom - num * ddenom / denom**2)

        else:
            raise ValueError("Should never get here")

        dJdB = dJdB.reshape((-1, 3))
        return self.field.B_vjp(dJdB)


class SquaredFluxMax(Optimizable):

    r"""
    Objective representing the maximum of quadratic-flux-like quantities. 
    This is typically more fragile than SquaredFlux, but possibly useful for 
    refinement. Several variations are available, which can be selected
    using the ``definition`` argument. For ``definition="quadratic flux"`` 
    (the default), the objective is defined as

    .. math::
        J = \max( (\mathbf{B}\cdot \mathbf{n} )^2 ),

    where :math:`\mathbf{n}` is the surface unit normal vector and
    :math:`B_T` is an optional (zero by default) target value for the
    magnetic field. 
    For ``definition="normalized"``, the objective is defined as

    .. math::
        J = \max( \frac{ (\mathbf{B}\cdot \mathbf{n} )^2}
                         {\int_{S} |\mathbf{B}|^2 ds} ).

    For ``definition="local"``, the objective is defined as

    .. math::
        J = \max( \frac{(\mathbf{B}\cdot \mathbf{n} )^2}{|\mathbf{B}|^2} ).

    The definition ``"quadratic flux"`` has the advantage of simplicity, and it
    is used in other contexts such as REGCOIL. However for stage-2 optimization,
    the optimizer can "cheat", lowering this objective by reducing the magnitude
    of the field. The definitions ``"normalized"`` and ``"local"`` close this loophole.

    Args:
        surface: A :obj:`simsopt.geo.surface.Surface` object on which to compute the flux
        field: A :obj:`simsopt.field.magneticfield.MagneticField` for which to compute the flux.
        target: A single target value for the maximum flux.
        definition: A string to select among the definitions above. The
          available options are ``"quadratic flux"``, ``"normalized"``, and ``"local"``.
        coverage: The fraction of surface points to include in optimization. The default 0.1
          will find the 10% of surface points with the largest squared flux.
    """

    def __init__(self, surface, field, target=None, coverage=None, definition="quadratic flux"):
        self.surface = surface
        if target is not None:
            self.target = target
        else:
            self.target = None
        self.field = field
        xyz = self.surface.gamma()
        self.field.set_points(xyz.reshape((-1, 3)))
        if definition not in ["quadratic flux", "normalized", "local"]:
            raise ValueError("Unrecognized option for 'definition'.")
        self.definition = definition
        if coverage is None:
            coverage = 0.1
        self.coverage_index =  int(np.round( xyz.size//3 * coverage ))
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[field])

    def maxBn(self):
        xyz = self.surface.gamma()
        self.field.set_points(xyz.reshape((-1,3)))
        n = self.surface.normal()
        absn = np.linalg.norm(n, axis=2)
        unitn = n * (1. / absn)[:, :, None]
        Bcoil = self.field.B().reshape(n.shape)
        Bcoil_n = np.absolute(np.sum(Bcoil * unitn, axis=2))
        if self.target is not None:
            B_n = np.maximum(Bcoil_n - self.target, np.zeros(Bcoil_n.shape))
        else:
            B_n = Bcoil_n

        if self.definition == "quadratic flux":
            Bn2 = B_n**2

        elif self.definition == "local":
            mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
            Bn2 = B_n**2 / mod_Bcoil**2

        elif self.definition == "normalized":
            mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
            Bn2 = B_n**2 / np.mean(mod_Bcoil**2 * absn)

        else:
            raise ValueError("Should never get here")

        loc_max = np.argpartition(Bn2, -self.coverage_index, axis=None)
        Bn2_max = Bn2.reshape((-1,1))[loc_max[-self.coverage_index:]]
        return Bn2_max, loc_max

    def J(self):
        Bn2_max, loc_max = self.maxBn()
        Bn2 = np.sum(Bn2_max) / Bn2_max.size
        return np.squeeze(Bn2)

    @derivative_dec
    def dJ(self):
        Bn2_max, loc_max = self.maxBn()
        n = self.surface.normal()
        absn = np.linalg.norm(n, axis=2)
        unitn = n * (1. / absn)[:, :, None]
        Bcoil = self.field.B().reshape(n.shape)
        B_n = np.sum(Bcoil * unitn, axis=2)
        B_n[np.unravel_index(loc_max[:-self.coverage_index], B_n.shape)] = 0

        if self.definition == "quadratic flux":
            dJdB = 2 * (B_n[..., None] * unitn * absn[..., None]) / absn.size

        elif self.definition == "local":
            mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
            dJdB = ((
                (2 * B_n/mod_Bcoil)[..., None] * (
                    unitn / mod_Bcoil[..., None] - (B_n / mod_Bcoil**3)[..., None] * Bcoil
                )) * absn[..., None]) / absn.size

        elif self.definition == "normalized":
            mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
            num = np.mean(B_n**2 * absn)
            denom = np.mean(mod_Bcoil**2 * absn)

            dnum = 2 * (B_n[..., None] * unitn * absn[..., None]) / absn.size
            ddenom = 2 * (Bcoil * absn[..., None]) / absn.size
            dJdB = (dnum / denom - num * ddenom / denom**2)

        else:
            raise ValueError("Should never get here")

        dJdB = dJdB.reshape((-1, 3))
        return self.field.B_vjp(dJdB)

