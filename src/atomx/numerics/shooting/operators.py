"""Pure radial Dirac operators in the native SLAG6 metric."""
import numpy as np
from numba import njit

from .constants import SL
from .quadrature import slag6

_FD_CENTRAL = (4.0 / 5.0, -1.0 / 5.0, 4.0 / 105.0, -1.0 / 280.0)


def _build_fd_edge_weights():
    W = np.empty((4, 9))
    nodes = np.arange(9, dtype=float)
    rhs = np.zeros(9)
    rhs[1] = 1.0  # Σ_j w_j x_j^m = m! δ_{m,1}
    for i in range(4):
        x = nodes - float(i)
        A = np.vander(x, 9, increasing=True).T
        W[i] = np.linalg.solve(A, rhs)
    return W


_FD_EDGE = _build_fd_edge_weights()


@njit(inline='always')
def _central_derivative(y, i):
    c1, c2, c3, c4 = _FD_CENTRAL
    return (c1*(y[i+1]-y[i-1]) + c2*(y[i+2]-y[i-2])
            + c3*(y[i+3]-y[i-3]) + c4*(y[i+4]-y[i-4]))


@njit(cache=True)
def _ds_derivative(y):
    """dy/ds on the uniform-s grid (unit index spacing), 8th order."""
    n = y.shape[0]
    if n < 9:
        raise ValueError(f"grid too small for 8th-order stencils (n={n})")
    d = np.empty_like(y)
    # Read four independent stencils before storing their outputs. This
    # permits SIMD packing without floating-point reassociation or fastmath;
    # interleaving scalar loads and stores inhibits it in the generic loop.
    stop = 4+4*((n-8)//4)
    for i in range(4, stop, 4):
        d0 = _central_derivative(y, i)
        d1 = _central_derivative(y, i+1)
        d2 = _central_derivative(y, i+2)
        d3 = _central_derivative(y, i+3)
        d[i], d[i+1], d[i+2], d[i+3] = d0, d1, d2, d3
    for i in range(stop, n-4):
        d[i] = _central_derivative(y, i)
    for i in range(4):
        head, tail = 0., 0.
        for j in range(9):
            head += _FD_EDGE[i, j]*y[j]
            tail += _FD_EDGE[i, j]*y[n-1-j]
        d[i] = head
        d[n - 1 - i] = -tail
    return d


@njit(cache=True)
def one_body_element(Pa, Qa, Pb, Qb, kappa, r, difr, rv):
    """Symmetric rest-mass-subtracted Dirac matrix element.

    Fuse pointwise contractions without reassociating arithmetic. Diagonal
    elements reuse derivatives only for identical array views; equal values
    do not imply shared storage. The derivative stencil and SLAG6 summation
    are unchanged, including boundary terms and the origin convention.
    """
    dPb, dQb = _ds_derivative(Pb), _ds_derivative(Qb)
    dPa = (dPb if Pa.ctypes.data == Pb.ctypes.data and Pa.strides == Pb.strides
           else _ds_derivative(Pa))
    dQa = (dQb if Qa.ctypes.data == Qb.ctypes.data and Qa.strides == Qb.strides
           else _ds_derivative(Qa))
    integrand = np.empty(len(r))
    for i in range(len(r)):
        inv_r = 1./r[i] if r[i] > 0 else 0.
        kin = .5*SL*((Qa[i]*dPb[i]+Qb[i]*dPa[i])
                     -(Pa[i]*dQb[i]+Pb[i]*dQa[i]))
        local = ((SL*kappa*(Pa[i]*Qb[i]+Qa[i]*Pb[i])
                  + rv[i]*(Pa[i]*Pb[i]+Qa[i]*Qb[i]))*inv_r
                 - 2*SL*SL*Qa[i]*Qb[i])
        integrand[i] = kin+local*difr[i]
    return float(slag6(1., integrand, len(r))[-1])
