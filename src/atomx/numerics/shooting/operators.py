"""Pure radial Dirac operators in the native SLAG6 metric."""
import numpy as np
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


def _ds_derivative(y):
    """dy/ds on the uniform-s grid (unit index spacing), 8th order."""
    n = y.shape[0]
    if n < 9:
        raise ValueError(f"grid too small for 8th-order stencils (n={n})")
    c1, c2, c3, c4 = _FD_CENTRAL
    d = np.empty_like(y)
    d[4:n - 4] = (c1 * (y[5:n - 3] - y[3:n - 5])
                  + c2 * (y[6:n - 2] - y[2:n - 6])
                  + c3 * (y[7:n - 1] - y[1:n - 7])
                  + c4 * (y[8:n] - y[0:n - 8]))
    head = y[:9]
    tail = y[n - 9:][::-1]
    for i in range(4):
        d[i] = _FD_EDGE[i] @ head
        d[n - 1 - i] = -(_FD_EDGE[i] @ tail)
    return d


def one_body_element(Pa, Qa, Pb, Qb, kappa, r, difr, rv):
    """Symmetric rest-mass-subtracted Dirac matrix element."""
    inv_r = np.divide(1., r, out=np.zeros_like(r), where=r > 0)
    kin = .5*SL*((Qa*_ds_derivative(Pb)+Qb*_ds_derivative(Pa))
                  -(Pa*_ds_derivative(Qb)+Pb*_ds_derivative(Qa)))
    local = ((SL*kappa*(Pa*Qb+Qa*Pb)+rv*(Pa*Pb+Qa*Qb))*inv_r
             -2*SL*SL*Qa*Qb)
    return float(slag6(1., kin+local*difr, len(r))[-1])
