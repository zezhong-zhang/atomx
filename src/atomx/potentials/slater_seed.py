"""Nuclear field and Thomas–Fermi–Molière seed on a supplied radial grid."""
import math

import numba as nb
import numpy as np

from atomx.numerics.shooting.constants import A0B, FR1O3
from atomx.numerics.shooting.quadrature import slag6


@nb.njit(cache=True)
def slater_initial_field(z, shell_occ, r, difr, aw, vscr):
    """Prepare the nuclear field and Thomas–Fermi–Molière seed density."""
    np_ = len(r)
    nshell = len(shell_occ)
    qelec = 0.0
    for s in range(nshell):
        qelec += shell_occ[s]

    # Nuclear potential: r*V_nuc = -Z (point nucleus)
    vnuc = np.zeros(np_)
    for i in range(np_):
        vnuc[i] = -z

    # Finite nuclear size (Fermi distribution) if aw > 0.5
    if aw > 0.5:
        f2bohr = 1.0e-13 / A0B
        r1_nuc = 1.07 * aw ** FR1O3 * f2bohr
        r2_nuc = 0.546 * f2bohr
        denn = np.zeros(np_)
        vnuc_work = np.zeros(np_)
        for i in range(np_):
            xx = math.exp((r1_nuc - r[i]) / r2_nuc) if (r1_nuc - r[i]) / r2_nuc < 500 else 1.0e200
            denn[i] = xx / (1.0 + xx)
            vnuc_work[i] = denn[i] * r[i] ** 2 * difr[i]
        fwd_nuc = slag6(1.0, vnuc_work, np_)
        aux_nuc = np.zeros(np_)
        for i in range(np_ - 1):
            idx = np_ - 1 - i
            aux_nuc[i] = denn[idx] * r[idx] * difr[idx]
        aux_nuc[np_ - 1] = 0.0
        bwd_nuc = slag6(1.0, aux_nuc, np_)
        fnorm = z / fwd_nuc[np_ - 1] if abs(fwd_nuc[np_ - 1]) > 1.0e-30 else z
        for i in range(np_):
            vnuc[i] = -fnorm * (fwd_nuc[i] + bwd_nuc[np_ - 1 - i] * r[i])

    # Plasma screening: the caller sampled r*delta_V(r) on this same SGRID
    # (sgrid(rn, r2, drn, np_) is deterministic), so a length mismatch is a
    # programming error and must not degrade into "no screening".
    if vscr.shape[0] != 0:
        if vscr.shape[0] != np_:
            raise ValueError(
                "plasma screening grid does not match the SCF grid")
        for i in range(np_):
            vnuc[i] += vscr[i]

    # Thomas-Fermi-Moliere starting density (DHFS.f:377-393)
    rtf = 0.88534 / z ** 0.33333333333
    al1 = -6.0 / rtf
    al2 = -1.2 / rtf
    al3 = -0.3 / rtf
    rden = np.zeros(np_)
    aux1 = np.zeros(np_)
    for i in range(np_):
        rr = r[i]
        rden[i] = rr * (0.10 * al1 * al1 * math.exp(al1 * rr)
                        + 0.55 * al2 * al2 * math.exp(al2 * rr)
                        + 0.35 * al3 * al3 * math.exp(al3 * rr))
        aux1[i] = rden[i] * difr[i]
    den_int = slag6(1.0, aux1, np_)
    rnorm = qelec / den_int[np_ - 1] if abs(den_int[np_ - 1]) > 1.0e-30 else 1.0
    for i in range(np_):
        rden[i] *= rnorm

    return r, difr, np_, vnuc, rden, qelec
