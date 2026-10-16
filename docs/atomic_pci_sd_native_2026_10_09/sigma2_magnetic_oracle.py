"""Complete finite two-valence magnetic screening and radial normalization."""

import itertools

import numpy as np

from atomx.angular.sd_rho import rho_coefficient
from atomx.angular.wigner import wigner_3j


def s1(rank, left, right):
    """Original pCI parity-free S1, used solely to undo radial normalization."""
    a, b = abs(left), abs(right)
    return (
        2
        * (-1) ** a
        * np.sqrt(a * b)
        * wigner_3j(2 * a - 1, 2 * b - 1, 2 * rank, -1, 1, 0)
    )


def expand_sigma2(labels, plan, valence, rows, radial):
    nc = sum(2 * abs(k) for _, k in plan["occupied_keys"])
    positions = {
        key: (ip, group.index(key))
        for ip, group in enumerate(plan["groups"])
        for key in group
    }
    vpositions = {key: i for i, key in enumerate(valence)}
    channels = {tuple(row): i + 1 for i, row in enumerate(rows)}
    output = np.zeros((len(labels) - nc,) * 4)
    for p, q, r, s in itertools.product(range(nc, len(labels)), repeat=4):
        a, b, v, w = (labels[i] for i in (p, q, r, s))
        ip, m = positions[a[:2]]
        jn, n = positions[b[:2]]
        iv, iw = vpositions[v[:2]], vpositions[w[:2]]
        coefficient = 0.0
        for rank in range(plan["dimensions"]["KK"] + 1):
            channel = channels.get((iv, iw, ip, jn, rank))
            if channel is not None:
                c0 = (-1) ** rank * s1(rank, a[1], v[1]) * s1(rank, b[1], w[1])
                coefficient += (
                    rho_coefficient(rank, a, b, v, w) * c0 * radial[m, n, channel]
                )
        output[p - nc, q - nc, r - nc, s - nc] = coefficient
    return output


def action(
    direct, eps, core_s, core_d, magnetic_sigmas, energies, *, magnetic_primes=None
):
    """Eq.38 complete signed magnetic sums, fixed solved core/valence input."""
    nc = core_s.shape[1]
    n = len(eps)
    occupied, virtual = slice(0, nc), slice(nc, n)
    es, ev = eps[occupied], eps[virtual]
    g = direct
    gt = g - g.swapaxes(2, 3)
    result = np.zeros(
        (n - nc,) * 4,
        dtype=np.result_type(direct, eps, core_s, core_d, magnetic_sigmas, energies),
    )
    for v, w in itertools.product(range(nc, n), repeat=2):
        E_v, E_w = energies[v], energies[w]
        shift_v, shift_w = E_v - eps[v], E_w - eps[w]
        ds = es[None, :] - ev[:, None] + shift_v + shift_w
        dc = (
            es[None, None, :, None]
            + es[None, None, None, :]
            - ev[:, None, None, None]
            - ev[None, :, None, None]
            + shift_v
            + shift_w
        )
        dv = E_v + es[None, None, :] - ev[:, None, None] - ev[None, :, None] + shift_w
        dw = E_w + es[None, None, :] - ev[:, None, None] - ev[None, :, None] + shift_v
        if any(np.any(d == 0) or not np.all(np.isfinite(d)) for d in (ds, dc, dv, dw)):
            raise ValueError("singular/nonfinite complete screening denominator")
        cs = core_s[virtual] / ds
        cd = core_d[virtual, virtual] / dc
        vd = magnetic_sigmas[virtual, virtual, :, v] / dv
        wd = magnetic_sigmas[virtual, virtual, :, w] / dw
        if magnetic_primes is not None:
            if magnetic_primes.shape != magnetic_sigmas.shape or not np.all(
                np.isfinite(magnetic_primes)
            ):
                raise ValueError(
                    "complete finite magnetic numerator derivatives required"
                )
            cs = -2 * cs / ds
            cd = -2 * cd / dc
            vd = magnetic_primes[virtual, virtual, :, v] / dv - 2 * vd / dv
            wd = magnetic_primes[virtual, virtual, :, w] / dw - 2 * wd / dw
        result[:, :, v - nc, w - nc] = (
            np.einsum("cd,mncd->mn", g[occupied, occupied, v, w], cd)
            - np.einsum("cn,mc->mn", g[occupied, virtual, v, w], cs)
            - np.einsum("mc,nc->mn", g[virtual, occupied, v, w], cs)
            + np.einsum("cnr,mrc->mn", gt[occupied, virtual, virtual, w], vd)
            - np.einsum("cnr,rmc->mn", g[occupied, virtual, virtual, w], vd)
            - np.einsum("mcr,rnc->mn", g[virtual, occupied, virtual, w], vd)
            + np.einsum("mcr,nrc->mn", gt[virtual, occupied, v, virtual], wd)
            - np.einsum("mcr,rnc->mn", g[virtual, occupied, v, virtual], wd)
            - np.einsum("cnr,rmc->mn", g[occupied, virtual, v, virtual], wd)
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("nonfinite independent complete screening")
    return result
