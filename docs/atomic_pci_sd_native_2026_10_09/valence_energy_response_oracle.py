"""Independent complete numerator derivative, including implicit response."""

import numpy as np
from valence_allorder_oracle import action


def energy_derivative(direct, eps, core_s, core_d, column, energy, sigma, prime):
    nc = core_s.shape[1]
    n = len(eps)
    o, v = slice(0, nc), slice(nc, n)
    shift = energy - eps[column]
    ds = eps[o][None, :] - eps[v][:, None] + shift
    dc = (
        eps[o][None, None, :, None]
        + eps[o][None, None, None, :]
        - eps[v][:, None, None, None]
        - eps[v][None, :, None, None]
        + shift
    )
    dv = energy + eps[o][None, None, :] - eps[v][:, None, None] - eps[v][None, :, None]
    if any(np.any(a == 0) or not np.all(np.isfinite(a)) for a in (ds, dc, dv)):
        raise ValueError("singular/nonfinite independent response denominator")
    cs, cd, vd = np.zeros_like(core_s), np.zeros_like(core_d), np.zeros_like(sigma)
    cs[v] = -core_s[v] / ds
    cd[v, v] = -core_d[v, v] / dc
    vd[v, v] = prime[v, v] - sigma[v, v] / dv
    # The action is affine in the numerators. These effective numerator
    # directions encode each denominator derivative at the same energy.
    rs, rd = action(direct, eps, cs, cd, column, energy, vd, include_source=False)
    return rs, rd
