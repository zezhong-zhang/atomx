"""Independent magnetic contractions and signed fermion Fock commutator."""

import itertools

import numpy as np

from atomx.angular.csf_hamiltonian import _ops_sign


def magnetic_fock(fock, eps, core_s, core_d, column, energy, sigma, prime):
    nc = core_s.shape[1]
    n = len(eps)
    o, v = slice(0, nc), slice(nc, n)
    off = fock - np.diag(np.diag(fock))
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
        raise ValueError("singular/nonfinite declared Fock denominator")
    cs = core_s[v] / ds
    cd = core_d[v, v] / dc
    vd = sigma[v, v] / dv
    vp = prime[v, v] / dv - vd / dv

    def contraction(cs, cd, vd):
        rs = np.zeros(n, dtype=np.result_type(fock, eps, cs, cd, vd))
        rd = np.zeros_like(sigma)
        rs[v] = -np.einsum("c,mc->m", off[o, column], cs) + np.einsum(
            "br,mrb->m", off[o, v], vd - vd.swapaxes(0, 1)
        )
        rd[v, v] = (
            np.einsum("mr,rnb->mnb", off[v, v], vd)
            + np.einsum("nr,mrb->mnb", off[v, v], vd)
            - np.einsum("cb,mnc->mnb", off[o, o], vd)
            - np.einsum("c,mncb->mnb", off[o, column], cd)
        )
        return rs, rd

    rs, rd = contraction(cs, cd, vd)
    ps, pd = contraction(-cs / ds, -cd / dc, vp)
    if not all(np.all(np.isfinite(a)) for a in (rs, rd, ps, pd)):
        raise ValueError("nonfinite complete Fock contraction")
    cs_full, cd_full, vd_full = (
        np.zeros_like(core_s),
        np.zeros_like(core_d),
        np.zeros_like(sigma),
    )
    cs_full[v], cd_full[v, v], vd_full[v, v] = cs, cd, vd
    return (rs, rd, ps, pd), (cs_full, cd_full, vd_full)


def fermion_fock(fock, core_s, core_d, valence_d, column):
    """Evaluate actual operators, without using the magnetic contraction formula."""
    nc = core_s.shape[1]
    n = len(fock)
    reference = (1 << nc) - 1
    attached = reference | (1 << column)
    off = fock - np.diag(np.diag(fock))
    fterms = [
        (float(off[p, q]), ((p, True), (q, False)))
        for p, q in itertools.product(range(n), repeat=2)
        if off[p, q] != 0
    ]
    cterms = []
    vterms = []
    for m, a in itertools.product(range(nc, n), range(nc)):
        if core_s[m, a] != 0:
            cterms.append((core_s[m, a], ((m, True), (a, False))))
    for m, nn, a, b in itertools.product(
        range(nc, n), range(nc, n), range(nc), range(nc)
    ):
        if core_d[m, nn, a, b] != 0:
            cterms.append(
                (
                    0.5 * core_d[m, nn, a, b],
                    ((m, True), (nn, True), (b, False), (a, False)),
                )
            )
    for m, nn, b in itertools.product(range(nc, n), range(nc, n), range(nc)):
        if valence_d[m, nn, b] != 0:
            vterms.append(
                (
                    valence_d[m, nn, b],
                    ((m, True), (nn, True), (b, False), (column, False)),
                )
            )

    def apply(terms, vector):
        result = {}
        for mask, coefficient in vector.items():
            for value, ops in terms:
                phase = _ops_sign(mask, ops)
                if phase:
                    target = mask
                    for bit, create in reversed(ops):
                        target = target | (1 << bit) if create else target & ~(1 << bit)
                    result[target] = result.get(target, 0) + coefficient * value * phase
        return result

    def commutator(terms, mask):
        vector = {mask: 1.0}
        ft = apply(fterms, apply(terms, vector))
        tf = apply(terms, apply(fterms, vector))
        return {key: ft.get(key, 0) - tf.get(key, 0) for key in ft.keys() | tf.keys()}

    result = commutator(cterms + vterms, attached)
    spectator = apply([(1.0, ((column, True),))], commutator(cterms, reference))
    for key, value in spectator.items():
        result[key] = result.get(key, 0) - value
    single = np.zeros(n)
    double = np.zeros_like(valence_d)
    for m in range(nc, n):
        ops = ((m, True), (column, False))
        target = (attached & ~(1 << column)) | (1 << m)
        single[m] = result.get(target, 0) * _ops_sign(attached, ops)
    for m, nn, b in itertools.product(range(nc, n), range(nc, n), range(nc)):
        ops = ((m, True), (nn, True), (b, False), (column, False))
        phase = _ops_sign(attached, ops)
        if phase:
            target = (attached & ~(1 << column) & ~(1 << b)) | (1 << m) | (1 << nn)
            double[m, nn, b] = phase * result.get(target, 0)
    return single, double
