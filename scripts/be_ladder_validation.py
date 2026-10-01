"""Ladder Sigma(2) validation on Be: CI(P) + Eq.(11) vs CI(P+Q_D).

Target from the direction test (2026-10-03): Q_D gap = -1078 cm^-1
(E_CI(P) = -14.6102 Ha, E_CI(P+Q_D) = -14.6151 Ha).  A correct ladder
construction applied to the P-space CI must recover a substantial
fraction of that gap with the right sign — no 5x overshoot, no wrong
sign.  GATES: (a) diagonal element negative (attractive), (b) recovery
fraction in [0.3, 1.5].
"""
import sys

import numpy as np

from atomx import Atom, Correlation
from atomx.methods.mcdhf.excitations import generate_sd_excitations

sys.path.insert(0, "/tmp/atomx-cimbpt")

CORE = ["1s"]
REF = "1s2 2s2"
VIRTUALS_P = ["2s", "2p", "3s", "3p", "3d"]
VIRTUALS_Q = ["4s", "4p", "4d", "4f", "5s", "5p", "5d", "5f"]

REF_EP = -14.610192426
REF_EF = -14.615103704
GAP_CM = (REF_EF - REF_EP) * 219474.63


def solve_E(cfgs):
    atom = Atom(Z=4, config=cfgs,
               correlation=Correlation(j_resolved_csfs=True,
                                      radial_optimization="fac_frozen"),
               corrections="none", engine="bspline")
    atom.solve()
    return atom, atom.result.total_energy


if __name__ == "__main__":
    cfgP = generate_sd_excitations(REF, VIRTUALS_P, core_orbitals=CORE,
                                    max_excitations=2)
    atomP, EP = solve_E(cfgP)
    print(f"E_CI(P) = {EP:.9f} Ha (reference {REF_EP:.9f})", flush=True)

    # build the ladder on the P-space orbitals with Q virtuals
    # Rebuild a DF workspace with all P+Q subshells (tiny for Be)
    from atomx.methods.dirac_fock import DFSolver, Subshell
    subs = [Subshell(n=1, kappa=-1, occupation=2.0),
            Subshell(n=2, kappa=-1, occupation=2.0)]
    for nk in [(2, 1), (2, -2), (3, -1), (3, 1), (3, -2), (3, 2), (3, -3),
                (4, -1), (4, 1), (4, -2), (4, 2), (4, -3), (4, -4),
                (5, -1), (5, 1), (5, -2), (5, 2), (5, -3), (5, -4)]:
        subs.append(Subshell(n=nk[0], kappa=nk[1], occupation=1e-6))
    dhf = DFSolver(Z=4, subshells=subs, ns=50, ks=9,
                    starting_potential="tfda")
    r = dhf.solve()
    bsp = dhf.bsp_data
    orbitals = {(n, k): (P, Q) for (n, k), (P, Q, _o) in
                 r.orbitals_bybitshell().items() if False}         if False else {(n, k): (P, Q) for (n, k), (P, Q, _o) in
                       r.orbitals_by_subshell().items()}

    # orbital energies
    import re as _re

    def _kappa_of(label):
        m = _re.fullmatch(r"(\d+)([spdfgh])(\d+)/(\d+)", label)
        n = int(m.group(1))
        l = "spdfgh".index(m.group(2))
        j2 = int(round(2 * int(m.group(3)) / int(m.group(4))))
        return n, (-(l + 1) if j2 == 2 * l + 1 else l)

    eps = {_kappa_of(lab): e for lab, e in r.orbital_energies.items()}

    from atomx.methods.mbpt_sigma2 import build_sigma2_ladder
    VIRT_Q_KEYS = sorted({nk for nk in
                          [_kappa_of(v + "1/2") for v in VIRTUALS_Q]}
                         | set())
    # build full kappa list for Q virtuals
    Q_KEYS = []
    for v in VIRTUALS_Q:
        n = int(v[0]); lch = v[1]; l = "spdfgh".index(lch)
        for j2 in (2 * l - 1, 2 * l + 1):
            if j2 <= 0:
                continue
            kap = (-(l + 1) if j2 == 2 * l + 1 else l)
            Q_KEYS.append((n, kap))
    Q_KEYS = [k for k in Q_KEYS if k in orbitals and k in eps]

    VAL_KEYS = []
    for v in VIRTUALS_P:
        n = int(v[0]); l = "spdfgh".index(v[1])
        for j2 in (2 * l - 1, 2 * l + 1):
            if j2 <= 0:
                continue
            kap = (-(l + 1) if j2 == 2 * l + 1 else l)
            VAL_KEYS.append((n, kap))
    VAL_KEYS = [k for k in VAL_KEYS if k in orbitals]

    ladder = build_sigma2_ladder(orbitals, bsp, 4, 50,
                                 virt_keys=Q_KEYS,
                                 valence_keys=VAL_KEYS, eps=eps)
    print(f"ladder intermediates = {ladder.n_intermediate}", flush=True)

    # GATE (a): diagonal attractive
    from atomx.angular.csf_hamiltonian import _m2_values, twoj
    idx = ladder.index
    up2s = idx[(2, -1, 1)]
    dn2s = idx[(2, -1, -1)]
    d_diag = ladder(up2s, dn2s, up2s, dn2s)
    print(f"ladder diagonal (2s2s) = {d_diag:.6e} Ha", flush=True)
    assert d_diag < 0.0, "diagonal must be attractive"

    # GATE (b): apply to the P-space CI and measure recovery
    import atomx.physics.csf.sparse_ci as SC
    from atomx.config.csf import enumerate_jj_csfs_many, patterns_from_csfs
    from atomx.physics.csf.sparse_ci import (
        enumerate_determinants as enum_dets)

    pats = patterns_from_csfs(enumerate_jj_csfs_many(cfgP))
    # restrict to the rebuilt orbitals' pattern shells
    masks, spinors_probe = enum_dets(
        [p for p in pats if all((n, k) in orbitals for (n, k, _q) in p)],
        orbitals, 0)

    def delta(ia, ib, ic, id_):
        la, lb = spinors_probe[ia], spinors_probe[ib]
        lc, ld = spinors_probe[ic], spinors_probe[id_]
        try:
            ea = ladder.index[la]; eb = ladder.index[lb]
            ec = ladder.index[lc]; ed = ladder.index[ld]
        except KeyError:
            return 0.0
        return ladder(ea, eb, ec, ed)

    orig_assemble = SC.assemble_sparse_hamiltonian

    def _with_ladder(masks, spinors, od, bsp_data, Z, ns, **kw):
        return orig_assemble(masks, spinors, od, bsp_data, Z, ns,
                            two_body_delta=delta, **kw)

    SC.assemble_sparse_hamiltonian = _with_ladder
    SC._SECTOR_CACHE.clear()
    try:
        atomL, EL = solve_E(cfgP)
    finally:
        SC.assemble_sparse_hamiltonian = orig_assemble
        SC._SECTOR_CACHE.clear()
    print(f"E_CI(P)+ladder = {EL:.9f} Ha", flush=True)

    recovery = (EP - EL) / (REF_EP - REF_EF)
    print(f"\nrecovery = ({EP:.6f} - {EL:.6f}) / {REF_EP - REF_EF:.6f} "
          f"= {recovery * 100:.1f}%", flush=True)
    print(f"PASS: 0.3 <= recovery <= 1.5"
          if 0.3 <= recovery <= 1.5 else "FAIL", flush=True)
