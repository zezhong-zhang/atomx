"""Fixed-workspace ladder validation on Be (rewritten 2026-10-01).

The first direction test compared two ``Atom.solve()`` runs whose config
sets differ — and the DF workspace differs with the config set (known
trap: a wider window changes the frozen-core solution itself), so its
-1078 cm^-1 gap was NOT a clean Q_D coupling measurement.  This rewrite
keeps ONE DF workspace and does three sparse diagonalisations at 2M=0:

    E_P  : CI over P determinants (SD excitations into 2s..3d)
    E_PQ : CI over P+Q determinants (adds 4s..5f) — the TRUE Q_D gap
    E_L  : the P space with the ladder Sigma(2) added (two_body_delta)

Calibration: eigen-PT2  dE = sum_{I in Q} |<Psi_P|H|I>|^2/(E_P - H_II)
computed in the same workspace — the leading-order quantity the ladder
must reproduce.  GATES: (a) ladder diagonal negative; (b) recovery
(E_P - E_L)/(E_P - E_PQ) in [0.3, 1.5].
"""
import re
import sys

import numpy as np
from scipy.sparse.linalg import eigsh

sys.path.insert(0, "/tmp/atomx-cimbpt")

from atomx import Correlation  # noqa: F401  (documented engine contrast)
from atomx.config.csf import enumerate_jj_csfs_many
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.mcdhf.excitations import generate_sd_excitations
from atomx.methods.mbpt_sigma2 import build_sigma2_ladder
from atomx.physics.csf import sparse_ci as SC
from atomx.physics.csf.sparse_ci import patterns_from_csfs

CORE = ["1s"]
REF = "1s2 2s2"
VIRTUALS_P = ["2s", "2p", "3s", "3p", "3d"]
VIRTUALS_Q = ["4s", "4p", "4d", "4f", "5s", "5p", "5d", "5f"]
CM = 219474.63


def keys_of(shells):
    out = []
    for v in shells:
        n = int(v[0])
        ll = "spdfgh".index(v[1])
        for j2 in (2 * ll - 1, 2 * ll + 1):
            if j2 <= 0:
                continue
            out.append((n, -(ll + 1) if j2 == 2 * ll + 1 else ll))
    return out


def kap(label):
    m = re.fullmatch(r"(\d+)([spdfgh])(\d+)/(\d+)", label)
    n = int(m.group(1))
    ll = "spdfgh".index(m.group(2))
    j2 = int(round(2 * int(m.group(3)) / int(m.group(4))))
    return n, (-(ll + 1) if j2 == 2 * ll + 1 else ll)


def lowest(H, k=2):
    n = H.shape[0]
    if n <= 8:
        w, v = np.linalg.eigh(H.toarray())
        return w[0], v[:, 0]
    ncv = min(n - 1, max(4 * k + 1, 60))
    w, v = eigsh(H, k=k, which="SA", ncv=ncv)
    i = int(np.argmin(w))
    return w[i], v[:, i]


def phys_occ(mask, spinors):
    return frozenset(spinors[i] for i in range(len(spinors)) if mask >> i & 1)


if __name__ == "__main__":
    # ---- 1. ONE DF workspace ------------------------------------------------
    extra = [(2, 1), (2, -2), (3, -1), (3, 1), (3, -2), (3, 2), (3, -3),
             (4, -1), (4, 1), (4, -2), (4, 2), (4, -3), (4, -4),
             (5, -1), (5, 1), (5, -2), (5, 2), (5, -3), (5, -4)]
    dhf = DFSolver(Z=4,
                   subshells=[Subshell(n=1, kappa=-1, occupation=2.0),
                              Subshell(n=2, kappa=-1, occupation=2.0)] +
                   [Subshell(n=nk[0], kappa=nk[1], occupation=1e-6)
                    for nk in extra],
                   ns=50, ks=9, starting_potential="tfda")
    r = dhf.solve()
    orbitals = {(n, k): (P, Q) for (n, k), (P, Q, _o)
                in r.orbitals_by_subshell().items()}
    eps = {kap(l): e for l, e in r.orbital_energies.items()}
    bsp = dhf.bsp_data

    VAL = [k for k in keys_of(VIRTUALS_P) if k in orbitals]
    QK = [k for k in keys_of(VIRTUALS_Q) if k in orbitals and k in eps]
    print(f"workspace: {len(orbitals)} subshells, VAL={len(VAL)}, Q={len(QK)}",
          flush=True)

    # ---- 2. ladder + GATE (a) ----------------------------------------------
    ladder = build_sigma2_ladder(orbitals, bsp, 4, 50, virt_keys=QK,
                                 valence_keys=VAL, eps=eps)
    up, dn = ladder.index[(2, -1, 1)], ladder.index[(2, -1, -1)]
    d_diag = ladder(up, dn, up, dn)
    print(f"ladder intermediates = {ladder.n_intermediate}", flush=True)
    print(f"ladder diagonal (2s2s) = {d_diag:.6e} Ha", flush=True)
    assert d_diag < 0.0, "diagonal must be attractive"

    # ---- 3. patterns / determinants ----------------------------------------
    def pats_of(virts):
        cfgs = generate_sd_excitations(REF, virts, core_orbitals=CORE,
                                       max_excitations=2)
        pats = patterns_from_csfs(enumerate_jj_csfs_many(cfgs))
        return [p for p in pats
                if all((n, k) in orbitals for (n, k, _q) in p)]

    patsP, patsPQ = pats_of(VIRTUALS_P), pats_of(VIRTUALS_P + VIRTUALS_Q)
    masksP, spP = SC.enumerate_determinants(patsP, orbitals, 0)
    masksPQ, spPQ = SC.enumerate_determinants(patsPQ, orbitals, 0)
    print(f"dets: P={len(masksP)}  P+Q={len(masksPQ)}", flush=True)

    HP = SC.assemble_sparse_hamiltonian(masksP, spP, orbitals, bsp, 4, 50)
    EP, vP = lowest(HP)
    HPQ = SC.assemble_sparse_hamiltonian(masksPQ, spPQ, orbitals, bsp, 4, 50)
    EPQ, _ = lowest(HPQ)
    gap = (EPQ - EP) * CM
    print(f"E_P  = {EP:.9f} Ha", flush=True)
    print(f"E_PQ = {EPQ:.9f} Ha   fixed-workspace Q_D gap = {gap:+.2f} cm^-1",
          flush=True)

    # ---- 4. eigen-PT2 calibration in the same workspace --------------------
    qkeys = set(QK)
    q_rows = [i for i, m in enumerate(masksPQ)
              if any(spPQ[b][:2] in qkeys
                     for b in range(len(spPQ)) if m >> b & 1)]
    # map the P eigenvector into the P+Q enumeration by physical occupation
    row_of_occ = {phys_occ(m, spPQ): i for i, m in enumerate(masksPQ)}
    v_ext = np.zeros(len(masksPQ))
    for j, m in enumerate(masksP):
        v_ext[row_of_occ[phys_occ(m, spP)]] = vP[j]
    w = HPQ @ v_ext
    diag = HPQ.diagonal()
    dE_pt2 = float(np.sum(w[q_rows] ** 2 / (EP - diag[q_rows])))
    print(f"eigen-PT2 into Q = {dE_pt2 * CM:+.2f} cm^-1 "
          f"({dE_pt2 / (EPQ - EP) * 100:.1f}% of the CI gap)", flush=True)
    # decompose by excitation class: n_q = 1 (singles / mixed doubles),
    # n_q = 2 (pure-Q doubles — the Eq.(11) ladder sector)
    contrib = {}
    for i in q_rows:
        occ = [spPQ[b] for b in range(len(spPQ)) if masksPQ[i] >> b & 1]
        nq = sum(1 for s in occ if s[:2] in qkeys)
        has2s = any(s[:2] == (2, -1) for s in occ)
        tag = {1: "single(q+2s)" if has2s else "mixed-double(q+P)",
               2: "pure-Q-double"}[nq]
        contrib[tag] = contrib.get(tag, 0.0) + w[i] ** 2 / (EP - diag[i])
    for tag in sorted(contrib):
        print(f"  {tag}: {contrib[tag] * CM:+.3f} cm^-1", flush=True)

    # ---- 5. GATE (b): ladder-augmented P CI --------------------------------
    lad_idx = ladder.index

    def delta(ia, ib, ic, id_):
        try:
            ea = lad_idx[spP[ia]]
            eb = lad_idx[spP[ib]]
            ec = lad_idx[spP[ic]]
            ed = lad_idx[spP[id_]]
        except KeyError:
            return 0.0
        return ladder(ea, eb, ec, ed)

    HL = SC.assemble_sparse_hamiltonian(masksP, spP, orbitals, bsp, 4, 50,
                                        two_body_delta=delta)
    EL, _ = lowest(HL)
    lad_shift = (EP - EL) * CM
    print(f"E_L  = {EL:.9f} Ha   ladder shift = {lad_shift:+.2f} cm^-1",
          flush=True)

    recovery = (EP - EL) / (EPQ - EP) if EPQ != EP else float("nan")
    print(f"\nrecovery = {recovery * 100:.1f}%", flush=True)
    ok = 0.3 <= recovery <= 1.5
    print("PASS: 0.3 <= recovery <= 1.5" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)
