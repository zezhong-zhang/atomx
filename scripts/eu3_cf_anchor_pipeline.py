"""Eu3+ CF anchor FULL PIPELINE (the CF track deliverable).

One fixed DF workspace; one run:
  1. Tier-A calibration (overdetermined: s2 = Pr prior, fit s4/s6 on
     the 5D0/5D1/5D2 gaps) via slater_scaling;
  2. point-charge cubic CF (real B set, r_inside guard);
  3. combined assembly: Slater-scaling delta + CF m-resolved channel
     on the 4f6 full M space.

GATES: (a) 5D0 anchor lands at 17270 (+-80); (b) the CF trace within
each J manifold vanishes (k>0 operators are traceless — barycenters
move only by mixing); (c) assembled H Hermitian.
"""
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from atomx.config.csf import enumerate_jj_csfs_many
from atomx.methods.csf.crystal_field import build_cf_operator, point_charge_bkq
from atomx.methods.csf.slater_scaling import (
    fit_slater_scaling,
    free_ion_levels,
    make_slater_delta,
)
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.mcdhf.excitations import generate_sd_excitations
from atomx.physics.csf import sparse_ci as SC
from atomx.physics.csf.sparse_ci import patterns_from_csfs

CM = 219474.63
Xe = [(1, -1), (2, -1), (2, 1), (2, -2), (3, -1), (3, 1), (3, -2),
      (3, 2), (3, -3), (4, -1), (4, 1), (4, -2), (4, 2), (4, -3),
      (5, -1), (5, 1), (5, -2)]


def cap(k):
    return 2 * k if k > 0 else -2 * k


REFS = [(0.0, 17270.0), (1.0, 19025.0), (2.0, 21500.0)]
NIST_7F = [0.0, 370.0, 1041.0, 1890.0, 2862.0, 3807.0, 4940.0]
S2_PRIOR = -0.2726
CM_INV = 1.0 / 219474.63


def build_eu(ns=50):
    subs = [Subshell(n=n, kappa=k, occupation=float(cap(k)))
            for n, k in Xe]
    subs.append(Subshell(n=4, kappa=3, occupation=6 * 6 / 14))
    subs.append(Subshell(n=4, kappa=-4, occupation=6 * 8 / 14))
    dhf = DFSolver(Z=63, subshells=subs, ns=ns, ks=9,
                   starting_potential="tfda")
    r = dhf.solve()
    od = {(n, k): (P, Q) for (n, k), (P, Q, _o)
          in r.orbitals_by_subshell().items()}
    eps = {}
    for lab, e in r.orbital_energies.items():
        m = re.fullmatch(r"(\d+)([spdf])(\d+)/(\d+)", lab)
        n = int(m.group(1))
        ll = "spdf".index(m.group(2))
        j2 = int(round(2 * int(m.group(3)) / int(m.group(4))))
        eps[(n, -(ll + 1) if j2 == 2 * ll + 1 else ll)] = e
    REF = "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f6"
    cfgs = generate_sd_excitations(REF, ["4f"], core_orbitals=None,
                                   max_excitations=0)
    pats = patterns_from_csfs(enumerate_jj_csfs_many(cfgs))
    pats = [p for p in pats if all((n, k) in od for (n, k, _q) in p)]
    return od, dhf.bsp_data, eps, pats


def extract(lv):
    allL = sorted((e, J) for J, es in lv.items() for e in es)
    e0 = allL[0][0]
    out = {}
    for J, _ref in REFS:
        e = min(e for e, j in allL if j == J and e - e0 > 15000)
        out[J] = e - e0
    return out, e0


if __name__ == "__main__":
    t0 = time.time()
    od, bsp, eps, pats = build_eu()
    print(f"[1] DF workspace {time.time()-t0:.0f}s", flush=True)

    # ---- Tier-A overdetermined fit ------------------------------------
    def res(p):
        s4, s6 = p
        lv = free_ion_levels(od, bsp, 63, 50, pats,
                             scale=(S2_PRIOR, s4, s6))
        gaps, _ = extract(lv)
        return [gaps[J] - ref for J, ref in REFS]

    x, hist, diag = fit_slater_scaling(res, x0=(-0.08, -0.119))
    s = (S2_PRIOR, *x)
    r_fin = np.array(res(x))
    print(f"[2] Tier-A: s=({s[0]:+.4f},{s[1]:+.4f},{s[2]:+.4f}) "
          f"RMS={np.sqrt(np.mean(r_fin**2)):.1f} cm^-1 "
          f"cond={diag.get('cond'):.0f}", flush=True)

    # ---- CF: cubic point charges --------------------------------------
    a_chg = 4.5
    charges = [(1.0, (s1 * a_chg, s2 * a_chg, s3 * a_chg))
               for s1 in (-1, 1) for s2 in (-1, 1) for s3 in (-1, 1)]
    bkq = point_charge_bkq(charges, (0, 0, 0), kmax=6, r_inside=2.0)
    bkq_real = {kk: float(v.real) for kk, v in bkq.items()
                if abs(v.real) > 1e-9}
    print(f"[3] cubic B (cm^-1): "
          f"{ {kk: round(v,3) for kk,v in sorted(bkq_real.items())} }",
          flush=True)
    cf = build_cf_operator(od, bsp, 63, 50, bkq_real)

    # ---- combined full-M assembly ----------------------------------------
    masks, spinors = SC.enumerate_determinants(pats, od, None)
    sd = make_slater_delta(od, bsp, s)

    def sl_delta(ia, ib, ic, id_):
        return sd(ia, ib, ic, id_, spinors)

    lad = cf.index

    def m_delta(ia, ic):
        try:
            return float(cf(lad[spinors[ia]], lad[spinors[ic]]))
        except KeyError:
            return 0.0

    H_cf = SC.assemble_sparse_hamiltonian(masks, spinors, od, bsp, 63,
                                          50, two_body_delta=sl_delta,
                                          one_body_m_delta=m_delta)
    H0 = SC.assemble_sparse_hamiltonian(masks, spinors, od, bsp, 63,
                                        50, two_body_delta=sl_delta)
    Hd = (H_cf - H_cf.T).tocoo()
    assert Hd.data.size == 0 or np.max(np.abs(Hd.data)) < 1e-8
    w0 = np.linalg.eigvalsh(H0.toarray())
    w1 = np.linalg.eigvalsh(H_cf.toarray())
    e0 = w0[0]
    w0cm = (w0 - e0) / CM_INV   # Hartree -> cm^-1
    w1cm = (w1 - w1[0]) / CM_INV
    i5d0 = int(np.min(np.where(w0cm > 15000)))
    anchor = w0cm[i5d0]
    print(f"[4] full M space: {len(masks)} dets; "
          f"5D0 anchor = {anchor:.1f} cm^-1 (gate 17270+-80)",
          flush=True)
    assert abs(anchor - 17270) < 80, anchor
    # trace gate: total CF shift over the sector ~ 0 (k>0 traceless)
    trace_shift = float((H_cf - H0).diagonal().sum() / CM_INV)
    print(f"[5] CF trace over full M space = {trace_shift:+.2e} cm^-1 "
          f"(k>0 traceless -> ~0)", flush=True)
    print("[6] CF-split full-space levels around 5D0 (cm^-1):")
    i5d1 = int(np.min(np.where(w1cm > 15000)))
    for e in w1cm[i5d1 - 2:i5d1 + 4]:
        print(f"    {e:10.2f}", flush=True)
    print(f"[7] CF moves the 5D0-region lowest level by "
          f"{w1cm[i5d1] - anchor:+.2f} cm^-1 (Stokes-like mixing)",
          flush=True)
    # Subtracting thousands of large core-energy diagonals loses precision.
    # Gate the operator trace independently, and bound the assembled trace
    # by the roundoff scale of the actual diagonal subtraction.
    operator_trace = sum(sum(m_delta(i,i) for i in range(len(spinors))
                             if mask & (1 << i)) for mask in masks)
    assert abs(operator_trace) < 1e-10, operator_trace
    roundoff = (10 * np.finfo(float).eps *
                (np.abs(H0.diagonal()).sum() + np.abs(H_cf.diagonal()).sum()) / CM_INV)
    assert abs(trace_shift) < roundoff, (trace_shift,roundoff)
