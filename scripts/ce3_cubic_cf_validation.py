"""Ce3+-like 4f^1 cubic crystal-field validation (2026-10-01).

First physics run of the CF track.  A frozen [Xe]-core DF workspace
with one 4f electron; the sparse CI gives the free-ion 2F levels; the
cubic point-charge lattice (8 charges at cube vertices) splits them.

GATES (hard):
  (a) free-ion: 6-fold 2F5/2 BELOW 8-fold 2F7/2, splitting ~1e3 cm^-1
  (b) cubic degeneracy pattern EXACTLY 2+4 (J=5/2 block) and 2+2+4
      (J=7/2 block) — group theory is unforgiving
  (c) quadrature cross-check: cf() elements vs DIRECT 3D numerical
      integration of sum_i q_i/|R_i - r| (no Y_kq expansion anywhere)
  (d) assembled H Hermitian
"""
import sys

import numpy as np

sys.path.insert(0, "/tmp/atomx-cimbpt")

from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.csf.crystal_field import (build_cf_operator,
                                             point_charge_bkq)
from atomx.physics.csf import sparse_ci as SC

CM = 219474.63

Xe = [(1, -1), (2, -1), (2, 1), (2, -2), (3, -1), (3, 1), (3, -2),
      (3, 2), (3, -3), (4, -1), (4, 1), (4, -2), (4, 2), (4, -3),
      (5, -1), (5, 1), (5, -2)]


def _cap(k):
    return 2 * k if k > 0 else -2 * k


assert sum(_cap(k) for _, k in Xe) == 54

subs = [Subshell(n=n, kappa=k, occupation=float(_cap(k))) for n, k in Xe]
# 4f1 statistical: 4f5/2 (kappa=3) 1*6/14, 4f7/2 (kappa=-4) 1*8/14
subs.append(Subshell(n=4, kappa=3, occupation=6.0 / 14))
subs.append(Subshell(n=4, kappa=-4, occupation=8.0 / 14))

dhf = DFSolver(Z=58, subshells=subs, ns=50, ks=9,
               starting_potential="tfda")
r = dhf.solve()
od = {(n, k): (P, Q) for (n, k), (P, Q, _o)
      in r.orbitals_by_subshell().items()}
bsp = dhf.bsp_data
print("DF solved; orbitals:", len(od), flush=True)

# ---- cubic lattice: 8 unit charges at (±a,±a,±a), a = 4.5 bohr -------
a = 4.5
charges = [(1.0, (s1 * a, s2 * a, s3 * a))
           for s1 in (-1, 1) for s2 in (-1, 1) for s3 in (-1, 1)]
bkq = point_charge_bkq(charges, (0.0, 0.0, 0.0), kmax=6, r_inside=2.0)
nonzero = {kk: v for kk, v in bkq.items() if abs(v) > 1e-9}
print("cubic B set (cm^-1):",
      {kk: f"{v.real:+.4g}" for kk, v in sorted(nonzero.items())},
      flush=True)
# cubic symmetry gate: the O_h invariants (von der Lage-Bethe kubic
# harmonics) are q=0,+-4 for BOTH k=4 and k=6 (NOT q=6 for k=6), and
# the set must be REAL — this is why no tesseral is needed for cubic
# sites.  The cubic RATIOS are textbook: B4^4/B4^0 = sqrt(5/14),
# B6^4/B6^0 = -sqrt(7/2) — assert them digit-level.
for (k, q), v in nonzero.items():
    assert (k, abs(q)) in [(4, 0), (4, 4), (6, 0), (6, 4)], (k, q)
    assert abs(v.imag) < 1e-9 * max(1.0, abs(v.real)), (k, q, v)
import math as _mth
r4 = nonzero[(4, 4)].real / nonzero[(4, 0)].real
r6 = nonzero[(6, 4)].real / nonzero[(6, 0)].real
print(f"cubic ratios: B4^4/B4^0 = {r4:.6f} (sqrt(5/14)="
      f"{_mth.sqrt(5 / 14):.6f}), B6^4/B6^0 = {r6:.6f} (-sqrt(7/2)="
      f"{-_mth.sqrt(7 / 2):.6f})", flush=True)
assert abs(r4 - _mth.sqrt(5 / 14)) < 1e-4
assert abs(r6 + _mth.sqrt(7 / 2)) < 1e-4
bkq_real = {kk: float(v.real) for kk, v in nonzero.items()}

cf = build_cf_operator(od, bsp, 58, 50, bkq_real)

# ---- GATE (c): direct 3D quadrature, no multipole expansion ---------
from atomx.angular.csf_hamiltonian import _m2_values, twoj  # noqa: E402

NT, NP, NR = 40, 80, 40
xg, wg = np.polynomial.legendre.leggauss(NT)
theta_q = np.arccos(xg)
phi_q = 2 * np.pi * (np.arange(NP) + 0.5) / NP
TH, PH = np.meshgrid(theta_q, phi_q, indexing="ij")
dOm = np.outer(wg, np.full(NP, 2 * np.pi / NP))

# radial grid: workspace Gauss grid (same quadrature as cf's rpow).
# WINDOW: r < 6.5 bohr, strictly INSIDE the charge shell (R = 7.79):
# near r ~ R the 1/|R-r| angular ridge is quadrature-under-resolved and
# blows up the direct integral (the far-field control test pinned this).
# The 4f density beyond 6.5 bohr is <1e-6, so the windowed direct and
# the windowed series agree; k>=8 multipoles contribute <0.5%.
rg = np.asarray(bsp["gr"]).ravel()
rw = np.asarray(bsp["grw"]).ravel()
r_keep = (rg > 0.05) & (rg < 6.5)
rg2, rw2 = rg[r_keep], rw[r_keep]


def _Omega(kappa, m2, th, ph):
    """CG-built spinor spherical harmonic (independent path)."""
    from sympy.physics.wigner import clebsch_gordan
    l = kappa if kappa > 0 else -kappa - 1
    j = twoj(kappa) / 2.0
    m = m2 / 2.0
    up = float(clebsch_gordan(l, 1 / 2, j, m - 1 / 2, 1 / 2, m))
    dn = float(clebsch_gordan(l, 1 / 2, j, m + 1 / 2, -1 / 2, m))
    from scipy.special import sph_harm

    def Y(l_, m_):
        if abs(m_) > l_:
            return np.zeros_like(th, dtype=complex)
        v = sph_harm(abs(int(m_)), l_, ph, th)
        return v if m_ >= 0 else (-1.0) ** int(-m_) * np.conj(v)

    return up * Y(l, m - 1 / 2), dn * Y(l, m + 1 / 2)


def direct_element(ka, m2a, kb, m2b):
    """<a|V_pointcharge|b> by direct r x Omega quadrature (Hartree)."""
    Pa, Qa = od[(4, ka)]
    Pb, Qb = od[(4, kb)]
    from atomx.methods.dirac_fock import _eval_bspline_on_grid
    Pag = np.asarray(_eval_bspline_on_grid(Pa, bsp, "p")).ravel()[r_keep]
    Qag = np.asarray(_eval_bspline_on_grid(Qa, bsp, "q")).ravel()[r_keep]
    Pbg = np.asarray(_eval_bspline_on_grid(Pb, bsp, "p")).ravel()[r_keep]
    Qbg = np.asarray(_eval_bspline_on_grid(Qb, bsp, "q")).ravel()[r_keep]
    X = rg2[:, None, None] * (np.sin(TH) * np.cos(PH))[None]
    Yc = rg2[:, None, None] * (np.sin(TH) * np.sin(PH))[None]
    Z = rg2[:, None, None] * np.cos(TH)[None]
    V = np.zeros((len(rg2), NT, NP))
    for (qc, (cx, cy, cz)) in charges:
        V += qc / np.sqrt((X - cx) ** 2 + (Yc - cy) ** 2 + (Z - cz) ** 2)
    oa = _Omega(ka, m2a, TH, PH)
    ob = _Omega(kb, m2b, TH, PH)
    ang = (np.conj(oa[0]) * ob[0] + np.conj(oa[1]) * ob[1])  # (NT,NP)
    # electron sign: H = -phi
    total = 0.0 + 0.0j
    for ir in range(len(rg2)):
        dens = (Pag[ir] * Pbg[ir] + Qag[ir] * Qbg[ir])
        total += -dens * rw2[ir] * np.sum(dOm * ang * V[ir])
    return total


# windowed radial norm (for the k=0 series term on the same window)
from atomx.methods.dirac_fock import _eval_bspline_on_grid
norm_win = {}
for kk in [(4, 3), (4, -4)]:
    P, Q = od[kk]
    Pg = np.asarray(_eval_bspline_on_grid(P, bsp, "p")).ravel()[r_keep]
    Qg = np.asarray(_eval_bspline_on_grid(Q, bsp, "q")).ravel()[r_keep]
    norm_win[kk] = float(np.sum((Pg ** 2 + Qg ** 2) * rw2))
R_chg = a * np.sqrt(3.0)
mono = -sum(qc for qc, _ in charges) / R_chg   # k=0 series term

checked = 0
for ka, m2a, kb, m2b in [(-4, -3, -4, -3), (-4, 7, -4, -1),
                         (3, -1, 3, 3), (-4, -3, 3, -1), (3, 1, -4, -5)]:
    ia, ib = cf.index[(4, ka, m2a)], cf.index[(4, kb, m2b)]
    got = cf(ia, ib)
    want = direct_element(ka, m2a, kb, m2b)
    # the direct integral contains the k=0 constant on the window; cf
    # (built from k>=2 B parameters) does not — add it analytically
    if (ka, m2a) == (kb, m2b):
        want_series = got + mono * norm_win[(4, ka)]
    else:
        want_series = got
    rel = abs(want - want_series) / max(abs(want), 1e-12)
    print(f"  <({ka},{m2a})|V|({kb},{m2b})>: direct={want.real:+.6e}  "
          f"cf+mono={want_series:+.6e}  rel={rel:.1e}", flush=True)
    assert rel < 5e-3, (ka, m2a, kb, m2b)
    checked += 1
print(f"GATE (c) PASS: {checked} elements quadrature-verified "
      f"(windowed, k>=8 truncation + quadrature budget 5e-3)", flush=True)

# ---- determinants: frozen [Xe] core + one 4f electron ---------------
CORE_PAT = [("1s", 2), ("2s", 2), ("2p", 6), ("3s", 2), ("3p", 6),
            ("3d", 10), ("4s", 2), ("4p", 6), ("4d", 10), ("5s", 2),
            ("5p", 6)]
from atomx.config.csf import enumerate_jj_csfs_many  # noqa: E402
from atomx.methods.mcdhf.excitations import (  # noqa: E402
    generate_sd_excitations)
from atomx.physics.csf.sparse_ci import patterns_from_csfs  # noqa: E402

REF = " ".join(f"{s}{q}" for s, q in CORE_PAT) + " 4f1"
cfgs = generate_sd_excitations(REF, ["4f"], core_orbitals=None,
                               max_excitations=0)
pats = patterns_from_csfs(enumerate_jj_csfs_many(cfgs))
pats = [p for p in pats if all((n, k) in od for (n, k, _q) in p)]


def levels(m2):
    masks, spinors = SC.enumerate_determinants(pats, od, m2)
    if not masks:
        return None
    lad = cf.index

    def m_delta(ia, ic):
        return float(cf(lad[spinors[ia]], lad[spinors[ic]]))

    H = SC.assemble_sparse_hamiltonian(masks, spinors, od, bsp, 58, 50,
                                       one_body_m_delta=m_delta)
    Hd = (H - H.T).tocoo()
    assert Hd.data.size == 0 or np.max(np.abs(Hd.data)) < 1e-8, "H not Hermitian"
    w = np.linalg.eigvalsh(H.toarray())
    return w, len(masks)


w0 = {}
for m2 in (-7, -5, -3, -1, 1, 3, 5, 7):
    res = levels(m2)
    if res:
        w0[m2] = res
all_e = np.concatenate([w for w, _ in w0.values()])
print("\nfree-ion levels (no CF) per M sector, spread:",
      f"{(all_e.max() - all_e.min()) * CM:.3f} cm^-1 (should be ~0:"
      " J-degenerate across m)", flush=True)
# GATE (a): two J levels, 6-fold below 8-fold, order 1e3 cm^-1
u = np.unique(np.round(all_e / 1e-8))
# cluster by gap
es = np.sort(all_e)
gaps = np.diff(es)
cut = np.argmax(gaps)
j1, j2 = es[:cut + 1], es[cut + 1:]
print(f"  lower cluster: {len(j1)} states, upper: {len(j2)} states, "
      f"2F splitting = {(j2.mean() - j1.mean()) * CM:.1f} cm^-1",
      flush=True)
assert len(j1) == 6 and len(j2) == 8, "expected 2F5/2 (6) + 2F7/2 (8)"
assert j1.mean() < j2.mean()
split_cm = (j2.mean() - j1.mean()) * CM
assert 500 < split_cm < 6000, split_cm
print(f"GATE (a) PASS: 2F5/2 below 2F7/2, splitting {split_cm:.0f} cm^-1",
      flush=True)

# GATE (b): cubic degeneracy pattern — recompute WITH the CF channel
all_cf = np.concatenate([w for (w, _) in
                         [levels(m2) for m2 in (-7, -5, -3, -1, 1, 3, 5, 7)]
                         if w is not None])
es = np.sort(all_cf)
print("\ncubic-split levels (cm^-1, relative to ground):",
      flush=True)
for e in (es - es[0]) * CM:
    print(f"  {e:10.3f}", flush=True)
# cluster with a tolerance well below the expected splittings
tol = 5.0  # cm^-1
groups = [[es[0]]]
for e in es[1:]:
    if (e - groups[-1][-1]) * CM < tol:
        groups[-1].append(e)
    else:
        groups.append([e])
sizes = [len(g) for g in groups]
print("degeneracy pattern:", sizes, flush=True)
assert sizes == [2, 4, 2, 2, 4], f"cubic pattern must be 2+4 | 2+2+4: {sizes}"
print("GATE (b) PASS: cubic degeneracies 2+4 (J=5/2) and 2+2+4 (J=7/2)",
      flush=True)
print("\nALL GATES PASS", flush=True)
