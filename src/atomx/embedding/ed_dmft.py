"""ED-DMFT: d shell + one discrete bath orbital per cubic d orbital, exact diagonalization.

Replaces the Hubbard-I (atomic-limit) impurity of ``dmft.py`` by an Anderson
impurity whose bath is fitted to the DMFT hybridization Delta(z). With the bath
the solver contains d8, d9L, d10L2 charge-transfer configurations — the physics
behind the main NiO photoemission peak (d8L / Zhang-Rice-like) that Hubbard-I
cannot produce.

Conventions
  spin-orbitals: d (cubic a = dz2,dxz,dyz,dx2-y2,dxy; ABACUS order) i = 2a+s, i < 10;
                 bath of orbital a: 10 + 2a + s.
  interaction:   H_U = 1/2 sum V_ijkl c_i^+ c_j^+ c_l c_k (d only), V from the atomx Slater
                 multipoles (same vraw as impurity.py) rotated spherical -> cubic.
  Delta_a(z) = V_a^2 / (z - eb_a);  Sigma_a = z - eps_a - Delta_a - 1/G_a.
Exact pole bookkeeping: G_a has a zero at eb_a with 1/G residue V_a^2 that cancels the
Delta pole; it is removed explicitly (and checked), the remaining Sigma poles are
causal (positive residues) and become auxiliary lattice orbitals (see dmft.py).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, least_squares

from atomx.embedding.abacus_cf import cubic_to_spherical_5
from atomx.embedding.dmft import count, edc_fll, extended_hamiltonian, slater_from_UJ
from atomx.embedding.impurity import _shell_k

ND = 10  # d spin-orbitals
NS = 20  # total spin-orbitals (d + bath)


# ---------------------------------------------------------------- interaction
def coulomb_cubic(slater: dict) -> np.ndarray:
    """V[i,j,k,l] for cubic spin-orbitals (i = 2a+s), H = 1/2 V c+i c+j c_l c_k."""
    vraw = _shell_k(10)
    f0, f2, f4 = slater["F0"], slater["F2"], slater["F4"]
    Vm = np.zeros((5, 5, 5, 5), complex)
    for m1, m2, m3, m4 in itertools.product(range(5), repeat=4):
        Vm[m1, m2, m3, m4] = vraw(m1 - 2, m2 - 2, m3 - 2, m4 - 2, f0, f2, f4)
    T = cubic_to_spherical_5()  # c^+_m = sum_a T[a,m] c^+_a ; c_m = sum_a T*[a,m] c_a
    Vc = np.einsum(
        "ijkl,ai,bj,ck,dl->abcd", Vm, T, T, T.conj(), T.conj(), optimize=True
    )
    V = np.zeros((ND, ND, ND, ND), complex)
    for a, b, c, d in itertools.product(range(5), repeat=4):
        v = Vc[a, b, c, d]
        if abs(v) < 1e-12:
            continue
        for s, t in itertools.product((0, 1), repeat=2):
            V[2 * a + s, 2 * b + t, 2 * c + s, 2 * d + t] = v
    return V


def _apply_c(state: int, i: int):
    if not (state >> i) & 1:
        return None, 0
    sgn = -1 if bin(state & ((1 << i) - 1)).count("1") % 2 else 1
    return state ^ (1 << i), sgn


def _apply_cdag(state: int, i: int):
    if (state >> i) & 1:
        return None, 0
    sgn = -1 if bin(state & ((1 << i) - 1)).count("1") % 2 else 1
    return state | (1 << i), sgn


def interaction_table(V: np.ndarray) -> dict:
    """H_U acting on the 2^10 d configurations: {d_state: [(d_state', amp), ...]}."""
    terms = [
        (i, j, k, l, V[i, j, k, l])
        for i, j, k, l in itertools.product(range(ND), repeat=4)
        if abs(V[i, j, k, l]) > 1e-12 and i != j and k != l
    ]
    table = {}
    for st in range(1 << ND):
        acc = {}
        for i, j, k, l, v in terms:
            s1, g1 = _apply_c(st, k)
            if s1 is None:
                continue
            s2, g2 = _apply_c(s1, l)
            if s2 is None:
                continue
            s3, g3 = _apply_cdag(s2, j)
            if s3 is None:
                continue
            s4, g4 = _apply_cdag(s3, i)
            if s4 is None:
                continue
            acc[s4] = acc.get(s4, 0) + 0.5 * v * g1 * g2 * g3 * g4
        table[st] = [(k_, a_) for k_, a_ in acc.items() if abs(a_) > 1e-12]
    return table


# ---------------------------------------------------------------- sector ED
def sector_states(N: int):
    if not 0 <= N <= NS:
        return []
    return sorted(sum(1 << i for i in c) for c in itertools.combinations(range(NS), N))


def _spin2(x):
    """(5,) -> (5, 2) spin-degenerate; (5, 2) passes through (orbital, spin)."""
    x = np.asarray(x, float)
    return np.stack([x, x], 1) if x.ndim == 1 else x


def build_sector(N, eps, eb, Vb, table):
    eps, eb, Vb = _spin2(eps), _spin2(eb), _spin2(Vb)
    states = sector_states(N)
    idx = {s: n for n, s in enumerate(states)}
    H = np.zeros((len(states), len(states)), complex)
    maskd = (1 << ND) - 1
    for n, s in enumerate(states):
        d, b = s & maskd, s >> ND
        # diagonal one-body
        e = 0.0
        for a in range(5):
            for sp in (0, 1):
                if (s >> (2 * a + sp)) & 1:
                    e += eps[a, sp]
                if (s >> (ND + 2 * a + sp)) & 1:
                    e += eb[a, sp]
        H[n, n] += e
        # interaction (d only; bath bits untouched, d indices lowest -> no extra sign)
        for d2, amp in table[d]:
            H[idx[d2 | (b << ND)], n] += amp
        # hybridization V_a (c+_{d a s} c_{b a s} + h.c.)
        for a in range(5):
            for sp in (0, 1):
                if Vb[a, sp] == 0:
                    continue
                i, j = 2 * a + sp, ND + 2 * a + sp
                for src, dst in ((j, i), (i, j)):
                    s1, g1 = _apply_c(s, src)
                    if s1 is None:
                        continue
                    s2, g2 = _apply_cdag(s1, dst)
                    if s2 is None:
                        continue
                    H[idx[s2], n] += Vb[a, sp] * g1 * g2
    return states, H


@dataclass
class EDSolution:
    N0: int
    E0: float
    poles: list = field(default_factory=list)  # per cubic orbital: (p, r) of G_imp
    sigma_inf: np.ndarray = None
    sigma_poles: list = field(default_factory=list)
    n_d_imp: float = 0.0
    weights: dict = field(
        default_factory=dict
    )  # configuration weights d^n in ground state


def solve_ed(
    eps,
    eb,
    Vb,
    slater,
    mu,
    table=None,
    Nrange=(17, 18, 19),
    T_K=100.0,
    wtol=1e-10,
    max_dim=6000,
    spins=(0,),
):
    """spins: spin channels for which G/Sigma are computed ((0,) for PM; (0, 1) for magnetic
    solutions). PM fields (poles, sigma_inf, sigma_poles) are spin-up; per-spin results are in
    poles_s / sigma_inf_s / sigma_poles_s keyed by (a, s)."""
    eps2, eb2, Vb2 = _spin2(eps), _spin2(eb), _spin2(Vb)
    if table is None:
        table = interaction_table(coulomb_cubic(slater))
    sect = {}

    def need(N):
        if N not in sect:
            if not 0 <= N <= NS:
                return False
            dim = math.comb(NS, N)
            if dim > max_dim:
                raise RuntimeError(
                    f"sector N={N} has dim {dim} > max_dim={max_dim}: impurity levels/mu are unphysical "
                    f"(eps-mu={np.round(np.asarray(eps) - mu, 2)}, eb-mu={np.round(np.asarray(eb) - mu, 2)})"
                )
            st, H = build_sector(N, eps, eb, Vb, table)
            ev, vec = np.linalg.eigh(H)
            sect[N] = (st, ev, vec)
        return True

    for N in Nrange:
        need(N)
    # E(N) - mu N is convex: walk the window until the minimum is interior
    for _ in range(6):
        N0 = min(sect, key=lambda N: sect[N][1][0] - mu * N)
        if (N0 - 1 in sect) and (N0 + 1 in sect):
            break
        if not (need(N0 - 1) and need(N0 + 1)) or (
            N0 - 1 not in sect or N0 + 1 not in sect
        ):
            raise RuntimeError(f"ground sector N0={N0} cannot be bracketed")
    else:
        raise RuntimeError("ground sector search did not converge")
    st0, e0, v0 = sect[N0]
    kT = 8.617333262e-5 * T_K
    w = np.exp(-(e0 - e0[0]) / kT)
    w /= w.sum()
    gs = np.nonzero(w > 1e-10)[0]
    sol = EDSolution(N0=N0, E0=float(e0[0]))
    # configuration weights and d occupancy
    maskd = (1 << ND) - 1
    nd_tot = 0.0
    conf = {}
    for g in gs:
        prob = np.abs(v0[:, g]) ** 2
        for n, s in enumerate(st0):
            k = bin(s & maskd).count("1")
            conf[k] = conf.get(k, 0) + w[g] * prob[n]
            nd_tot += w[g] * prob[n] * k
    sol.n_d_imp = float(nd_tot)
    sol.weights = {k: float(v) for k, v in sorted(conf.items())}
    stm, em, vm = sect[N0 - 1]
    stp, ep, vp = sect[N0 + 1]
    im = {s: n for n, s in enumerate(stm)}
    ip = {s: n for n, s in enumerate(stp)}
    sig_inf = np.zeros(5)
    sol.poles_s, sol.sigma_poles_s, sol.sigma_inf_s = {}, {}, np.zeros((5, 2))
    for sp in spins:
        for a in range(5):
            orb = 2 * a + sp
            Cm = np.zeros((len(stm), len(st0)))
            Cp = np.zeros((len(stp), len(st0)))
            for n, s in enumerate(st0):
                s1, g1 = _apply_c(s, orb)
                if s1 is not None:
                    Cm[im[s1], n] = g1
                s2, g2 = _apply_cdag(s, orb)
                if s2 is not None:
                    Cp[ip[s2], n] = g2
            Am = vm.conj().T @ Cm @ v0[:, gs]
            Ap = vp.conj().T @ Cp @ v0[:, gs]
            p = np.concatenate([np.concatenate([e0[g] - em for g in gs]), np.concatenate([ep - e0[g] for g in gs])])
            r = np.concatenate(
                [
                    np.concatenate([w[g] * np.abs(Am[:, k]) ** 2 for k, g in enumerate(gs)]),
                    np.concatenate([w[g] * np.abs(Ap[:, k]) ** 2 for k, g in enumerate(gs)]),
                ]
            )
            keep = r > wtol
            p, r = p[keep], r[keep]
            o = np.argsort(p)
            p, r = p[o], r[o]
            P, R = [p[0]], [r[0]]
            for pi, ri in zip(p[1:], r[1:], strict=True):
                if pi - P[-1] < 1e-9:
                    R[-1] += ri
                else:
                    P.append(pi)
                    R.append(ri)
            p, r = np.array(P), np.array(R)
            m1 = float((r * p).sum() / r.sum())

            def f(x, r=r, p=p):
                return float((r / (x - p)).sum())

            zs, Rs = [], []
            for i in range(len(p) - 1):
                try:
                    zj = brentq(
                        f,
                        p[i] + 1e-10 * max(1, abs(p[i])),
                        p[i + 1] - 1e-10 * max(1, abs(p[i + 1])),
                        xtol=1e-13,
                        maxiter=300,
                    )
                except ValueError:
                    continue
                zs.append(zj)
                Rs.append(1.0 / float((r / (zj - p) ** 2).sum()))
            zs, Rs = np.array(zs), np.array(Rs)
            # cancel the bath pole of Delta: zero of G at eb with 1/G residue Vb^2
            vb, ebv = Vb2[a, sp], eb2[a, sp]
            if vb != 0 and len(zs):
                j = int(np.argmin(np.abs(zs - ebv)))
                # pruned poles (w < wtol) next to the zero perturb R at ~1e-5
                if abs(zs[j] - ebv) > 1e-4 or abs(Rs[j] - vb**2) > 1e-3 * max(1, vb**2):
                    raise RuntimeError(
                        f"bath-pole cancellation failed for orbital {a} spin {sp}: zeta={zs[j]:.6f} vs "
                        f"eb={ebv:.6f}, R={Rs[j]:.6f} vs V^2={vb**2:.6f}"
                    )
                zs, Rs = np.delete(zs, j), np.delete(Rs, j)
            sol.poles_s[(a, sp)] = (p, r)
            sol.sigma_poles_s[(a, sp)] = (zs, Rs)
            sol.sigma_inf_s[a, sp] = m1 - eps2[a, sp]
            if sp == spins[0]:
                sol.poles.append((p, r))
                sol.sigma_poles.append((zs, Rs))
                sig_inf[a] = m1 - eps2[a, sp]
    # d moment (n_up - n_dn) of the thermal ground multiplet
    mom = 0.0
    for g in gs:
        prob = np.abs(v0[:, g]) ** 2
        for n, s in enumerate(st0):
            nu = sum((s >> (2 * a)) & 1 for a in range(5))
            ndn = sum((s >> (2 * a + 1)) & 1 for a in range(5))
            mom += w[g] * prob[n] * (nu - ndn)
    sol.moment = float(mom)
    sol.sigma_inf = sig_inf
    return sol


# ---------------------------------------------------------------- self-consistency
def local_green(E, V, W, nb, z, chunk=48):
    """G_loc,aa(z) per cubic orbital from the extended-Hamiltonian eigen decomposition
    (evaluated in z chunks: the (k, state, z) resolvent is the memory hot spot)."""
    proj = np.abs(np.einsum("ia,kin->kan", W.conj(), V[:, :nb, :])) ** 2  # (k, a, n)
    z = np.atleast_1d(np.asarray(z, dtype=complex))
    out = np.empty((5, len(z)), dtype=complex)
    for i in range(0, len(z), chunk):
        zc = z[i : i + chunk]
        out[:, i : i + chunk] = np.einsum(
            "kan,knz->az", proj, 1.0 / (zc[None, None, :] - E[:, :, None]), optimize=True
        )
    return out / E.shape[0]


def hybridization(E, V, W, nb, eps, sol, z):
    """Delta_a(z) = z - eps_a - Sigma_a(z) - 1/G_loc,a(z) on any complex contour."""
    z = np.atleast_1d(np.asarray(z, dtype=complex))
    G = local_green(E, V, W, nb, z)
    D = np.empty_like(G)
    for a in range(5):
        if sol is None:
            sig = 0.0 * z
        else:
            zs, Rs = sol.sigma_poles[a]
            sig = sol.sigma_inf[a] + ((Rs[None, :] / (z[:, None] - zs[None, :])).sum(1) if len(zs) else 0.0 * z)
        D[a] = z - eps[a] - sig - 1.0 / G[a]
    return D


def project_bath(delta_fn, mu_ref, window=(-10.0, 0.0), eta=0.15, dw=0.02):
    """Ligand bath from the REAL-AXIS hybridization spectrum Gamma(w) = -Im Delta(w + i eta)/pi:
        V^2 = int_window Gamma,   eb = first moment of Gamma over the window,
        c   = Re[Delta(z0) - V^2/(z0 - eb)] at z0 = mu_ref + 0.5i  (static remainder).
    The window selects the occupied ligand (O p) part. Needed because the t2g hybridization
    carries ~75% of its weight at +8..+25 eV (Ni 4s/4p), which dominates any Matsubara fit and
    drives a single pole off to the fit bound; the ligand part (V_t2g^2 ~ 1.4 eV^2) is what the
    charge-transfer physics needs."""
    om = np.arange(mu_ref + window[0], mu_ref + window[1], dw)
    D = delta_fn(om + 1j * eta)
    gam = np.clip(-D.imag / math.pi, 0.0, None)
    v2 = gam.sum(1) * dw
    eb = (gam * om[None, :]).sum(1) * dw / np.maximum(v2, 1e-12)
    z0 = np.array([mu_ref + 0.5j])
    d0 = delta_fn(z0)[:, 0]
    c = np.real(d0 - v2 / (z0[0] - eb))
    return eb, np.sqrt(v2), c


def fit_bath(Delta, z, w, mu, gap_min=0.3, window=10.0, vmax=6.0):
    """Low-energy representation of the DMFT hybridization:
        Delta(z) ~ c + V^2 / (z - eb),   mu - window <= eb <= mu - gap_min,  0 <= V <= vmax
    The constant c is the static effect of high-energy hybridization (Ni 4s/4p, O 3s ...),
    folded into the impurity level. Bounds are physical, and necessary: unbounded, a single
    pole runs to the high-energy weight (eb-mu = +60 eV) or, for weakly structured t2g, to
    infinity where c + V^2/(z-eb) degenerates into a constant (eb ~ -5000 eV)."""

    def res(x):
        eb, v, c = x
        d = c + v * v / (z - eb) - Delta
        return np.concatenate([np.sqrt(w) * d.real, np.sqrt(w) * d.imag])

    lo = [mu - window, 0.0, -50.0]
    hi = [mu - gap_min, vmax, 50.0]
    best = None
    for de in (1.0, 3.0, 6.0):
        for v0 in (0.8, 2.5):
            x0 = [mu - de, v0, float(np.real(Delta[0]))]
            r = least_squares(res, x0, bounds=(lo, hi))
            if best is None or r.cost < best.cost:
                best = r
    eb, v, c = best.x
    return float(eb), float(v), float(c)


def symmetrize(x):
    x = np.asarray(x, float).copy()
    x[[0, 3]] = x[[0, 3]].mean()
    x[[1, 2, 4]] = x[[1, 2, 4]].mean()
    return x


def dos_gap(E, w, mu, sig=0.05, frac=0.01, span=8.0):
    """Gap edges from the physical-weight DOS (Gaussian sig): the empty interval around mu where
    DOS < frac * max. Robust against isolated eigenstates carrying ~1e-3 physical weight."""
    om = np.linspace(mu - span, mu + span, 3201)
    sel = w > 1e-6
    e, ww = E[sel], w[sel]
    dos = np.zeros_like(om)
    for chunk in range(0, len(e), 20000):
        dos += (
            ww[chunk : chunk + 20000, None]
            * np.exp(-0.5 * ((e[chunk : chunk + 20000, None] - om) / sig) ** 2)
        ).sum(0)
    filled = dos > frac * dos.max()
    i0 = int(np.argmin(np.abs(om - mu)))
    if filled[i0]:
        return float(mu), float(mu)
    lo = i0
    while lo > 0 and not filled[lo]:
        lo -= 1
    hi = i0
    while hi < len(om) - 1 and not filled[hi]:
        hi += 1
    return float(om[lo]), float(om[hi])


def ed_dmft_loop(
    Hl,
    W,
    n_electrons,
    U,
    J,
    mu0,
    n_d0,
    *,
    beta=20.0,
    nw=120,
    maxit=40,
    mix=0.3,
    tol=2e-3,
    bath="project",
    window=(-10.0, 0.0),
    edc_shift=0.0,
    init=None,
    log=print,
):
    """bath = "project" (default): ligand bath from the real-axis hybridization in `window`
    (relative to the fixed reference mu0); "matsubara": bounded fit c + V^2/(z - eb).
    edc_shift: constant added to the FLL double counting (double-counting scan).
    init: optional dict(eb, Vb, cb, mu, n_d) to warm-start from a converged solution."""
    slater = slater_from_UJ(U, J)
    table = interaction_table(coulomb_cubic(slater))
    Hloc = np.real(np.diag(np.einsum("ia,kij,jb->ab", W.conj(), Hl, W) / len(Hl)))
    wn = (2 * np.arange(nw) + 1) * math.pi / beta
    wts = 1.0 / wn
    # initial bath: PBE hybridization Delta_0 = z - Hloc - 1/G0_loc
    E0, V0 = np.linalg.eigh(Hl)
    nb0 = Hl.shape[1]
    if bath == "project":
        eb, Vb, cb = project_bath(lambda zz: hybridization(E0, V0, W, nb0, Hloc, None, zz), mu0, window)
    else:
        z = mu0 + 1j * wn
        D0 = hybridization(E0, V0, W, nb0, Hloc, None, z)
        eb = np.zeros(5)
        Vb = np.zeros(5)
        cb = np.zeros(5)
        for a in range(5):
            eb[a], Vb[a], cb[a] = fit_bath(D0[a], z, wts, mu0)
    eb, Vb, cb = symmetrize(eb), symmetrize(Vb), symmetrize(cb)
    if init is not None:
        eb, Vb, cb = (np.asarray(init[k], float).copy() for k in ("eb", "Vb", "cb"))
    log(
        f"  bath(PBE): eb-mu eg={eb[0] - mu0:+.3f} t2g={eb[1] - mu0:+.3f}  V eg={Vb[0]:.3f} t2g={Vb[1]:.3f}  c eg={cb[0]:+.3f} t2g={cb[1]:+.3f}"
    )
    mu, n_d = mu0, n_d0
    if init is not None:
        mu, n_d = float(init.get("mu", mu0)), float(init.get("n_d", n_d0))
    hist = []
    for it in range(maxit):
        edc = edc_fll(n_d, U, J) + edc_shift
        eps = Hloc - edc
        sol = solve_ed(
            eps + cb, eb, Vb, slater, mu, table=table
        )  # Sigma referenced to eps + c
        H, nb = extended_hamiltonian(Hl, W, sol, edc)
        E, V = np.linalg.eigh(H)
        wph = (np.abs(V[:, :nb, :]) ** 2).sum(1)
        wd = (np.abs(np.einsum("ia,kin->kan", W.conj(), V[:, :nb, :])) ** 2).sum(1)
        lo, hi = E.min() - 1, E.max() + 1
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if count(E, wph, mid) < n_electrons:
                lo = mid
            else:
                hi = mid
        vbm, cbm = dos_gap(E, wph, hi)
        mu_new = 0.5 * (vbm + cbm) if cbm - vbm > 0.05 else hi
        n_d_new = 2.0 * float((wd * (mu_new > E)).sum()) / len(Hl)
        N_at_mu = count(E, wph, mu_new)
        # new hybridization and bath fit on a FIXED grid z = mu_ref + i w_n (Delta(z) does not depend on
        # where mu sits inside the gap; re-centering the grid on a mid-gap mu that wanders made the fit oscillate)
        if bath == "project":
            eb_new, Vb_new, cb_new = project_bath(
                lambda zz, E=E, V=V, nb=nb, eps=eps, sol=sol: hybridization(E, V, W, nb, eps, sol, zz), mu0, window
            )
        else:
            z = mu0 + 1j * wn
            Dz = hybridization(E, V, W, nb, eps, sol, z)
            eb_new = np.zeros(5)
            Vb_new = np.zeros(5)
            cb_new = np.zeros(5)
            for a in range(5):
                eb_new[a], Vb_new[a], cb_new[a] = fit_bath(Dz[a], z, wts, mu0)
        eb_new, Vb_new, cb_new = (
            symmetrize(eb_new),
            symmetrize(Vb_new),
            symmetrize(cb_new),
        )
        d_bath = float(np.abs(eb_new - eb).max() + np.abs(Vb_new - Vb).max())
        hist.append(
            dict(
                it=it,
                mu=mu_new,
                n_d=n_d_new,
                n_d_imp=sol.n_d_imp,
                N0=sol.N0,
                gap=max(cbm - vbm, 0.0),
                N_at_mu=N_at_mu,
                eb_eg=eb[0] - mu_new,
                eb_t2g=eb[1] - mu_new,
                V_eg=Vb[0],
                V_t2g=Vb[1],
                conf=sol.weights,
            )
        )
        log(
            f"  it {it:2d}: N0={sol.N0} mu={mu_new:8.4f} gap={max(cbm - vbm, 0):.3f} n_d(latt)={n_d_new:.3f} n_d(imp)={sol.n_d_imp:.3f} "
            f"N(mid)={N_at_mu:.3f} eb-mu eg={eb[0] - mu_new:+.2f} t2g={eb[1] - mu_new:+.2f} V eg={Vb[0]:.2f} t2g={Vb[1]:.2f} "
            f"c eg={cb[0]:+.2f} t2g={cb[1]:+.2f} d8/d9/d10={sol.weights.get(8, 0):.2f}/{sol.weights.get(9, 0):.2f}/{sol.weights.get(10, 0):.2f}"
        )
        d_bath += float(np.abs(cb_new - cb).max())
        eb = (1 - mix) * eb + mix * eb_new
        Vb = (1 - mix) * Vb + mix * Vb_new
        cb = (1 - mix) * cb + mix * cb_new
        n_d = (1 - mix) * n_d + mix * n_d_new
        mu = mu_new
        if d_bath < tol and it > 2:
            break
    return dict(
        mu=mu,
        n_d=n_d,
        sol=sol,
        edc=edc_fll(n_d, U, J) + edc_shift,
        E=E,
        V=V,
        nb=nb,
        history=hist,
        Hloc=Hloc,
        eb=eb,
        Vb=Vb,
        cb=cb,
    )
