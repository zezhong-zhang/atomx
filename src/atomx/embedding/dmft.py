"""One-shot DFT (ABACUS) + self-consistent DMFT with the atomx Hubbard-I impurity.

Scheme (Lichtenstein & Katsnelson 1998 "LDA++"; Hubbard-I impurity):
  lattice   G(k,z) = [z - H(k) - W (Sigma(z) - Edc) W^dag]^{-1}     (Loewdin basis, PM)
  impurity  H_at = sum eps0_ab c_a^dag c_b + Slater(F0,F2,F4)  with eps0 = H_loc - Edc
            N0 = argmin_N [E_0(N) - mu N] (grand canonical, atomic limit)
            Sigma(z) = z - eps0 - G_at(z)^{-1}
  loop      mu (total electron count) -> n_d -> Edc(n_d) -> eps0 -> N0, Sigma -> ...

Exact pole expansion. In the cubic basis (Oh, PM, no SOC) G_at is diagonal:
G_a(z) = sum_i r_i / (z - p_i). Its zeros zeta_j interlace the poles, so
    Sigma_a(z) = Sigma_inf,a + sum_j R_j / (z - zeta_j),
    Sigma_inf,a = m1_a - eps0_a,   R_j = 1 / sum_i r_i / (zeta_j - p_i)^2  (> 0).
Each pole is an auxiliary orbital coupled to the correlated orbital with
amplitude sqrt(R_j): the lattice problem becomes a larger Hermitian matrix.
Occupations (T = 0) and real-frequency spectra are then exact — no Matsubara
truncation, no analytic continuation.

Scope (stated, not hidden): not charge self-consistent (ABACUS density is not
updated from DMFT); Hubbard-I neglects the dynamical hybridization in the
impurity solver (atomic limit), which is the standard first rung for NiO-type
charge-transfer insulators.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from atomx.embedding.abacus_cf import cubic_to_spherical_5
from atomx.embedding.impurity import (
    ImpuritySpec,
    _eigen_amplitudes,
    _hamiltonian,
    _shell_k,
    solve_atomic_limit,
)


def slater_from_UJ(U: float, J: float) -> dict:
    """d shell: F0 = U, F2 = 14 J / (1 + 0.625), F4 = 0.625 F2 (atomic ratio)."""
    F2 = 14.0 * J / 1.625
    return {"F0": U, "F2": F2, "F4": 0.625 * F2}


def edc_fll(n_d: float, U: float, J: float) -> float:
    """Fully-localized-limit double counting (per spin-orbital potential)."""
    return U * (n_d - 0.5) - J * (n_d / 2.0 - 0.5)


@dataclass
class HubbardISolution:
    N0: int
    sol: dict
    poles: list = field(default_factory=list)  # per cubic orbital: (p, r)
    sigma_inf: np.ndarray = None  # (5,)
    sigma_poles: list = field(default_factory=list)  # per orbital: (zeta, R)
    n_imp: float = 0.0


def _eps_spherical(eps_cub: np.ndarray) -> np.ndarray:
    T = cubic_to_spherical_5()
    return T.conj().T @ np.diag(eps_cub).astype(complex) @ T


def ground_energy(N: int, eps_cub, slater) -> float:
    spec = ImpuritySpec(
        n_electrons=N,
        slater_eV=slater,
        zeta_d_eV=0.0,
        cf_matrix=_eps_spherical(eps_cub),
    )
    H = _hamiltonian(N, spec, _shell_k(10))
    return float(np.linalg.eigvalsh(H)[0])


def solve_hubbard_I(
    eps_cub: np.ndarray,
    slater: dict,
    mu: float,
    T_K: float = 300.0,
    Nrange=(6, 7, 8, 9, 10),
    wtol=1e-10,
) -> HubbardISolution:
    E = {N: ground_energy(N, eps_cub, slater) - mu * N for N in Nrange}
    N0 = min(E, key=E.get)
    spec = ImpuritySpec(
        n_electrons=N0,
        slater_eV=slater,
        zeta_d_eV=0.0,
        cf_matrix=_eps_spherical(eps_cub),
        temperature_K=T_K,
    )
    sol = solve_atomic_limit(spec)
    Am, Ap = _eigen_amplitudes(sol)  # (n, g, spinorb sph)
    Tc = cubic_to_spherical_5()  # rows cubic, cols m
    eN = sol[f"N={N0}"]["evals"]
    em = sol[f"N={N0 - 1}"]["evals"]
    ep = sol[f"N={N0 + 1}"]["evals"]
    w = sol["weights_N"]
    out = HubbardISolution(N0=N0, sol=sol)
    sig_inf = np.zeros(5)
    n_imp = 0.0
    for a in range(5):
        # spin-up cubic orbital a: c_a = sum_m T[a,m] c_{m,up}   (G_cub = T G_sph T^dag)
        am = np.einsum("m,ngm->ng", Tc[a], Am[:, :, 0::2])
        ap = np.einsum("m,ngm->ng", Tc[a].conj(), Ap[:, :, 0::2])
        p, r = [], []
        for g in np.nonzero(w > 1e-12)[0]:
            p += list(eN[g] - em)
            r += list(w[g] * np.abs(am[:, g]) ** 2)
            p += list(ep - eN[g])
            r += list(w[g] * np.abs(ap[:, g]) ** 2)
        p = np.array(p)
        r = np.array(r)
        keep = r > wtol
        p, r = p[keep], r[keep]
        order = np.argsort(p)
        p, r = p[order], r[order]
        # merge (near-)degenerate poles
        P, Rr = [p[0]], [r[0]]
        for pi, ri in zip(p[1:], r[1:], strict=True):
            if pi - P[-1] < 1e-9:
                Rr[-1] += ri
            else:
                P.append(pi)
                Rr.append(ri)
        p, r = np.array(P), np.array(Rr)
        n_imp += float(r[p < mu].sum()) if False else 0.0
        out.poles.append((p, r))
        m1 = float((r * p).sum()) / float(r.sum())
        sig_inf[a] = m1 - eps_cub[a]
        def f(x, r=r, p=p):
            return float((r / (x - p)).sum())
        zetas, Rs = [], []
        for i in range(len(p) - 1):
            lo, hi = (
                p[i] + 1e-12 * max(1, abs(p[i])),
                p[i + 1] - 1e-12 * max(1, abs(p[i + 1])),
            )
            if hi <= lo:
                continue
            # f(lo+) = +inf side, f(hi-) = -inf side
            try:
                zj = brentq(f, lo + 1e-10, hi - 1e-10, xtol=1e-13, maxiter=200)
            except ValueError:
                continue
            zetas.append(zj)
            Rs.append(1.0 / float((r / (zj - p) ** 2).sum()))
        out.sigma_poles.append((np.array(zetas), np.array(Rs)))
    out.sigma_inf = sig_inf
    from atomx.embedding.impurity import occupation

    out.n_imp = float(occupation(sol).sum())
    return out


def extended_hamiltonian(
    Hl: np.ndarray, W: np.ndarray, hub: HubbardISolution, edc: float
):
    """H_ext(k): physical block + auxiliary pole orbitals (k-independent)."""
    nk, nb, _ = Hl.shape
    aux_e, aux_v = [], []
    for a in range(5):
        z, R = hub.sigma_poles[a]
        for zj, Rj in zip(z, R, strict=True):
            aux_e.append(zj)
            aux_v.append(np.sqrt(Rj) * W[:, a])
    na = len(aux_e)
    Hphys = Hl + (W @ np.diag(hub.sigma_inf - edc) @ W.conj().T)[None]
    H = np.zeros((nk, nb + na, nb + na), complex)
    H[:, :nb, :nb] = Hphys
    if na:
        V = np.stack(aux_v, 1)  # (nb, na)
        H[:, :nb, nb:] = V[None]
        H[:, nb:, :nb] = V.conj().T[None]
        H[:, nb:, nb:] = np.diag(aux_e)[None]
    return H, nb


def count(E, Vphys_w, mu, nspin=2):
    return nspin * float((Vphys_w * (mu > E)).sum()) / E.shape[0]


def dmft_loop(
    Hl,
    W,
    n_electrons,
    U,
    J,
    mu0,
    n_d0,
    *,
    maxit=40,
    mix=0.5,
    tol=1e-4,
    T_K=300.0,
    log=print,
):
    slater = slater_from_UJ(U, J)
    Hloc = np.real(np.diag(np.einsum("ia,kij,jb->ab", W.conj(), Hl, W) / len(Hl)))
    mu, n_d = mu0, n_d0
    hist = []
    for it in range(maxit):
        edc = edc_fll(n_d, U, J)
        eps0 = Hloc - edc
        hub = solve_hubbard_I(eps0, slater, mu, T_K)
        H, nb = extended_hamiltonian(Hl, W, hub, edc)
        E, V = np.linalg.eigh(H)
        wphys = (np.abs(V[:, :nb, :]) ** 2).sum(1)  # (nk, nstate)
        wd = (np.abs(np.einsum("ia,kin->kan", W.conj(), V[:, :nb, :])) ** 2).sum(1)
        lo, hi = E.min() - 1, E.max() + 1
        for _ in range(80):  # bisection on mu for the electron count
            mid = 0.5 * (lo + hi)
            if count(E, wphys, mid) < n_electrons:
                lo = mid
            else:
                hi = mid
        # place mu mid-gap if the count is flat (insulator)
        occ_e = E[(hi > E) & (wphys > 1e-6)]
        emp_e = E[(hi <= E) & (wphys > 1e-6)]
        vbm, cbm = float(occ_e.max()), float(emp_e.min())
        mu_new = 0.5 * (vbm + cbm) if cbm - vbm > 1e-3 else hi
        n_d_new = 2.0 * float((wd * (mu_new > E)).sum()) / len(Hl)
        hist.append(
            dict(
                it=it,
                mu=mu_new,
                n_d=n_d_new,
                N0=hub.N0,
                edc=edc,
                gap=max(cbm - vbm, 0.0),
                n_imp=hub.n_imp,
                n_aux=H.shape[1] - nb,
            )
        )
        log(
            f"  it {it:2d}: N0={hub.N0}  mu={mu_new:8.4f}  n_d={n_d_new:.4f}  n_imp={hub.n_imp:.3f}  Edc={edc:7.3f}  gap={max(cbm - vbm, 0):.3f} eV  aux={H.shape[1] - nb}"
        )
        conv = abs(n_d_new - n_d) < tol and abs(mu_new - mu) < 1e-3
        n_d = (1 - mix) * n_d + mix * n_d_new
        mu = mu_new
        if conv and it > 1:
            break
    return dict(
        mu=mu,
        n_d=n_d,
        hub=hub,
        edc=edc_fll(n_d, U, J),
        E=E,
        V=V,
        nb=nb,
        history=hist,
        Hloc=Hloc,
        slater=slater,
    )
