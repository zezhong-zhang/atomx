"""Antiferromagnetic (AFM-II) ED-DMFT on a two-sublattice cell.

The two Ni sites A, B are related by spin reversal, so one impurity is solved:
    Sigma_A,s(z) = Sigma_imp,s(z),    Sigma_B,s(z) = Sigma_imp,-s(z).
For each spin channel s the lattice problem is an extended Hermitian matrix (auxiliary
pole orbitals of site A from spin s, of site B from spin -s; see dmft.extended_hamiltonian).
The bath of the impurity is spin dependent and is projected from the spin-resolved
real-axis hybridization of site A (ed_dmft.project_bath).

Symmetry breaking: a staggered field h on the impurity levels for the first `seed_its`
iterations only; afterwards the moment must sustain itself (otherwise the solution
relaxes to the paramagnet).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from atomx.embedding.dmft import edc_fll, slater_from_UJ
from atomx.embedding.ed_dmft import (
    coulomb_cubic,
    dos_gap,
    hybridization,
    interaction_table,
    project_bath,
    solve_ed,
    symmetrize,
)


def _spin_sigma(sol, s):
    """Sigma of one spin channel as an object accepted by hybridization()."""
    return SimpleNamespace(
        sigma_inf=sol.sigma_inf_s[:, s].copy(),
        sigma_poles=[sol.sigma_poles_s[(a, s)] for a in range(5)],
    )


def afm_extended_hamiltonian(Hl, WA, WB, sol, s, edc):
    """H_ext for spin s: site A carries Sigma_s, site B carries Sigma_-s."""
    nk, nb, _ = Hl.shape
    sA, sB = _spin_sigma(sol, s), _spin_sigma(sol, 1 - s)
    aux_e, aux_v = [], []
    for sig, W in ((sA, WA), (sB, WB)):
        for a in range(5):
            z, R = sig.sigma_poles[a]
            for zj, Rj in zip(z, R, strict=True):
                aux_e.append(zj)
                aux_v.append(np.sqrt(Rj) * W[:, a])
    na = len(aux_e)
    Hphys = Hl + (WA @ np.diag(sA.sigma_inf - edc) @ WA.conj().T)[None] + (WB @ np.diag(sB.sigma_inf - edc) @ WB.conj().T)[None]
    H = np.zeros((nk, nb + na, nb + na), complex)
    H[:, :nb, :nb] = Hphys
    if na:
        Vm = np.stack(aux_v, 1)
        H[:, :nb, nb:] = Vm[None]
        H[:, nb:, :nb] = Vm.conj().T[None]
        H[:, nb:, nb:] = np.diag(aux_e)[None]
    return H, nb


def afm_dmft_loop(
    Hl,
    WA,
    WB,
    n_electrons,
    U,
    J,
    mu0,
    n_d0,
    *,
    init,
    maxit=30,
    mix=0.4,
    tol=2e-3,
    h_seed=0.5,
    seed_its=3,
    window=(-10.0, 0.0),
    edc_shift=0.0,
    log=print,
):
    """init: dict(eb, Vb, cb) per cubic orbital (e.g. the converged paramagnetic bath)."""
    slater = slater_from_UJ(U, J)
    table = interaction_table(coulomb_cubic(slater))
    Hloc = np.real(np.diag(np.einsum("ia,kij,jb->ab", WA.conj(), Hl, WA) / len(Hl)))
    eb = np.stack([np.asarray(init["eb"], float)] * 2, 1)
    Vb = np.stack([np.asarray(init["Vb"], float)] * 2, 1)
    cb = np.stack([np.asarray(init["cb"], float)] * 2, 1)
    mu, n_d = mu0, n_d0
    hist = []
    for it in range(maxit):
        edc = edc_fll(n_d, U, J) + edc_shift
        eps = Hloc - edc
        h = h_seed if it < seed_its else 0.0
        eps_s = np.stack([eps - h, eps + h], 1) + cb      # staggered seed field on site A
        sol = solve_ed(eps_s, eb, Vb, slater, mu, table=table, spins=(0, 1))
        Es, Vs, nbs = [], [], None
        for s in (0, 1):
            H, nbs = afm_extended_hamiltonian(Hl, WA, WB, sol, s, edc)
            E, V = np.linalg.eigh(H)
            Es.append(E)
            Vs.append(V)
        wph = [(np.abs(V[:, :nbs, :]) ** 2).sum(1) for V in Vs]

        def count(m, wph=wph, Es=Es):
            return sum(float((w * (m > E)).sum()) for w, E in zip(wph, Es, strict=True)) / len(Hl)

        lo, hi = min(E.min() for E in Es) - 1, max(E.max() for E in Es) + 1
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if count(mid) < n_electrons:
                lo = mid
            else:
                hi = mid
        Eall = np.concatenate([E.ravel() for E in Es])
        wall = np.concatenate([w.ravel() for w in wph])
        vbm, cbm = dos_gap(Eall.reshape(1, -1), wall.reshape(1, -1), hi)
        mu_new = 0.5 * (vbm + cbm) if cbm - vbm > 0.05 else hi
        nds = []
        for V, E in zip(Vs, Es, strict=True):
            wd = (np.abs(np.einsum("ia,kin->kan", WA.conj(), V[:, :nbs, :])) ** 2).sum(1)
            nds.append(float((wd * (mu_new > E)).sum()) / len(Hl))
        n_d_new, mom_latt = nds[0] + nds[1], nds[0] - nds[1]
        N_mid = count(mu_new)
        # spin-resolved hybridization of site A -> projected bath
        eb_new, Vb_new, cb_new = np.zeros((5, 2)), np.zeros((5, 2)), np.zeros((5, 2))
        for s in (0, 1):
            sig = _spin_sigma(sol, s)
            e_s, v_s, c_s = project_bath(
                lambda zz, E=Es[s], V=Vs[s], sig=sig, nbs=nbs, eps=eps: hybridization(E, V, WA, nbs, eps, sig, zz),  # bare eps
                mu0,
                window,
            )
            eb_new[:, s], Vb_new[:, s], cb_new[:, s] = symmetrize(e_s), symmetrize(v_s), symmetrize(c_s)
        d_bath = float(np.abs(eb_new - eb).max() + np.abs(Vb_new - Vb).max() + np.abs(cb_new - cb).max())
        hist.append(dict(it=it, mu=mu_new, gap=max(cbm - vbm, 0.0), n_d=n_d_new, m_latt=mom_latt, m_imp=sol.moment,
                         N_mid=N_mid, N0=sol.N0, h=h, conf=sol.weights))
        log(f"  it {it:2d}: h={h:.2f} N0={sol.N0} mu={mu_new:8.4f} gap={max(cbm - vbm, 0):.3f} n_d={n_d_new:.3f} "
            f"m(latt)={mom_latt:+.3f} m(imp)={sol.moment:+.3f} N(mid)={N_mid:.3f} "
            f"V eg up/dn={Vb[0, 0]:.2f}/{Vb[0, 1]:.2f} c eg up/dn={cb[0, 0]:+.2f}/{cb[0, 1]:+.2f}")
        eb = (1 - mix) * eb + mix * eb_new
        Vb = (1 - mix) * Vb + mix * Vb_new
        cb = (1 - mix) * cb + mix * cb_new
        n_d = (1 - mix) * n_d + mix * n_d_new
        mu = mu_new
        if it >= seed_its + 2 and d_bath < tol:
            break
    return dict(mu=mu, n_d=n_d, sol=sol, edc=edc_fll(n_d, U, J) + edc_shift, Es=Es, Vs=Vs, nb=nbs, history=hist,
                Hloc=Hloc, eb=eb, Vb=Vb, cb=cb)
