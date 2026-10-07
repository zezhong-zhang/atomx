"""Anomalous X-ray scattering factors by the FAC method (2026-10-02).

f2(omega) is obtained from the SAME electric-dipole photoionization
matrix elements that drive the spectroscopy (optical theorem), and
f1(omega) from the Kramers-Kronig dispersion over that f2 — the
scattering factors are therefore consistent with the bound-free
spectroscopy BY CONSTRUCTION, not via an independent form-factor
model.  This is the resonant/dispersive counterpart of the
nonresonant Waller-Hartree module (``atomx.scattering.xray``); the
two share the independent-orbital model and the orbital set.

Conventions (atomic units): photon energies in Hartree, cross
sections in a_0^2, electron radius r_e = alpha^2 a_0.

Optical theorem
---------------
sigma_tot(omega) = (4 pi / k) Im f(0), k = alpha*omega, and
f = r_e (f0 + f' + i f2), so

    f2(omega) = omega * sigma_abs(omega) / (4 pi alpha)

(check: H 1s at threshold, sigma = 0.225 a_0^2, omega = 0.5 Ha gives
f2 = 1.23 — the literature K-edge peak value).

Dispersion
----------
    f1(omega) = (2/pi) P \\int_0^inf omega' f2(omega') / (omega^2 -
    omega'^2) d omega'

with the ω→0 limit (2/pi)\\int f2/omega' d omega' = -N_e — the
Thomas-Reiche-Kuhn sum rule in scattering form (the consistency gate
in tests/test_scattering_spectroscopy_consistency.py).
"""

from __future__ import annotations

import numpy as np

from atomx.angular.recoupling import lj_from_kappa
from atomx.config.subshell import l_letter
from atomx.constants import ALPHA


def f2_from_sigma(omega_ha: np.ndarray, sigma_a0sq: np.ndarray) -> np.ndarray:
    """Optical theorem: f2 = omega*sigma/(4 pi alpha).

    Parameters
    ----------
    omega_ha : photon energies in Hartree (nonnegative)
    sigma_a0sq : absorption cross section per ELECTRON in a_0^2

    Returns
    -------
    ndarray of the imaginary anomalous scattering factor f2.
    """
    omega_ha = np.asarray(omega_ha, dtype=float)
    sigma_a0sq = np.asarray(sigma_a0sq, dtype=float)
    if omega_ha.shape != sigma_a0sq.shape:
        raise ValueError("omega and sigma must have the same shape")
    if omega_ha.ndim != 1 or not np.all(np.isfinite(omega_ha)):
        raise ValueError("omega must be a finite 1-d array")
    if np.any(omega_ha < 0) or np.any(sigma_a0sq < 0):
        raise ValueError("omega and sigma must be nonnegative")
    return omega_ha * sigma_a0sq / (4.0 * np.pi * ALPHA)


def f1_kramers_kronig(omega_grid: np.ndarray, f2_grid: np.ndarray,
                      omega_eval: np.ndarray) -> np.ndarray:
    """Principal-value dispersion integral for f1.

    f1(w) = (2/pi) P\\int w' f2(w') / (w^2 - w'^2) dw'.

    The principal value is handled by excluding a symmetric window
    around each evaluation point (the log-grid makes the excluded
    weight second order in the grid spacing).

    Parameters
    ----------
    omega_grid, f2_grid : the f2 tabulation (strictly increasing,
        starting above zero)
    omega_eval : evaluation energies

    Returns
    -------
    ndarray of f1 at ``omega_eval``.
    """
    omega_grid = np.asarray(omega_grid, dtype=float)
    f2_grid = np.asarray(f2_grid, dtype=float)
    omega_eval = np.asarray(omega_eval, dtype=float)
    if omega_grid.ndim != 1 or omega_grid.size < 2:
        raise ValueError("omega_grid must have at least 2 points")
    if omega_grid.shape != f2_grid.shape:
        raise ValueError("omega_grid and f2_grid must have the same shape")
    if np.any(np.diff(omega_grid) <= 0) or omega_grid[0] <= 0:
        raise ValueError("omega_grid must be strictly increasing, > 0")
    out = np.zeros_like(omega_eval)
    dw = np.gradient(omega_grid)
    for i, w in enumerate(omega_eval):
        # principal value: drop the symmetric cell around w
        mask = np.abs(omega_grid - w) > 0.5 * np.min(dw)
        if not np.any(mask):
            out[i] = 0.0
            continue
        wg, fg = omega_grid[mask], f2_grid[mask]
        out[i] = (2.0 / np.pi) * np.trapezoid(
            wg * fg / (w * w - wg * wg), wg)
    return out


def trk_scattering_sum(omega_grid: np.ndarray, f2_grid: np.ndarray) -> float:
    """(2/pi) \\int f2/omega domega — the electron count by TRK.

    Equals the number of active electrons when the f2 tabulation
    covers the full photoabsorption (bound-bound plus continuum).
    Continuum-only tabulations recover the subshell electron count
    above their bound-bound content.
    """
    omega_grid = np.asarray(omega_grid, dtype=float)
    f2_grid = np.asarray(f2_grid, dtype=float)
    good = f2_grid > 0
    return float(
        (2.0 / np.pi)
        * np.trapezoid(f2_grid[good] / omega_grid[good], omega_grid[good])
    )


def _subshell_table(result):
    """(n, kappa) -> (P, Q, occupation, energy_Ha) across result types.

    ``SCFResult`` exposes ``orbitals_by_subshell()`` and label-keyed
    ``orbital_energies``; ``MCDHFResult`` exposes ``orbitals_dict`` with
    (n, kappa)-keyed occupations/energies.  Both describe the same
    physics; support either so the anomalous factors consume whatever
    the spectroscopy pipelines produce.
    """
    def _label(n, kappa):
        orbital_l, j2 = lj_from_kappa(kappa)
        return f"{n}{l_letter(orbital_l)}{j2}/2"

    table = {}
    if hasattr(result, "orbitals_by_subshell"):
        energies = getattr(result, "orbital_energies", None) or {}
        for key, (P, Q, occ) in result.orbitals_by_subshell().items():
            e = energies.get(_label(*key))
            if e is None:
                # (n, kappa)-keyed fallback on the MCDHF shape
                e = (energies.get(key) if hasattr(energies, "get") else None)
            table[key] = (P, Q, float(occ),
                          float(e) if e is not None else None)
    elif hasattr(result, "orbitals_dict"):
        occ = getattr(result, "occupations", None) or {
            (sh.n, sh.kappa): float(sh.occupation)
            for sh in getattr(result, "subshells", [])
        }
        energies = getattr(result, "orbital_energies", None) or {}
        for key, (P, Q) in result.orbitals_dict.items():
            e = energies.get(key, energies.get(_label(*key)))
            table[key] = (P, Q, float(occ.get(key, 0.0)),
                          float(e) if e is not None else None)
    else:
        raise TypeError(
            f"{type(result).__name__} exposes neither orbitals_by_subshell "
            "nor orbitals_dict"
        )
    return table


def atomic_sigma_pi(result, photon_energies_ha, *, orbitals=None) -> np.ndarray:
    """Occupation-summed independent-particle sigma_pi(omega) in a_0^2.

    Sums :func:`atomx.scattering.photoionization.direct
    .photoionization_cross_section` over the occupied subshells
    weighted by their occupations — the same orbital set and dipole
    operator as the bound-free spectroscopy.  Accepts SCFResult and
    MCDHFResult-shaped solves.
    """
    from atomx.scattering.photoionization.direct import (
        photoionization_cross_section,
    )

    bsp = getattr(result, "bsp_data", None) or result._bsp_data
    sigma = np.zeros_like(np.asarray(photon_energies_ha, dtype=float))
    for key, (P, Q, q, e_b) in _subshell_table(result).items():
        if q <= 0 or e_b is None or e_b >= 0:
            continue
        sigma += q * photoionization_cross_section(
            P, Q, key[1], e_b, photon_energies_ha,
            int(result.Z), bsp,
        )
    return sigma
