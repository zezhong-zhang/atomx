"""Regression tests for the κ > 0 heavy-Z ``dbound`` eigenvalue bug.

Prior to 2026-04-20, atomx's shooting ``dbound`` converged to a
spurious eigenvalue ~3× too deep for 2p₁/₂ and other κ > 0 orbitals
at Z ≳ 50 when the potential was bare Coulomb or weakly screened.
Root cause: ``doutw``'s past-turning-point blow-up check used a
3-point finite-difference proxy for ``P''·P`` that mis-detected the
sign for sharp heavy-Z wavefunctions, letting integration continue
into the classically-forbidden region and pick up irregular-solution
contamination that registered as a spurious radial node.  That drove
``dbound``'s "too many nodes" branch into the wrong eigenvalue.

Fix: thread the analytical power-series ``ca[2]`` coefficient out
of ``dir_integrate`` and use ``TCONV = 2·ca[2]·PPI`` exactly as in
Fortran radial.f DOUTW (line 1780).

These tests pin both the bare-Coulomb hydrogenic and the in-SCF
highly-charged-ion cases so the bug can't silently reappear.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from atomx import Atom
from atomx.constants import HARTREE_EV
from atomx.numerics.shooting.dirac_bound import dbound
from atomx.numerics.shooting.grid import sgrid
from atomx.numerics.shooting.spline import splin0


def _exact_dirac_2p_half(Z: int) -> float:
    """Exact Dirac hydrogenic 2p₁/₂ binding energy in Hartree."""
    alpha = 1.0 / 137.035999139
    gamma = math.sqrt(1.0 - (Z * alpha) ** 2)
    mc2 = 1.0 / (alpha * alpha)
    return mc2 * (1.0 / math.sqrt(1.0 + (Z * alpha / (1.0 + gamma)) ** 2) - 1.0)


@pytest.mark.parametrize("Z", [10, 30, 50, 74, 82])
@pytest.mark.parametrize("e_guess_label", ["shallow", "hydrogenic"])
def test_dbound_bare_coulomb_2p_half(Z: int, e_guess_label: str) -> None:
    """``dbound`` at bare-Coulomb 2p₁/₂ must hit the exact Dirac
    eigenvalue regardless of how shallow the initial guess is.

    Before the TCONV fix, this failed at Z ≥ 50 with both shallow
    (−0.1 Ha) and hydrogenic (−Z²/(2n²)) guesses, returning a
    spurious 3× too-deep eigenvalue.
    """
    n = 2
    r, difr, np_ = sgrid(75.0, 1.0e-7, 0.5, 1000)
    rvg = -float(Z) * np.ones(np_)
    va, vb, vc, vd = splin0(r, rvg, np_, 0.0, 0.0)

    e_guess = -0.1 if e_guess_label == "shallow" else -Z ** 2 / (2 * n * n)
    e_got, ier, p, q, il = dbound(
        e_guess, 1.0e-11, n, 1, r, rvg, va, vb, vc, vd, np_, r, np_,
    )
    e_exact = _exact_dirac_2p_half(Z)
    rel = abs(e_got - e_exact) / abs(e_exact)
    assert ier == 0 and rel < 1e-6, (
        f"Z={Z}, e_guess={e_guess_label}({e_guess:.3f}): "
        f"dbound={e_got:.6f} exact={e_exact:.6f} rel={rel:.2e} ier={ier}"
    )


@pytest.mark.parametrize(
    "Z, charge, config, sfac_eV",
    [
        (74, 64, "1s2 2s2 2p6", -301327.0),       # W⁶⁴⁺ Ne-like
        (82, 64, "1s2 2s2 2p6 3s2 3p6", -445061.0),  # Pb⁶⁴⁺ Ar-like
        (92, 82, "1s2 2s2 2p6", -494345.0),       # U⁸²⁺ Ne-like
    ],
)
def test_heavy_z_ion_shooting_matches_sfac(
    Z: int, charge: int, config: str, sfac_eV: float
) -> None:
    """Ne-/Ar-like ions at Z ≥ 74 via ``exchange='shooting'`` must now
    reproduce sfac/Fortran-DHFS total binding energy to 10⁻³
    relative — the gate tightened from 0.30 after the TCONV fix.

    This is the end-to-end sanity check for the original motivating
    bug reported in ``benchmarks/REPORT.md`` §6.4: pre-fix these rows
    over-bound by 20–30 %.
    """
    atom = Atom(Z=Z, charge=charge, config=config, exchange="shooting")
    atom.solve(max_iter=300)
    assert atom.total_energy is not None
    e_eV = atom.total_energy * HARTREE_EV
    rel = abs(e_eV - sfac_eV) / abs(sfac_eV)
    assert rel < 1e-3, (
        f"Z={Z}: atomx shooting={e_eV:.3f} eV, "
        f"sfac/Fortran-DHFS={sfac_eV:.3f} eV, rel={rel:.2e}"
    )
