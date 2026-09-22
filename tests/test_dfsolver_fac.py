"""DFSolver's ``exchange='gu'`` — unit-level wiring test.

Pins what ``exchange='gu'`` does today so future DFSolver refactors
don't silently regress the Gu α_xc wiring.  The B-spline Slater
normalisation bug (2× binding energy at Z=10 Ne) that the FAC plan
called out as orthogonal has since been fixed — see
``tests/test_dfsolver_slater.py`` for the closed-shell regression
gate.

The α_xc values below are FAC 1.1.5's own ``AHX`` numbers — the ``AHX_fac``
column of ``docs/plans/2026-04-20-fac-calibration/calibration.csv``, printed
by FAC's ``radial.c`` and captured in the 2026-04-20 calibration run:
Z=10 N=10 → 0.514651052, Z=26 N=10 → 0.504451201.  They are *not* atomx
output.  ``test_gu_alpha_x_reaches_the_energy`` then checks the consequence
rather than the constant: the gu run's total energy must equal a Slater run
at exactly that α_xc.
"""

from __future__ import annotations

import numpy as np
import pytest

from atomx.methods.dirac_fock import DFSolver, Subshell


class TestFacAlphaWiring:
    """``exchange='gu'`` must resolve α_xc = ahx·(4π²/3)^{1/3}."""

    def test_alpha_x_matches_gu_conversion_for_ne(self):
        """Ne (Z=10, N=10): FAC AHX = 0.514651052 → α_Slater = 1.215045."""
        subshells = [
            Subshell(n=1, kappa=-1, occupation=2),
            Subshell(n=2, kappa=-1, occupation=2),
            Subshell(n=2, kappa=1,  occupation=2),
            Subshell(n=2, kappa=-2, occupation=4),
        ]
        solver = DFSolver(
            Z=10, subshells=subshells, exchange="gu",
            fac_ihx=-1.0, fac_hxs=1.0,
        )
        expected = 0.51465105 * (4.0 * np.pi**2 / 3.0) ** (1.0/3.0)
        assert solver.alpha_x == pytest.approx(expected, rel=1e-7)

    def test_alpha_x_matches_gu_conversion_for_fe_ne_like(self):
        """Fe^{16+} (Z=26, N=10): FAC AHX = 0.504451201 → α_Slater ≈ 1.1910."""
        subshells = [
            Subshell(n=1, kappa=-1, occupation=2),
            Subshell(n=2, kappa=-1, occupation=2),
            Subshell(n=2, kappa=1,  occupation=2),
            Subshell(n=2, kappa=-2, occupation=4),
        ]
        solver = DFSolver(
            Z=26, subshells=subshells, exchange="gu",
            fac_ihx=-1.0, fac_hxs=1.0,
        )
        expected = 0.50445120 * (4.0 * np.pi**2 / 3.0) ** (1.0/3.0)
        assert solver.alpha_x == pytest.approx(expected, rel=1e-7)

    @pytest.mark.parametrize("Z,charge,ahx_fac", [
        (10, 0, 0.514651052),    # calibration.csv, tag Z010_N10_1s2_2s2_2p6
        (26, 16, 0.504451201),   # calibration.csv, tag Z026_N10_1s2_2s2_2p6
    ])
    def test_gu_alpha_x_reaches_the_energy(self, Z, charge, ahx_fac):
        """The FAC α_xc must reach the *energy*, not just a solver attribute.

        A ``gu`` run must be bit-for-bit the Slater model evaluated at
        α = AHX_fac·(4π²/3)^{1/3}; if ``exchange="gu"`` silently fell back to
        the Slater default 2/3 the two energies would differ by ~10 Ha here.
        Measured relative difference: 1.85e-13 (Ne), 1.39e-15 (Fe XVII) —
        i.e. the same SCF fixed point.  Gate 1e-9.
        """
        from atomx import Atom, Exchange

        alpha = ahx_fac * (4.0 * np.pi**2 / 3.0) ** (1.0 / 3.0)
        common = dict(Z=Z, charge=charge, config="1s2 2s2 2p6",
                      engine="shooting", corrections="none")
        gu = Atom(**common, exchange="gu").solve()
        slater = Atom(
                     **common,
                     exchange=Exchange(model="slater", alpha_x=alpha),
                 ).solve()
        rel = abs(gu.total_energy - slater.total_energy) / abs(
            slater.total_energy)
        assert rel < 1e-9, (
            f"Z={Z}: gu E={gu.total_energy!r} vs slater(alpha_x={alpha!r}) "
            f"E={slater.total_energy!r}, rel={rel:.3e}"
        )

    def test_slater_mode_leaves_alpha_x_alone(self):
        """``exchange='slater'`` must NOT re-resolve α_x."""
        subshells = [Subshell(n=1, kappa=-1, occupation=2)]
        solver = DFSolver(
            Z=2, subshells=subshells, exchange="slater", alpha_x=0.7,
        )
        assert solver.alpha_x == pytest.approx(0.7, rel=1e-12)

    def test_hf_mode_leaves_alpha_x_alone(self):
        """``exchange='hf'`` keeps alpha_x at its default 2/3."""
        subshells = [Subshell(n=1, kappa=-1, occupation=2)]
        solver = DFSolver(Z=2, subshells=subshells, exchange="hf")
        assert solver.alpha_x == pytest.approx(2.0 / 3.0, rel=1e-12)
