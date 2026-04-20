"""DFSolver's ``exchange='fac'`` — unit-level wiring test.

Pins what ``exchange='fac'`` does today so future DFSolver refactors
don't silently regress the Gu α_xc wiring.  The B-spline Slater
normalisation bug (2× binding energy at Z=10 Ne) that the FAC plan
called out as orthogonal has since been fixed — see
``tests/test_dfsolver_slater.py`` for the closed-shell regression
gate.  Flipping the ``Atom(..., exchange='fac').solve()`` facade
away from the shooting backend onto DFSolver is now a one-line
change in ``atom.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from atomx.methods.dirac_fock import DFSolver, Subshell


class TestFacAlphaWiring:
    """``exchange='fac'`` must resolve α_xc = ahx·(4π²/3)^{1/3}."""

    def test_alpha_x_matches_gu_conversion_for_ne(self):
        """Ne (Z=10, N=10): ahx_Gu = 0.51465105 → α_Slater = 1.2150."""
        subshells = [
            Subshell(n=1, kappa=-1, occupation=2),
            Subshell(n=2, kappa=-1, occupation=2),
            Subshell(n=2, kappa=1,  occupation=2),
            Subshell(n=2, kappa=-2, occupation=4),
        ]
        solver = DFSolver(
            Z=10, subshells=subshells, exchange="fac",
            fac_ihx=-1.0, fac_hxs=1.0,
        )
        expected = 0.51465105 * (4.0 * np.pi**2 / 3.0) ** (1.0/3.0)
        assert solver.alpha_x == pytest.approx(expected, rel=1e-7)

    def test_alpha_x_matches_gu_conversion_for_fe_ne_like(self):
        """Fe^{16+} (Z=26, N=10): ahx_Gu = 0.50445120 → α_Slater ≈ 1.1910."""
        subshells = [
            Subshell(n=1, kappa=-1, occupation=2),
            Subshell(n=2, kappa=-1, occupation=2),
            Subshell(n=2, kappa=1,  occupation=2),
            Subshell(n=2, kappa=-2, occupation=4),
        ]
        solver = DFSolver(
            Z=26, subshells=subshells, exchange="fac",
            fac_ihx=-1.0, fac_hxs=1.0,
        )
        expected = 0.50445120 * (4.0 * np.pi**2 / 3.0) ** (1.0/3.0)
        assert solver.alpha_x == pytest.approx(expected, rel=1e-7)

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
