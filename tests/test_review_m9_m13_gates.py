"""Regression gates for review 2026-09-19 findings M9–M13.

One gate per finding, each pinning the *measured* before/after value:

* **M9** — hyperfine nuclear moments (¹³³Cs, ⁶³Cu), the spurious factor 2
  in the A formula documented in ``properties/hyperfine/__init__.py``,
  and the dominant missing reduced-mass term in
  ``properties/hyperfine/dipole.py`` (+610 ppm on H 1s).
* **M10** — ``dipole_polarizability`` on a shooting SCF result must raise
  a clear ValueError naming the engine and ``numerics="bspline"``, and
  the bspline path must give H α(0) = 4.49975 a₀³ vs exact 4.5.
* **M11** — ``ls_jj_transformation`` matrices are square and orthogonal
  for every (l1, l2, J) block: measured 0 of 33 non-empty blocks bad
  (|dev| ≤ 4e-15).  The review's "22 of 45 non-square" figure does not
  reproduce against this code; passing J where the API expects 2J is the
  only usage that produces rectangular blocks (15 of 45) — see the
  function docstring.  The suggested parity guards are vacuous.
* **M12** — the five stale CODATA-2018 literals are gone and each edited
  module now tracks ``atomx.constants`` (CODATA 2022); the gate also
  proves the 2018 values are *detectably* different, so a re-introduced
  literal cannot pass.
* **M13** — ``slater_condon_f`` returns |<l‖C^k‖l>|² = (2l+1)²(3j)²,
  verified against sympy's exact 3j symbols, with the explicit relation
  to the Clebsch-Gordan square (2l+1)·CG² (= 6/5 vs 2/5 for k=2, l=1).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from sympy import Rational, nsimplify, sqrt
from sympy.physics.wigner import wigner_3j

from atomx import Atom
from atomx.angular.ls_coupling import slater_condon_f
from atomx.angular.recoupling import ls_jj_transformation
from atomx.constants import (
    ALPHA,
    ALPHA_SQ,
    BOHR_RADIUS_FM,
    BOHR_RADIUS_M,
    C_SQ,
    CODATA,
    DEFAULT_CODATA_VERSION,
    PROTON_ELECTRON_MASS_RATIO,
)
from atomx.properties.hyperfine import ELECTRON_ANOMALOUS_MOMENT, nuclear_moment

SRC = Path(__file__).resolve().parent.parent / "src" / "atomx"


# ---------------------------------------------------------------------------
# M9 — hyperfine nuclear moments, factor 2, reduced mass
# ---------------------------------------------------------------------------

class TestM9NuclearMoments:
    def test_cs133_moment_is_accepted_value(self):
        """¹³³Cs μ_I: was +2.5828327 (+3.1e-4 high) → +2.58202503 μ_N
        (Stone 2019)."""
        mom = nuclear_moment(55, 133)
        assert mom.mu_I == pytest.approx(2.58202503, rel=1e-9)
        assert mom.mu_I == pytest.approx(2.58202503, abs=1e-9)
        # the old literal is rejected at the 3e-4 level
        assert abs(mom.mu_I - 2.5828327) / 2.5828327 > 3e-4
        assert mom.g_I == pytest.approx(2.58202503 / 3.5, rel=1e-12)

    def test_cu63_moment_is_accepted_value(self):
        """⁶³Cu μ_I: was +2.2273456 (+1.8e-3 high) → +2.2233 μ_N."""
        mom = nuclear_moment(29, 63)
        assert mom.mu_I == pytest.approx(2.2233, rel=1e-9)
        assert abs(mom.mu_I - 2.2273456) / 2.2273456 > 1e-3
        assert mom.g_I == pytest.approx(2.2233 / 1.5, rel=1e-12)


class TestM9HyperfineDocstrings:
    def test_A_formula_has_no_factor_two(self):
        """``hyperfine_A`` = α g_I (m_e/m_p)[κ/j(j+1)]⟨PQ/r²⟩ — the module
        docstring said ``2 α g_I …``, which would double H 1s to 2842 MHz
        (review M9)."""
        doc = (SRC / "properties/hyperfine/__init__.py").read_text()
        formula = next(ln for ln in doc.splitlines() if "A_nj =" in ln)
        assert "A_nj = α g_I" in formula
        assert "A_nj = 2" not in formula
        assert "2 α g_I" not in doc          # the pre-review text

    def test_h1s_A_is_1421_not_2842(self):
        """The implementation is the no-factor-2 form: H 1s A = 1421.27
        MHz (measured 1420.4058, +610 ppm) and the anomaly rung is exactly
        (1 + a_e)."""
        from atomx.methods.dirac_fock import DFSolver, Subshell
        from atomx.properties.hyperfine import hyperfine_A_MHz

        solver = DFSolver(Z=1, subshells=[Subshell(n=1, kappa=-1, occupation=1)],
                          ns=60, ks=9, exchange="hf")
        result = solver.solve(max_iter=50, tol=1e-8)
        P, Q, _occ, kappa, _lbl = solver._flatten_orbitals(result.orbitals_raw)[0]
        g_I = nuclear_moment(1, 1).g_I
        A = hyperfine_A_MHz(P, Q, result._bsp_data, kappa=kappa, g_I=g_I)
        assert pytest.approx(1421.272852, abs=2e-3) == A
        assert A < 1500.0  # a spurious factor 2 would give 2842.5 MHz
        A_ae = hyperfine_A_MHz(P, Q, result._bsp_data, kappa=kappa, g_I=g_I,
                               include_anomalous_moment=True)
        assert A_ae / A == pytest.approx(1.0 + ELECTRON_ANOMALOUS_MOMENT,
                                         rel=1e-12)

    def test_reduced_mass_term_is_documented_and_dominant(self):
        """M9(c): the error budget must name the reduced-mass term
        (1 + m_e/m_p)^−3 ≈ −1632 ppm — the dominant omission behind the
        +610 ppm H 1s gap.  The old docstring omitted it entirely."""
        doc = (SRC / "properties/hyperfine/dipole.py").read_text()
        # the docstring uses the Unicode minus sign
        ascii_doc = doc.replace("\u2212", "-")
        assert "reduced mass" in doc
        assert "reduced mass" in ascii_doc.split("Error budget")[1]
        assert "-1632 ppm" in ascii_doc
        assert "+610 ppm" in ascii_doc
        # the factor-3-short value (-0.054 %, i.e. one power of m_e/m_p)
        # must not be presented as the correction
        assert "-0.054" not in ascii_doc
        # physics behind the documented number
        ppm = ((1.0 + 1.0 / PROTON_ELECTRON_MASS_RATIO) ** -3 - 1.0) * 1e6
        assert ppm == pytest.approx(-1632.1, abs=0.5)
        assert pytest.approx(1418.9532, abs=1e-3) \
            == (1.0 + 1.0 / PROTON_ELECTRON_MASS_RATIO) ** -3 * 1421.272852


# ---------------------------------------------------------------------------
# M10 — polarizability engine guard + the exact H value
# ---------------------------------------------------------------------------

class TestM10PolarizabilityEngine:
    def test_shooting_result_raises_actionable_error(self):
        """A shooting SCFResult has no B-spline workspace; the old failure
        was an opaque ``TypeError: 'NoneType' object is not subscriptable``
        deep in methods/mbpt.py.  Now: ValueError naming the engine."""
        from atomx.properties.polarizability import dipole_polarizability

        atom = Atom(Z=1, config="1s1")          # default = shooting
        atom.solve()
        assert atom._result._bsp_data is None
        with pytest.raises(ValueError) as excinfo:
            dipole_polarizability(atom._result)
        msg = str(excinfo.value)
        assert "bspline" in msg
        assert "shooting" in msg          # names the engine
        assert "B-spline" in msg

    def test_bspline_h_alpha0_matches_exact(self):
        """bspline engine: α(0) = 4.49975 a₀³ vs exact 4.5 (review M10),
        i.e. the guard does not block the working path."""
        from atomx.properties.polarizability import dipole_polarizability

        atom = Atom(Z=1, config="1s1", engine="bspline")
        atom.solve()
        res = dipole_polarizability(atom._result)
        assert res.alpha_static == pytest.approx(4.49975, abs=1e-4)
        assert abs(res.alpha_static - 4.5) < 1e-3


# ---------------------------------------------------------------------------
# M11 — LS↔jj transformation unitarity
# ---------------------------------------------------------------------------

class TestM11RecouplingUnitarity:
    @pytest.mark.parametrize("l1,l2,J", [
        (1, 1, 0), (1, 1, 1), (1, 1, 2),            # p²
        (2, 2, 0), (2, 2, 1), (2, 2, 2), (2, 2, 3), (2, 2, 4),   # d²
        (0, 2, 2), (1, 2, 1), (1, 2, 3),
    ])
    def test_block_is_square_and_orthogonal(self, l1, l2, J):
        T = ls_jj_transformation(l1, l2, 2 * J)
        n, m = T.shape
        assert n == m and n > 0
        assert np.allclose(T @ T.T, np.eye(n), atol=1e-12)
        assert np.allclose(T.T @ T, np.eye(m), atol=1e-12)

    def test_full_l1_l2_table_is_unitary(self):
        """All 45 (l1, l2, J) blocks with l1, l2 ≤ 2, J ≤ 4: n_jj == n_ls
        and orthonormal to ≤ 1e-12.  Measured: 0 of the 33 non-empty
        blocks bad; the review's 22/45 figure does not reproduce (passing
        J where the API expects 2J yields 15 rectangular blocks of 45)."""
        checked = 0
        for l1 in range(3):
            for l2 in range(3):
                for J in range(5):
                    T = ls_jj_transformation(l1, l2, 2 * J)
                    n, m = T.shape
                    assert n == m
                    if n == 0:
                        continue      # J outside the |l1-l2|..l1+l2 window
                    assert np.allclose(T @ T.T, np.eye(n), atol=1e-12), (l1, l2, J)
                    checked += 1
        assert checked == 33          # 45 blocks minus the 12 empty ones


# ---------------------------------------------------------------------------
# M12 — stale CODATA-2018 literals → atomx.constants (CODATA 2022)
# ---------------------------------------------------------------------------

class TestM12CodataLiterals:
    def test_stale_2018_values_are_detectably_different(self):
        """The gate below is only meaningful if 2018 ≠ 2022 at machine
        precision (review: +1.4e-9 … +6.8e-10 relative)."""
        assert pytest.approx(
            1.372e-9, rel=1e-2) == C_SQ / (137.0359990836 ** 2) - 1.0
        assert pytest.approx(
            6.784e-10, rel=1e-2) == 5.29177210903e-11 / BOHR_RADIUS_M - 1.0
        assert 9.2740100783e-24 / (
            CODATA["2022"]["_e"] * CODATA["2022"]["_hplanck"]
            / (4.0 * np.pi * CODATA["2022"]["_me"])) - 1.0 == pytest.approx(
            1.355e-9, rel=1e-2)

    def test_five_sites_track_constants(self):
        from atomx.numerics import quadrature
        from atomx.numerics.shooting import dirac_continuum
        from atomx.plasma import average_atom
        from atomx.properties.zeeman import lande

        # 1-3. c² in quadrature.py, dirac_continuum.py and α² in born.py
        from atomx.scattering.distorted_wave import born
        assert quadrature.C_SQ == C_SQ
        assert dirac_continuum._C_SQ == C_SQ
        assert born.ALPHA2 == ALPHA_SQ
        assert born.ALPHA2 == ALPHA ** 2

        # 4. Bohr radius in average_atom.py (module-level import)
        assert average_atom.BOHR_RADIUS_M == BOHR_RADIUS_M

        # 5. Bohr magneton in lande.py, derived from the active CODATA table
        mu_B = (CODATA[DEFAULT_CODATA_VERSION]["_e"]
                * CODATA[DEFAULT_CODATA_VERSION]["_hplanck"]
                / (4.0 * np.pi * CODATA[DEFAULT_CODATA_VERSION]["_me"]))
        assert pytest.approx(mu_B, rel=1e-15) == lande._BOHR_MAGNETON_J_PER_T
        assert pytest.approx(
            13996.24491711, rel=1e-11) == lande.BOHR_MAGNETON_MHZ_PER_TESLA
        assert pytest.approx(-1.355e-9, rel=1e-2) \
            == lande.BOHR_MAGNETON_MHZ_PER_TESLA / 13996.2449361 - 1.0

    def test_g_factor_uses_codata_2022_bohr_radius(self):
        """``g_factor_finite_nuclear_size`` ∝ (R/a₀)²: using the 2018 a₀
        would shift the result by +1.36e-9 relative — well above the
        1e-12 agreement with the 2022 expression asserted here."""
        from atomx.properties.g_factor import g_factor_finite_nuclear_size

        g = g_factor_finite_nuclear_size(82, -1, 5.5)
        z_alpha = 82 * ALPHA
        g_2022 = -(8.0 / 3.0) * z_alpha ** 4 * (5.5 / BOHR_RADIUS_FM) ** 2 * 2.0
        g_2018 = -(8.0 / 3.0) * z_alpha ** 4 * (5.5 / 5.29177210903e4) ** 2 * 2.0
        assert g == pytest.approx(g_2022, rel=1e-12)
        assert abs(g / g_2018 - 1.0) > 1e-10


# ---------------------------------------------------------------------------
# M13 — slater_condon_f convention (|<l||C^k||l>|², not the CG square)
# ---------------------------------------------------------------------------

class TestM13SlaterCondonF:
    def test_matches_reduced_matrix_element_square(self):
        """c^k(l,l) = (2l+1)²(l k l;000)² = |<l||C^k||l>|² against sympy's
        exact 3j symbols for every even k ≤ 2l, l ≤ 3."""
        for l in range(4):
            for k in range(0, 2 * l + 1, 2):
                expected = float((2 * l + 1) ** 2
                                 * wigner_3j(l, k, l, 0, 0, 0) ** 2)
                assert slater_condon_f(k, l) == pytest.approx(expected, rel=1e-12)

    def test_closed_form_values(self):
        """Exact rationals: f(0,0)=1, f(0,1)=3, f(2,1)=6/5, f(0,2)=5,
        f(2,2)=f(4,2)=10/7, f(2,3)=28/15."""
        for (k, l), exact in {
            (0, 0): Rational(1), (0, 1): Rational(3), (2, 1): Rational(6, 5),
            (0, 2): Rational(5), (2, 2): Rational(10, 7),
            (4, 2): Rational(10, 7), (2, 3): Rational(28, 15),
        }.items():
            assert slater_condon_f(k, l) == pytest.approx(float(exact), rel=1e-12)
        # the docstring's pre-review claim (CG² = 3j² = 2/15 for k=2,l=1)
        # is a different quantity, smaller by 2l+1
        assert slater_condon_f(2, 1) == pytest.approx(1.2, rel=1e-12)
        assert slater_condon_f(2, 1) != pytest.approx(2.0 / 15.0, rel=1e-6)

    def test_relation_to_clebsch_gordan_square(self):
        """f(k,l) = (2l+1)·CG² with CG = <l 0 k 0|l 0>:
        CG² = (2l+1)(3j)², so the documented ratio is exactly 2l+1."""
        for l in range(4):
            for k in range(0, 2 * l + 1, 2):
                cg_sq = float((2 * l + 1)
                              * nsimplify(wigner_3j(l, k, l, 0, 0, 0) ** 2))
                assert slater_condon_f(k, l) / cg_sq == pytest.approx(
                    2 * l + 1, rel=1e-12)
        # explicit (k, l) = (2, 1) triple quoted in the docstring
        cg_21 = float(nsimplify(3 * wigner_3j(1, 2, 1, 0, 0, 0) ** 2))
        assert cg_21 == pytest.approx(0.4, rel=1e-12)          # 2/5
        assert float(nsimplify(wigner_3j(1, 2, 1, 0, 0, 0) ** 2)) == \
            pytest.approx(float(sqrt(Rational(2, 15)) ** 2), rel=1e-12)

    def test_selection_rules(self):
        assert slater_condon_f(1, 1) == 0.0        # odd k
        assert slater_condon_f(4, 1) == 0.0        # k > 2l

    def test_docstring_and_comment_agree_with_code(self):
        src = (SRC / "angular/ls_coupling.py").read_text()
        block = src[src.index("def slater_condon_f"):]
        assert "Clebsch-Gordan" in block            # relation is spelled out
        assert "(2l+1)² · (l k l; 0 0 0)²" in block  # docstring formula
        assert "(2l+1)^2 * (l k l;000)^2" in block   # comment formula
        assert re.search(r"return \(2 \* l \+ 1\) \*\* 2 \* w3j \*\* 2", block)
