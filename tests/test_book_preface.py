"""Executable claims from the book's preface (docs/book/vol1/frontmatter/preface.md).

The preface promises: "The book has a test suite, and it fails when a claim
fails to verify — starting with the claims in this preface."  This file is
that gate for the preface.  Run:  pytest -m book

Companion script: docs/book/vol1/frontmatter/calc/preface_numbers.py

What is (and is not) verifiable here (2026-09-17 validation-integrity pass)
-------------------------------------------------------------------------
The previous version asserted relations among hardcoded literals for the
g-factor and optical-clock claims and never called atomx, although the ``book``
marker promises "a claim that fails to verify fails this gate".  Now:

* H 1s Dirac 11-digit claim   -> computed live with both atomx engines.
* Si13+ bound g-factor claim  -> computed live with atomx's g-factor library
  (``g_factor``), compared against the Sturm 2011 theory value.  NOTE: the
  preface's "few parts in ten billion" is a property of the *cited theory vs
  experiment*; atomx reproduces the cited theory only to 1.5e-6 (measured
  2026-09-17: atomx 1.9953504546 vs theory 1.9953489580), so the preface
  sentence is a literature comparison, not an atomx precision claim, and the
  live gate below is sized to atomx's actual 1e-6 accuracy.
* optical-clock nineteenth-digit claim -> NOT atomx-computable (it is the
  Sr clock systematic uncertainty from Aeppli et al. 2024).  The tautological
  literal assertion is deleted; the citation is recorded in
  ``test_optical_clock_claim_is_not_atomx_computable`` as documentation only.
* "the call that solves hydrogen solves tungsten unchanged" -> live (slow).
"""
import numpy as np
import pytest

pytestmark = pytest.mark.book


def test_hydrogen_dirac_1s_eleven_digits():
    # Preface: "an engine that cannot reproduce the analytic Dirac ground
    # state to eleven digits has no business computing tungsten."
    # Both engines must deliver the point-nucleus H 1s eigenvalue to 1e-11
    # relative against the Sommerfeld formula.
    from atomx import Atom, Sampling
    from atomx.constants import ALPHA, C

    exact = C ** 2 * (np.sqrt(1.0 - ALPHA ** 2) - 1.0)  # -0.5000066565953 Eh
    for numerics in ("bspline", "shooting"):
        # The claim is against the PURE analytic Dirac eigenvalue —
        # run correction-free (breit/qed default True in Atom).  This is
        # a publication-precision claim, not a default-grid smoke test:
        # keep the assertion engine-facing while selecting the documented
        # high-accuracy sampling tier explicitly.
        result = Atom(Z=1, nuclear="point", numerics=numerics,
                      breit=False, qed=False,
                      sampling=Sampling.publication).solve()
        e = result.orbital_energies["1s1/2"]
        assert abs(e / exact - 1.0) < 1e-11, numerics


def test_bound_g_factor_atomx_vs_sturm2011():
    # Preface: "a bound electron's magnetic moment, computed and measured,
    # agrees to a few parts in ten billion."
    # g(28Si13+) exp: Sturm et al., PRL 107, 023002 (2011):
    #   1.995 348 958 7 (5)(3)(8)
    # theory quoted in the same comparison:
    #   1.995 348 958 0 (17)
    #
    # This test computes g from atomx and compares it with the cited theory
    # value.  Measured 2026-09-17: atomx g_total(Si13+, 1s1/2) = 1.9953504546,
    # i.e. 1.5e-6 above the cited theory — three orders of magnitude coarser
    # than the "few parts in ten billion" the preface quotes.  The gate is
    # therefore sized to atomx's real accuracy; the 1e-10 statement is about
    # the literature pair (exp vs theory), not about atomx.
    from atomx import Atom
    from atomx.properties.g_factor import g_factor

    g_theory = 1.9953489580  # Sturm 2011, as quoted in the preface
    g_exp = 1.9953489587
    # Literature pair (no atomx): "a few parts in ten billion".
    lit_rel = abs(g_exp - g_theory) / g_theory
    assert 1e-10 < lit_rel < 1e-9

    atom = Atom(Z=14, config="1s1", numerics="bspline", breit=False, qed=False)
    atom.solve()
    # Si nuclear rms charge radius ~3.12 fm (IAEA/CODATA-adjacent value).
    g_atomx = g_factor(atom._result, "1s1/2", R_rms_fm=3.12, include_fns=True).g_total
    rel = abs(g_atomx - g_theory) / g_theory
    assert rel <= 5.0e-6, (
        f"atomx g(28Si13+) = {g_atomx:.10f} vs cited theory {g_theory:.10f} "
        f"(rel {rel:.1e}); preface quotes 1e-10 — atomx's g-factor library is "
        "not at that level"
    )


def test_optical_clock_claim_is_not_atomx_computable():
    # Preface: "optical clocks uncertain only in the nineteenth digit."
    # Aeppli et al., PRL 133, 023401 (2024): 8.1e-19 total systematic (Sr).
    # This is an experimental metrology number with no atomx-computable
    # content; the old test only asserted 1e-19 <= 8.1e-19 < 1e-18, which is
    # true of the literal and verifies nothing.  Recorded here as a citation
    # so the claim's absence from the executable set is explicit.
    u_sys_sr = 8.1e-19
    assert u_sys_sr > 0.0  # citation recorded; NOT an atomx verification


@pytest.mark.slow
def test_tungsten_same_call():
    # Preface: "The call that solves hydrogen solves tungsten unchanged."
    from atomx import Atom

    result = Atom(Z=74).solve()
    assert result.converged
    assert len(result.subshells) > 0
