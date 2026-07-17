"""CorrectionStack injection into solvers (review E1).

The solvers' include_breit/include_qed booleans are sugar building a
CorrectionStack; a user-supplied stack overrides them and reaches the
single post-SCF seam.  Legacy energy_components keys are preserved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx import Atom  # noqa: E402
from atomx.corrections.stack import CorrectionStack  # noqa: E402


def test_flags_build_equivalent_stack():
    a = Atom(Z=10, config="1s2", charge=8, breit=True, qed=True)
    a.solve()
    c = a._result.energy_components
    for key in ("E_Breit", "E_SE", "E_VP", "E_QED_total", "se_method"):
        assert key in c
    assert c["E_QED_total"] == pytest.approx(c["E_SE"] + c["E_VP"],
                                             rel=1e-12)


def test_custom_stack_breit_only():
    """Breit on / QED off through a stack — impossible pre-injection
    without editing solver source."""
    st = CorrectionStack(se_backend=None, include_vp=False,
                         include_breit=True)
    a = Atom(Z=10, config="1s2", charge=8, correction_stack=st)
    a.solve()
    c = a._result.energy_components
    assert "E_Breit" in c
    assert "E_QED_total" not in c and "E_SE" not in c

    # and the value matches the boolean-sugar Breit (qed explicitly off
    # so the sugar stack is Breit-only, mirroring the custom stack)
    b = Atom(Z=10, config="1s2", charge=8, breit=True, qed=False)
    b.solve()
    assert c["E_Breit"] == pytest.approx(
        b._result.energy_components["E_Breit"], rel=1e-12)


def test_default_atom_has_corrections():
    """breit/qed default True (2026-07 defaults flip): a bare Atom
    carries the full default correction stack."""
    a = Atom(Z=10, config="1s2", charge=8)
    a.solve()
    c = a._result.energy_components
    for key in ("E_Breit", "E_SE", "E_VP", "E_QED_total"):
        assert key in c


def test_explicit_flags_off_no_corrections():
    """breit=False, qed=False builds no stack — no correction keys."""
    a = Atom(Z=10, config="1s2", charge=8, breit=False, qed=False)
    a.solve()
    c = a._result.energy_components
    assert "E_Breit" not in c and "E_QED_total" not in c
