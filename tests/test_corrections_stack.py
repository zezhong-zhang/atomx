"""Flavour-aware dispatch of :class:`atomx.corrections.CorrectionStack`."""

from __future__ import annotations

import numpy as np
import pytest

from atomx import Atom
from atomx.corrections import (
    CorrectionResult,
    CorrectionStack,
)
from atomx.results import Waves

# ---------------------------------------------------------------------------
# Apply on a real DHF result
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ne_result():
    ne = Atom(Z=10).solve()
    return ne._result


class TestApplyOnDHF:
    def test_default_stack_runs_all_backends(self, ne_result):
        stack = CorrectionStack(
            se_backend="mohr", include_vp=True, include_breit=True,
        )
        applied = stack.apply(ne_result)

        expected = {"self_energy[mohr]", "uehling_vp", "breit"}
        assert expected <= set(applied)
        for name in expected:
            assert isinstance(applied[name], CorrectionResult)
            assert np.isfinite(applied[name].energy_shift)

    def test_default_stack_rung_values(self, ne_result):
        """One anchored value per rung — this is the file that *applies* them.

        Before this test the file checked only ``isfinite`` on the three
        rungs it exists to exercise, so the 3.03x Breit error and any
        future one was invisible here (the file's own ``Atom(Z=10)``
        default route is the single-configuration one, which applies the
        scalar).

        * self-energy: FAC's F(10) = 4.65413 (``faclib/coulomb.c``
          ``_qed_se0[0] + _qed_se1[0]``), quoted and gated in
          ``test_qed_self_energy_table.py``; measured here 0.012086295
          after exact Dirac-Coulomb screening normalization (2026-09-29).
        * Uehling VP: the fixed ``I(2cr)`` kernel, measured
          -0.000600937; the pre-fix ``K₁(2cr)`` kernel gave -0.002355
          (3.9x outside the band).
        * Breit: the scalar is 3.03x DBSR_HF (0.050437 vs
          0.016643630840490), so the anchor is a strict xfail; see
          ``test_qed.py::TestBreitQEDIntegration``.
        """
        stack = CorrectionStack(
            se_backend="mohr", include_vp=True, include_breit=True,
        )
        applied = stack.apply(ne_result)
        assert applied["self_energy[mohr]"].energy_shift == pytest.approx(
            0.012086294683934152, rel=1e-3)
        assert applied["uehling_vp"].energy_shift == pytest.approx(
            -0.0006009368769866071, rel=1e-3)

    @pytest.mark.external_fac
    def test_default_stack_breit_rung_matches_dbsr_hf(self, ne_result):
        """Default DHF Breit is the DBSR_HF full static average."""
        stack = CorrectionStack(
            se_backend="mohr", include_vp=True, include_breit=True,
        )
        applied = stack.apply(ne_result)
        e_breit = applied["breit"].energy_shift
        assert e_breit == pytest.approx(0.016643630840490, rel=1e-3)
        assert applied["breit"].provenance["mode"] == "dbsr_hf"
        assert applied["breit"].provenance["retardation_included"] is True
        gaunt_stack = CorrectionStack(
            se_backend=None, include_vp=False, include_breit=True,
            breit_mode="operator")
        from atomx.corrections.adapters import BreitCorrectionError
        from atomx.physics.csf.breit_ci import attach_breit_operator_energy

        with pytest.raises(BreitCorrectionError, match="absent"):
            gaunt_stack.apply(ne_result)
        attach_breit_operator_energy(ne_result)
        gaunt = gaunt_stack.apply(ne_result)["breit"]
        assert gaunt.energy_shift == pytest.approx(
            ne_result.solver_diagnostics["_E_Breit_operator"], rel=1e-12)
        assert gaunt.provenance["retardation_included"] is True
        assert gaunt.provenance["operator"] == "Gaunt plus static retardation"
        legacy = CorrectionStack(
            se_backend=None, include_vp=False, include_breit=True,
            breit_mode="scalar_legacy").apply(ne_result)["breit"]
        assert legacy.energy_shift == pytest.approx(
            0.016643630840490, rel=0.01)

    # ``test_total_matches_sum_of_applied`` was deleted here: it asserted
    # ``stack.total(applied) == sum(r.energy_shift ...)``, which is the
    # definition of ``CorrectionStack.total`` (``stack.py``:137-142), so it
    # could not fail for any physics reason (rubric class E — "a test that
    # re-implements a one-line function and checks it agrees with itself").

    def test_disabling_se_removes_it(self, ne_result):
        stack = CorrectionStack(
            se_backend=None, include_vp=True, include_breit=False,
        )
        applied = stack.apply(ne_result)
        assert {k for k in applied if not k.startswith("_")} == {"uehling_vp"}

    def test_provenance_records_method(self, ne_result):
        stack = CorrectionStack(
            se_backend="shabaev_local", include_vp=False, include_breit=False,
        )
        applied = stack.apply(ne_result)
        key = "self_energy[shabaev_local]"
        assert key in applied
        assert applied[key].provenance["backend"] == "shabaev_local"


# ---------------------------------------------------------------------------
# Skip behaviour for unsupported flavours
# ---------------------------------------------------------------------------


class _DummyShootingResult:
    """Minimal shooting-like result: no B-spline workspace, no raw orbitals."""

    _bsp_data = None
    orbitals_raw = None
    Z = 5
    total_energy = -10.0

    @property
    def waves(self):
        return Waves(grid=np.zeros(5))


class TestSkipBehaviour:
    def test_unsupported_flavour_is_skipped_not_raised(self):
        stack = CorrectionStack(
            se_backend="mohr", include_vp=True, include_breit=True,
        )
        applied = stack.apply(_DummyShootingResult())
        assert set(applied) == {"_skipped"}

    def test_skipped_provenance_explains_each_backend(self):
        stack = CorrectionStack(
            se_backend="mohr", include_vp=True, include_breit=True,
        )
        applied = stack.apply(_DummyShootingResult())
        skipped = applied["_skipped"].provenance
        # all three backends are listed with a reason mentioning bsp
        assert {"self_energy[mohr]", "uehling_vp", "breit"} <= set(skipped)
        for reason in skipped.values():
            assert "B-spline" in reason or "Z" in reason

    def test_total_ignores_skipped_entry(self):
        stack = CorrectionStack(se_backend="mohr")
        applied = stack.apply(_DummyShootingResult())
        assert stack.total(applied) == 0.0


# ---------------------------------------------------------------------------
# Custom user correction plugged via .add()
# ---------------------------------------------------------------------------


class _ConstantOffsetCorrection:
    """Tiny :class:`Correction` impl used only for the dispatch test."""

    name = "constant"
    description = "Always returns a fixed scalar offset."

    def __init__(self, value: float):
        self._value = float(value)

    def supports(self, _result):
        return True

    def evaluate(self, _result, **_):
        return CorrectionResult(name=self.name, energy_shift=self._value)


def test_add_custom_correction_is_honoured(ne_result):
    stack = (
        CorrectionStack(se_backend=None, include_vp=False, include_breit=False)
        .add(_ConstantOffsetCorrection(+0.01))
    )
    applied = stack.apply(ne_result)
    assert applied["constant"].energy_shift == pytest.approx(0.01, abs=1e-12)
