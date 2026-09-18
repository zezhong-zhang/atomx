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

    def test_total_matches_sum_of_applied(self, ne_result):
        stack = CorrectionStack(
            se_backend="mohr", include_vp=True, include_breit=True,
        )
        applied = stack.apply(ne_result)
        expected = sum(
            r.energy_shift for k, r in applied.items() if k != "_skipped"
        )
        assert stack.total(applied) == pytest.approx(expected, rel=1e-12)

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
