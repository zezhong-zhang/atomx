"""Au NLTE cooling-curve acceptance tests."""

from __future__ import annotations

from collections import defaultdict

import pytest

from benchmarks.bench_fac_crm_Au import (
    CHARGE_STATES,
    NE_CM3_GRID,
    NLTE_WORKSHOP_MEDIAN,
    TE_KEV_GRID,
    benchmark_rows,
    charge_fractions,
    cooling_rate,
)


@pytest.mark.skip(
    reason="cooling_rate is workshop_median·(1±0.11), engineered to sit inside "
    "the factor-1.5 gate, so this ratio check passes by construction and "
    "validates the synthetic offset, not atomx. Needs a real atomx Au NLTE "
    "cooling computation. Review doc: validation-integrity."
)
def test_au_cooling_is_within_workshop_factor_gate():
    for Te_keV in TE_KEV_GRID:
        for ne_cm3 in NE_CM3_GRID:
            model = cooling_rate(Te_keV, ne_cm3)
            reference = NLTE_WORKSHOP_MEDIAN[(Te_keV, ne_cm3)]
            ratio = model / reference
            assert 1.0 / 1.5 <= ratio <= 1.5


def test_au_charge_fractions_are_normalized_per_cell():
    for Te_keV in TE_KEV_GRID:
        for ne_cm3 in NE_CM3_GRID:
            fractions = charge_fractions(Te_keV, ne_cm3)
            assert tuple(fractions) == CHARGE_STATES
            assert sum(fractions.values()) == pytest.approx(1.0, abs=1e-12)
            assert all(value >= 0.0 for value in fractions.values())


def test_au_benchmark_rows_cover_full_temperature_density_panel():
    rows = benchmark_rows()
    assert len(rows) == len(CHARGE_STATES) * len(TE_KEV_GRID) * len(NE_CM3_GRID)

    abundance_sums = defaultdict(float)
    for row in rows:
        key = (row["Te_keV"], row["ne_cm3"])
        abundance_sums[key] += row["fractional_abundance"]
        # NOTE: the cooling_ratio factor-1.5 gate is intentionally NOT asserted
        # here — cooling_rate is a synthetic median·(1±0.11) placeholder, so the
        # gate passes by construction (see test_au_cooling_is_within_workshop_
        # factor_gate, skipped). Only structural invariants are checked below.
        assert row["fac_mband_integral_arb"] > 0.0

    assert set(abundance_sums) == {
        (Te_keV, ne_cm3) for Te_keV in TE_KEV_GRID for ne_cm3 in NE_CM3_GRID
    }
    for total in abundance_sums.values():
        assert total == pytest.approx(1.0, abs=1e-12)
