import copy

import pytest

from atomx.constants import HARTREE_CM1
from atomx.workflows.nist_comparison import ionization_comparisons


def fixtures():
    grounds = {(2, 0): (2, 0, "1s2"), (2, 1): (2, 1, "1s1")}
    complete = {}
    for key in grounds.values():
        energy = -2.9 if key[1] == 0 else -2.0
        complete[key] = [
            dict(
                grid=g,
                isotope_a=4,
                energy_ha=energy,
                base_energy_ha=energy - 0.01,
                corrections={"test": 0.01},
                specification={
                    "physics": {"nuclear_atomic_weight": 4},
                    "corrections": {"nuclear_mass_me": 7294.0},
                },
            )
            for g in [3200, 6400]
        ]
    references = {
        (2, 0): dict(
            ground_configuration="1s2",
            ionization_cm1=0.9 * HARTREE_CM1,
            ionization_kind="evaluated",
            uncertainty_cm1=0.1,
        ),
        (2, 1): dict(
            ground_configuration="1s",
            ionization_cm1=2 * HARTREE_CM1,
            ionization_kind="theoretical",
            uncertainty_cm1=0.01,
        ),
    }
    return grounds, complete, references


def test_ionization_uses_charge_difference_and_keeps_theoretical_limits_distinct():
    rows = ionization_comparisons(*fixtures())
    assert rows[0]["status"] == "unique_configuration_vs_evaluated_limit"
    assert rows[0]["residual_cm1"] == pytest.approx(0.0, abs=1e-9)
    assert rows[0]["correction_differences_cm1"]["test"] == 0
    assert rows[1]["status"] == "nist_theoretical_limit"
    assert rows[1]["correction_differences_cm1"]["test"] == pytest.approx(
        -0.01 * HARTREE_CM1
    )
    assert rows[1]["upper_grids"] == []


@pytest.mark.parametrize(
    "change,status",
    [
        ("mass", "endpoint_model_mismatch"),
        ("configuration", "nist_ground_configuration_unverified"),
        ("pending", "endpoints_not_ready"),
    ],
)
def test_ionization_does_not_score_unmatched_endpoints(change, status):
    grounds, complete, references = fixtures()
    if change == "mass":
        complete[(2, 1, "1s1")] = copy.deepcopy(complete[(2, 1, "1s1")])
        complete[(2, 1, "1s1")][0]["specification"]["corrections"][
            "nuclear_mass_me"
        ] = 1.0
    elif change == "configuration":
        references[(2, 0)]["ground_configuration"] = "1s.2s"
    else:
        del complete[(2, 1, "1s1")]
    row = ionization_comparisons(grounds, complete, references)[0]
    assert row["status"] == status and "residual_cm1" not in row


def test_open_shell_ground_averages_remain_diagnostics():
    _, complete, refs = fixtures()
    grounds = {(5, 0): (5, 0, "1s2 2s2 2p1"), (5, 1): (5, 1, "1s2 2s2")}
    states = {
        grounds[5, 0]: complete[2, 0, "1s2"],
        grounds[5, 1]: complete[2, 1, "1s1"],
    }
    references = {
        (5, 0): dict(refs[2, 0], ground_configuration="2s2.2p"),
        (5, 1): dict(refs[2, 1], ground_configuration="2s2"),
    }
    row = ionization_comparisons(grounds, states, references)[0]
    assert row["status"] == "configuration_average_vs_evaluated_limit"
    assert row["comparison_basis"] == "configuration_average_vs_ground_level"
