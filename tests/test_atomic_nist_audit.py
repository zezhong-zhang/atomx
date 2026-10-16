"""Independent matching/coverage/convention/identity controls for resolved panels."""

import copy
from dataclasses import replace

import pytest

from atomx.constants import HARTREE_CM1
from atomx.reference_data.matching import observed_vacuum_nm
from tools.atomic_nist_audit import (
    IDENTITY_FIELDS,
    SpectralState,
    audit_spectrum,
    interval_standard_uncertainty,
    selection_rule,
)

KEYS = ((6, -1), (6, 1), (6, -2), (5, 2), (5, -3))


def panel():
    identity = {
        name: {"binding": ["abstract fixed identity"]} for name in IDENTITY_FIELDS
    }
    identity["units"] = "Ha"
    states, levels = [], []
    for i, (key, twoj, parity, config, term, energy) in enumerate(
        zip(
            KEYS,
            (1, 1, 3, 3, 5),
            (1, -1, -1, 1, 1),
            ("4f14 6s", "4f14 6p", "4f14 6p", "4f14 5d", "4f14 5d"),
            ("2S", "2P*", "2P*", "2D", "2D"),
            (0, 10000, 12000, 8000, 8500),
            strict=True,
        )
    ):
        states.append(
            SpectralState(
                key, 70, 1, config, twoj, parity, -0.5 + energy / HARTREE_CM1, identity
            )
        )
        levels.append(
            dict(
                z=70,
                charge=1,
                row_index=i,
                source_row=i + 100,
                configuration=config,
                configuration_nist=config,
                term=term,
                j_text=f"{twoj}/2",
                energy_cm1=float(energy),
                uncertainty_cm1=0.0 if i == 0 else None,
                reference="abstract independent control",
                flags="[]",
                bracketed=False,
                energy_kind="evaluated",
                level_id=f"fixture-{i}",
            )
        )
    return states, levels


def line(levels, upper=1, medium="vacuum", observed=1000.0, ritz=None):
    lower, higher = levels[0], levels[upper]
    return dict(
        z=70,
        charge=1,
        lower_level_id=lower["level_id"],
        upper_level_id=higher["level_id"],
        lower_configuration=lower["configuration"],
        upper_configuration=higher["configuration"],
        lower_j=lower["j_text"],
        upper_j=higher["j_text"],
        lower_term=lower["term"],
        upper_term=higher["term"],
        observed_nm=observed,
        ritz_nm=ritz,
        medium=medium,
        observed_unc_nm=None,
        source_row=999,
        line_reference="abstract wavelength control",
        transition_type="",
    )


def audit(states, levels, lines=()):
    return audit_spectrum(
        states,
        levels,
        lines,
        zero_key=KEYS[0],
        required_keys=KEYS,
        splitting_pairs=((KEYS[1], KEYS[2]), (KEYS[3], KEYS[4])),
        transition_targets=tuple(
            (KEYS[0], k, "E1" if k[0] == 6 else "E2") for k in KEYS[1:]
        ),
    )


def test_exact_levels_keep_ground_out_of_score_and_missing_lines_out_of_pass():
    states, levels = panel()
    value = audit(states, levels)
    assert value["excitation"][0]["error"] is None
    assert value["excitation"][0]["goal_pass"] is None
    assert value["metrics"]["excitation_cm1"]["scored"] == 4
    assert value["metrics"]["excitation_cm1"]["complete_and_all_pass"]
    assert value["metrics"]["splitting_cm1"]["scored"] == 2
    assert value["metrics"]["observed_wavelength_nm"]["scored"] == 0
    assert not value["metrics"]["observed_wavelength_nm"]["complete_and_all_pass"]
    assert all(row["error"] is None for row in value["transitions"])
    assert all(row["standardized_residual"] is None for row in value["excitation"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("z", 71),
        ("charge", 0),
        ("j_text", "5/2"),
        ("term", "2P"),
        ("energy_kind", "theoretical"),
        ("bracketed", True),
        ("flags", '["?"]'),
        ("configuration_nist", "4f14 6p?"),
        ("energy_cm1", None),
    ],
)
def test_reference_mismatch_keeps_required_denominator(field, value):
    states, levels = panel()
    levels[1][field] = value
    result = audit(states, levels)
    assert result["status"] == "incomplete_reference_coverage"
    row = result["excitation"][1]
    assert row["error"] is None and row["goal_pass"] is None
    assert result["metrics"]["excitation_cm1"]["required"] == 4
    assert result["metrics"]["excitation_cm1"]["scored"] == 3
    assert not result["metrics"]["excitation_cm1"]["complete_and_all_pass"]


def test_duplicate_reference_not_chosen_by_energy_proximity():
    states, levels = panel()
    duplicate = {**levels[1], "energy_cm1": 9999.9, "level_id": "fixture-duplicate"}
    result = audit(states, levels + [duplicate])
    assert result["excitation"][1]["candidates"] == 2
    assert result["excitation"][1]["error"] is None


def test_zero_duplicate_partial_panel_and_mixed_space_reject():
    states, levels = panel()
    with pytest.raises(ValueError, match="unique evaluated zero"):
        audit(states, levels + [copy.deepcopy(levels[0])])
    with pytest.raises(ValueError, match="complete, unique"):
        audit(states[:-1], levels)
    changed = {name: "different" for name in IDENTITY_FIELDS}
    changed["units"] = "Ha"
    with pytest.raises(ValueError, match="all states must share"):
        audit(states[:-1] + [replace(states[-1], physical_identity=changed)], levels)


def test_binding_is_deep_owned_and_immutable():
    identity = {name: {"nested": [1, 2]} for name in IDENTITY_FIELDS}
    identity["units"] = "Ha"
    state = SpectralState(KEYS[0], 70, 1, "4f14 6s", 1, 1, -0.5, identity)
    identity["space"]["nested"][0] = 999
    assert state.physical_identity["space"]["nested"] == (1, 2)
    with pytest.raises(TypeError):
        state.physical_identity["space"]["nested"] = (999,)


@pytest.mark.parametrize(
    "field,value",
    [
        ("z", True),
        ("charge", False),
        ("z", 70.0),
        ("two_j", True),
        ("two_j", 1.0),
        ("parity", True),
        ("energy_Ha", True),
    ],
)
def test_bool_or_noninteger_identity_rejected(field, value):
    states, _ = panel()
    with pytest.raises(ValueError):
        replace(states[0], **{field: value})


def test_observed_not_ritz_or_level_derived():
    states, levels = panel()
    result = audit(states, levels, [line(levels, observed=None, ritz=1000.0)])
    assert result["transitions"][0]["ritz_candidates"] == 1
    assert result["transitions"][0]["observed_candidates"] == 0
    assert result["transitions"][0]["error"] is None
    result = audit(states, levels, [line(levels)])
    assert result["metrics"]["observed_wavelength_nm"]["scored"] == 1
    assert result["metrics"]["observed_wavelength_nm"]["required"] == 4
    assert result["transitions"][0]["observed_vacuum_nm"] == 1000.0
    assert not result["transitions"][0]["rate_or_intensity_computed"]
    duplicate = audit(states, levels, [line(levels), line(levels, observed=1000.001)])
    assert duplicate["transitions"][0]["observed_candidates"] == 2
    assert duplicate["transitions"][0]["error"] is None


def test_medium_conversion_and_unknown_medium_never_scores():
    states, levels = panel()
    air = audit(states, levels, [line(levels, medium="air", observed=999.0)])
    assert air["transitions"][0]["observed_vacuum_nm"] == observed_vacuum_nm(
        999.0, "air"
    )
    assert air["transitions"][0]["observed_vacuum_nm"] != 999.0
    unknown = audit(states, levels, [line(levels, medium="unspecified")])
    assert unknown["transitions"][0]["matching"] == "unknown_medium"
    assert unknown["transitions"][0]["error"] is None
    wrong = line(levels)
    wrong["upper_configuration"] = "4f13 5d 6s"
    with pytest.raises(ValueError, match="endpoint IDs"):
        audit(states, levels, [wrong])


def test_parity_and_triangle_distinguish_E1_E2_and_J0_forbidden():
    states, _ = panel()
    assert selection_rule(states[0], states[1], "E1")["symmetry_allowed"]
    assert not selection_rule(states[0], states[3], "E1")["symmetry_allowed"]
    assert selection_rule(states[0], states[3], "E2")["symmetry_allowed"]
    assert selection_rule(states[3], states[4], "M1")["symmetry_allowed"]
    j0 = replace(states[0], two_j=0)
    assert not selection_rule(j0, j0, "E1")["angular_allowed"]
    with pytest.raises(ValueError, match="explicit"):
        selection_rule(states[0], states[1], "E3")


def test_uncertainty_needs_covariance_never_digits_or_numeric_residuals():
    assert interval_standard_uncertainty(None, 1.0) is None
    assert interval_standard_uncertainty(1.0, 2.0) is None
    assert interval_standard_uncertainty(0.0, 2.0) == 2.0
    assert interval_standard_uncertainty(1.0, 2.0, 2.0) == 1.0
    assert interval_standard_uncertainty(1.0, 2.0, 0.0) == pytest.approx(5**0.5)
    with pytest.raises(ValueError, match="Cauchy"):
        interval_standard_uncertainty(1.0, 2.0, 2.01)
    states, levels = panel()
    levels[1]["uncertainty_cm1"] = 1.0
    value = audit(states, levels)
    assert value["excitation"][1]["reference_standard_uncertainty_cm1"] == 1.0
    assert value["excitation"][1]["standardized_residual"] is None


def test_failed_gap_requests_bind_all_states_without_fitting_or_dispatch():
    states, levels = panel()
    states[1] = replace(states[1], energy_Ha=states[1].energy_Ha + 0.001)
    value = audit(states, levels)
    assert not value["metrics"]["excitation_cm1"]["complete_and_all_pass"]
    request = value["iteration_requests"][0]
    assert request["all_required_state_keys"] == list(KEYS)
    assert request["accuracy_goal_unchanged"] == 0.01
    assert not request["diagnostic_causes_proven"]
    assert not value["highest_level_reached"]


@pytest.mark.parametrize(
    "lower,upper,covariance", [(0.0, 1.0, 1e-16), (1e-20, 1e-20, 1e-30)]
)
def test_exact_zero_and_tiny_uncertainty_reject_invalid_covariance(
    lower, upper, covariance
):
    with pytest.raises(ValueError, match="Cauchy"):
        interval_standard_uncertainty(lower, upper, covariance)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_uncertainty_scaling_preserves_finite_nonzero_results(scale):
    value = interval_standard_uncertainty(scale, scale, 0.0)
    assert value > 0
    assert value / scale == pytest.approx(2**0.5)
