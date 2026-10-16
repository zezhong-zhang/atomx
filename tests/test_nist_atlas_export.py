"""Atlas must not turn ground/ambiguous/theoretical references into passes."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "atlas_export", Path(__file__).resolve().parents[1] / "tools/export_nist_atlas.py"
)
atlas = importlib.util.module_from_spec(spec)
spec.loader.exec_module(atlas)


def record(index, config, j, energy, term="2S", kind="evaluated"):
    return dict(
        z=70,
        charge=1,
        row_index=index,
        source_row=1,
        configuration=config,
        configuration_nist=config,
        term=term,
        j_text=j,
        energy_cm1=energy,
        uncertainty_cm1=None,
        reference="Ltest",
        flags="[]",
        bracketed=False,
        energy_kind=kind,
        level_id=str(index),
    )


def test_nominal_match_keeps_zero_and_reference_versions_separate():
    levels = [record(0, "4f14.6s", "1/2", 0), record(1, "4f14.6p", "1/2", 27062, "2P*")]
    rows = [
        dict(key=[6, -1]),
        dict(
            key=[6, 1],
            parity="odd",
            twoJ=1,
            computed_cm1=27312,
            experimental_cm1=27061.82,
        ),
    ]
    result = atlas.match_yb(rows, levels)
    assert len(result) == 1
    assert result[0]["reference_cm1"] == 27062
    assert result[0]["error_cm1"] == 250
    assert result[0]["asd_minus_handbook_cm1"] == pytest.approx(0.18)
    assert result[0]["uncertainty_cm1"] is None


@pytest.mark.parametrize(
    "change", ["duplicate", "theoretical", "wrong_parity", "wrong_j"]
)
def test_ambiguous_or_wrong_identity_is_unscored(change):
    levels = [record(0, "4f14 6s", "1/2", 0), record(1, "4f14 6p", "1/2", 27062, "2P*")]
    if change == "duplicate":
        levels.append(dict(levels[-1], row_index=2))
    elif change == "theoretical":
        levels[-1]["energy_kind"] = "theoretical"
    elif change == "wrong_parity":
        levels[-1]["term"] = "2P"
    else:
        levels[-1]["j_text"] = "3/2"
    r = atlas.match_yb(
        [
            dict(
                key=[6, 1],
                parity="odd",
                twoJ=1,
                computed_cm1=27312,
                experimental_cm1=27061.82,
            )
        ],
        levels,
    )[0]
    assert r["matching"] == "ambiguous_or_missing"
    assert "goal_pass" not in r and "error_cm1" not in r


def test_all_periodic_positions_unique():
    positions = atlas.positions()
    assert len(set(map(tuple, positions.values()))) == 118
    assert positions[57] == [9, 3]
    assert positions[71] == [9, 17]
    assert positions[118] == [7, 18]
