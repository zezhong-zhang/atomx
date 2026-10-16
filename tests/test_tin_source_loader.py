"""State/provenance and resource admission at the native tin/plasma boundary."""

import json

import pytest

from atomx.workflows.tin_source import load_tin_lte_plasma


def write_fixture(root):
    data = {
        "input.json": dict(Z=50, charge=12, nucleus="120Sn Fermi", operator="Dirac-Coulomb",
                           E1="length long_wavelength"),
        "result.json": dict(levels=3, lines=1, source_unchanged=True),
        "levels.json": [dict(id="ground", excitation_eV=0, J2=0, dominant_fold=0),
                        dict(id="upper", excitation_eV=90, J2=2, dominant_fold=1),
                        dict(id="dark", excitation_eV=160, J2=4, dominant_fold=4)],
    }
    for name, value in data.items():
        (root / name).write_text(json.dumps(value))
    (root / "lines.json").write_text('[\n' + json.dumps(dict(upper="upper", lower="ground", energy_eV=90,
                                                          A_per_s=1e10)) + '\n]\n')


def test_partition_preserves_nonradiating_native_levels(tmp_path):
    write_fixture(tmp_path)
    plasma, meta, endpoints, folds = load_tin_lte_plasma(tmp_path, bin_width_eV=0.1, fwhm_eV=0.2)
    assert len(plasma.energy) == 3 and plasma.n_lines == 1
    assert endpoints.tolist() == [True, True, False]
    assert folds.tolist() == [0, 1, 4]
    assert meta["radiative_endpoint_levels"] == 2
    assert meta["maximum_endpoint_gap_defect_eV"] == 0
    assert len(meta["source_id"]) == 64 and len(meta["sha256"]) == 4
    assert plasma.lte(30)[0][2] > 0


@pytest.mark.parametrize("change", ["ion", "gap", "count", "source", "budget"])
def test_mismatched_or_over_budget_tin_data_rejected(tmp_path, change):
    write_fixture(tmp_path)
    if change in ("ion", "count", "source"):
        file = tmp_path / ("input.json" if change == "ion" else "result.json")
        payload = json.loads(file.read_text())
        key, value = {"ion": ("charge", 13), "count": ("lines", 2), "source": ("source_unchanged", False)}[change]
        payload[key] = value
        file.write_text(json.dumps(payload))
    elif change == "gap":
        file = tmp_path / "lines.json"
        file.write_text(file.read_text().replace('"energy_eV": 90', '"energy_eV": 91'))
    with pytest.raises((ValueError, MemoryError)):
        load_tin_lte_plasma(tmp_path, bin_width_eV=0.1, fwhm_eV=0.2,
                            memory_cap_bytes=1 if change == "budget" else 2**30)
