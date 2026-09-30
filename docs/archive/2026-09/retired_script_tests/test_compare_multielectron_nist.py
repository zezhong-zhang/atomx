"""State identity is required before an observed line is scored."""

import sqlite3

import pytest

from scripts.compare_multielectron_nist import (
    add_model_audit, configuration_average_levels, correction_branch_warning,
    endpoint, paired_levels, uniquely_resolved_configuration)
from benchmarks.nist_asd import Ion, Level
from benchmarks.verify_nist_levels import Case, ci_model_physical_error


def level(index, configuration, term, j, energy):
    return Level("Li I", 3, 0, index, index, configuration, configuration,
                 term, j, energy, None, "", (), False)


def test_only_validated_level_endpoints_get_a_line():
    model = ("1s2 2s1", "1s2 2p1")
    case = Case("Li I", 3, model, model[0])
    ground = level(0, model[0], "2S", "1/2", 0)
    excited = level(1, model[1], "2P*", "1/2", 14903.66)
    ion = Ion("Li I", 3, 0, levels=[ground, excited])
    payload = dict(energies=[-7.4, -7.33], two_j=[1, 1],
                   parity=[1, -1], dominant=list(model))
    matched = paired_levels(case, ion, model, payload)
    assert set(matched) == {0, 1}
    assert endpoint(ion, matched, model[1], "2P*", "1/2", 14903.66) == excited
    assert endpoint(ion, matched, model[1], "2P*", "3/2", 14903.66) is None
    assert endpoint(ion, matched, model[1], "2P*", "1/2", 16000) is None


def test_configuration_mismatch_excludes_block():
    model = ("1s2 2s1", "1s2 2p1")
    case = Case("Li I", 3, model, model[0])
    ion = Ion("Li I", 3, 0, levels=[level(0, model[0], "2S", "1/2", 0),
                                    level(1, model[1], "2P*", "1/2", 14903.66)])
    payload = dict(energies=[-7.4, -7.33], two_j=[1, 1],
                   parity=[1, -1], dominant=[model[0], model[0]])
    assert set(paired_levels(case, ion, model, payload)) == {0}


def test_he_like_continuum_pseudostate_cannot_be_paired():
    model = ("1s2", "1s1 3s1")
    case = Case("He I", 2, model, "1s2")
    ion = Ion("He I", 2, 0, levels=[])
    payload = dict(energies=[-2.88, -0.46], two_j=[0, 0],
                   parity=[1, 1], dominant=list(model))
    assert "continuum pseudostate" in ci_model_physical_error(
        case, payload["energies"])
    with pytest.raises(ValueError, match="continuum pseudostate"):
        paired_levels(case, ion, model, payload)


def test_dhf_configuration_average_has_no_fine_structure():
    model = ("1s2 2s1", "1s2 2p1")
    case = Case("Li I", 3, model, model[0])
    ion = Ion("Li I", 3, 0, levels=[
        level(0, model[0], "2S", "1/2", 0),
        level(1, model[1], "2P*", "1/2", 14903.66),
        level(2, model[1], "2P*", "3/2", 14904.0)])
    energies = {model[0]: -7.4, model[1]: -7.33}
    matched = configuration_average_levels(case, ion, model, energies)
    assert matched[1] == matched[2]
    assert matched[0] == 0
    assert uniquely_resolved_configuration(ion, model[0])
    assert not uniquely_resolved_configuration(ion, model[1])


def test_large_breit_qed_excitation_jump_flags_state_branch():
    plain = dict(energies=[-162.0, -158.2], two_j=[1, 1], parity=[1, 1],
                 dominant=["3s", "4s"])
    corrected = dict(energies=[-162.0, -161.9], two_j=[1, 1], parity=[1, 1],
                     dominant=["3s", "4s"])
    assert correction_branch_warning(plain, corrected)
    corrected["energies"][1] = -158.19
    assert not correction_branch_warning(plain, corrected)


def test_audit_keeps_both_models_and_prefers_corrected_dhf(tmp_path):
    path = tmp_path / "audit.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript("""
          CREATE TABLE line_audit(z INTEGER,charge INTEGER,row_index INTEGER,
            status TEXT,calculated_nm REAL,observed_energy_error_percent REAL,
            ritz_energy_error_percent REAL,model TEXT);
          CREATE TABLE level_audit(z INTEGER,charge INTEGER,row_index INTEGER,
            status TEXT,calculated_energy_cm1 REAL,residual_cm1 REAL,model TEXT);
          INSERT INTO line_audit VALUES(3,0,1,'no_atomx_line_calculation',NULL,NULL,NULL,NULL);
          INSERT INTO line_audit VALUES(3,0,2,'no_atomx_line_calculation',NULL,NULL,NULL,NULL);
          INSERT INTO level_audit VALUES(3,0,1,'no_atomx_level_calculation',NULL,NULL,NULL);
        """)
    common = dict(z=3, charge=0, ritz_nm=None, lower_level_row=0,
                  upper_level_row=1)
    rows = [dict(common, nist_row_index=1, model="DHF+B+QED",
                 calculated_nm=680, energy_error_percent=1.5),
            dict(common, nist_row_index=1, model="MCDHF+B+QED",
                 calculated_nm=675, energy_error_percent=.7),
            dict(common, nist_row_index=2, model="DHF+B+QED",
                 calculated_nm=700, energy_error_percent=2.0)]
    levels = [dict(z=3,charge=0,row_index=1,model="DHF+B+QED",
                   calculated_energy_cm1=15000,residual_cm1=100)]
    add_model_audit(path, rows, levels)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM line_model_audit").fetchone()[0] == 3
        assert db.execute("SELECT model FROM line_audit WHERE row_index=1").fetchone()[0] == "DHF+B+QED"
        assert db.execute("SELECT model FROM line_audit WHERE row_index=2").fetchone()[0] == "DHF+B+QED"
