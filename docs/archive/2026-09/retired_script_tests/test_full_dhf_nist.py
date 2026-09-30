"""Mass-aware DHF jobs preserve source identity and compare like media."""

import json
import os
import sqlite3
from types import SimpleNamespace

import pytest

from atomx import Atom, Basis
from atomx.constants import PROTON_ELECTRON_MASS_RATIO
from atomx.methods.dirac_fock import DiracFock
from atomx.physics.dirac import hydrogenic_dirac_energy
from atomx.types import Exchange
from scripts.apply_dhf_damping_probe import apply as apply_damping_probe
from scripts.compare_full_dhf_nist import collect as collect_full_dhf
from scripts.generate_full_dhf_nist import (
    SCHEMA,
    RadialBasisLimitError,
    backfill_nist_level_ids,
    basis_for_job,
    ground_configuration_for_species,
    ground_configuration_with_dhf_candidates,
    has_core_hole,
    one_electron_basis_valid,
    orbital_node_valid,
    orbital_order_valid,
    pending_job_priority,
    prepare,
    process_start_ticks,
    radial_box_unresolved,
    run,
    solve_configuration,
    update_lines,
)
from atomx.reference_data.matching import (
    configuration_has_single_level,
    expand_nist_config,
    observed_vacuum_nm,
    refractive_index_air,
)
from scripts.import_nist_isotopes import parse_isotopes
from scripts.repair_dhf_ground import repair_ground
from scripts.repair_dhf_ground_selection import repair as repair_ground_selection
from scripts.supervise_first20_dhf import (
    recover_abandoned_claims,
    running_ground_count,
    stage_failed_jobs,
)


def test_pending_queue_favors_compact_ion_over_diffuse_neutral():
    diffuse = (24, 0, "1s2 2s2 2p6 3s2 3p6 3d5 35p1", 94657.0, 0)
    compact = (25, 2, "1s2 2s2 2p6 3s2 3p6 3d4", 96500.0, 0)
    ground = (24, 0, "1s2 2s2 2p6 3s2 3p6 3d5 4s1", 94657.0, 1)
    assert min((diffuse, compact), key=pending_job_priority) == compact
    assert min((diffuse, compact, ground), key=pending_job_priority) == ground
    assert min((diffuse,), key=pending_job_priority) == diffuse


def test_damping_probe_rejects_wrong_isotope_before_writing(tmp_path):
    ledger, report_path = (tmp_path / "jobs.sqlite", tmp_path / "probe.json")
    config = "1s2 2s2 2p6 3s2 3p6 3d5"
    with sqlite3.connect(ledger) as db:
        db.executescript(SCHEMA)
        db.execute("""INSERT INTO config_jobs
            (z,charge,configuration,status,isotope_a,nuclear_mass_me,is_ground)
            VALUES (23,0,?,'failed',51,92842.0,0)""", (config,))
    report = dict(z=23, charge=0, configuration=config, breit=True, qed=True,
                  converged=True, orbital_order_valid=True,
                  orbital_node_valid=True, energy_ha=-947.4,
                  normal_mass_shift_ha=0.01, residual=1e-6,
                  starting_potential="tfda", scf_damping=0.7,
                  scf_damping_release_residual=1e-6, r_max_bohr=50.0,
                  grid_hmax_bohr=1.0, nuclear_mass_me=92843.0,
                  elapsed_seconds=40.0)
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="isotope mass"):
        apply_damping_probe(report_path, ledger)
    with sqlite3.connect(ledger) as db:
        assert db.execute("SELECT status,energy_ha FROM config_jobs").fetchone() == (
            "failed", None)
    report["nuclear_mass_me"] = 92842.0
    report_path.write_text(json.dumps(report))
    assert apply_damping_probe(report_path, ledger)["energy_ha"] == -947.4
    with sqlite3.connect(ledger) as db:
        assert db.execute("""SELECT status,energy_ha,nms_ha FROM config_jobs""").fetchone() == (
            "done", -947.4, 0.01)


def test_nist_isotope_choice_uses_abundance_and_exact_proton_mass():
    source = """Atomic Number = 1
Atomic Symbol = H
Mass Number = 1
Relative Atomic Mass = 1.00782503223(9)
Isotopic Composition = 0.999885(70)

Atomic Number = 1
Atomic Symbol = D
Mass Number = 2
Relative Atomic Mass = 2.01410177812(12)
Isotopic Composition = 0.000115(70)
"""
    isotope = parse_isotopes(source)[1]
    assert isotope["a"] == 1
    assert isotope["nuclear_mass_me"] == PROTON_ELECTRON_MASS_RATIO


def test_configuration_expansion_only_adds_closed_core():
    assert expand_nist_config("3d6 4s2", 26) == (
        "1s2 2s2 2p6 3s2 3p6 3d6 4s2")
    assert expand_nist_config("2p1 3p1", 2) == "2p1 3p1"
    assert expand_nist_config("3s2", 13) is None


def test_incomplete_ion_ground_uses_nist_ground_shells_and_repairs_ledger(tmp_path):
    archive = tmp_path / "source.sqlite"
    ledger = tmp_path / "jobs.sqlite"
    old = expand_nist_config("4f13 6s2", 69)
    expected = expand_nist_config("[Xe].4f14.5d", 69)
    assert old is not None and expected is not None and old != expected
    with sqlite3.connect(archive) as db:
        db.executescript("""CREATE TABLE species(z INTEGER,charge INTEGER);
            CREATE TABLE levels(z INTEGER,charge INTEGER,row_index INTEGER,
                configuration TEXT,energy_cm1 REAL);
            CREATE TABLE nist_ionization(z INTEGER,charge INTEGER,
                ground_shells TEXT);
            INSERT INTO species VALUES (76,7);
            INSERT INTO levels VALUES (76,7,0,'5d1',0);
            INSERT INTO nist_ionization VALUES (76,7,'[Xe].4f14.5d');""")
        assert ground_configuration_for_species(
            db, 76, 7, has_ionization=True) == (
                expected, "nist_ground_shells")
    with sqlite3.connect(ledger) as db:
        db.executescript(SCHEMA)
        db.execute("""INSERT INTO config_jobs
            (z,charge,configuration,status,energy_ha,isotope_a,
             nuclear_mass_me,is_ground)
            VALUES (76,7,?,'done',-17218.8,192,350000,1)""", (old,))
    assert repair_ground_selection(archive, ledger, dry_run=True)["repaired_count"] == 1
    with sqlite3.connect(ledger) as db:
        assert db.execute("SELECT COUNT(*) FROM config_jobs").fetchone()[0] == 1
    assert repair_ground_selection(archive, ledger)["repaired_count"] == 1
    with sqlite3.connect(ledger) as db:
        assert db.execute("""SELECT configuration,status,is_ground FROM config_jobs
            ORDER BY is_ground DESC""").fetchall() == [
                (expected, "pending", 1), (old, "done", 0)]
    assert repair_ground_selection(archive, ledger)["repaired_count"] == 0


def test_missing_superheavy_ground_uses_ionic_isoelectronic_reference(tmp_path):
    archive = tmp_path / "source.sqlite"
    ledger = tmp_path / "jobs.sqlite"
    old = expand_nist_config("3d8 4s2", 28)
    expected = expand_nist_config("[Ar].3d10", 28)
    assert old is not None and expected is not None and old != expected
    with sqlite3.connect(archive) as db:
        db.executescript("""CREATE TABLE species(z INTEGER,charge INTEGER);
            CREATE TABLE levels(z INTEGER,charge INTEGER,row_index INTEGER,
                configuration TEXT,energy_cm1 REAL);
            CREATE TABLE nist_ionization(z INTEGER,charge INTEGER,
                ground_shells TEXT);
            INSERT INTO species VALUES (114,86);
            INSERT INTO nist_ionization VALUES (110,82,'[Ar].3d10');""")
        assert ground_configuration_for_species(
            db, 114, 86, has_ionization=True) == (
                expected, "nist_isoelectronic_z110")
    with sqlite3.connect(ledger) as db:
        db.executescript(SCHEMA)
        db.execute("""INSERT INTO config_jobs
            (z,charge,configuration,status,energy_ha,isotope_a,
             nuclear_mass_me,is_ground)
            VALUES (114,86,?,'done',-41044.34,289,529000,1)""", (old,))
    assert repair_ground_selection(archive, ledger)["repaired_count"] == 1
    with sqlite3.connect(ledger) as db:
        assert db.execute("""SELECT configuration,status,is_ground
            FROM config_jobs ORDER BY is_ground DESC""").fetchall() == [
                (expected, "pending", 1), (old, "done", 0)]
        db.execute("""UPDATE config_jobs SET status='done',energy_ha=-41044.0
            WHERE z=114 AND charge=86 AND configuration=?""", (expected,))
    # The ionic analogue is a provisional candidate: if both DHF branches
    # converge and the old configuration is lower, select the old branch.
    selected = repair_ground_selection(archive, ledger)
    assert selected["repaired_count"] == 1
    assert selected["repaired"][0]["source"] == (
        "dhf_lower_than_nist_isoelectronic_z110")
    with sqlite3.connect(ledger) as db:
        assert db.execute("""SELECT configuration FROM config_jobs
            WHERE z=114 AND charge=86 AND is_ground=1""").fetchone() == (old,)
    assert repair_ground_selection(archive, ledger, dry_run=True)["repaired_count"] == 0


def test_second_nearest_ionic_candidate_can_win_after_dhf_convergence():
    with sqlite3.connect(":memory:") as source:
        source.executescript("""CREATE TABLE levels(z INTEGER,charge INTEGER,
            row_index INTEGER,configuration TEXT,energy_cm1 REAL);
            CREATE TABLE nist_ionization(z INTEGER,charge INTEGER,
                ground_shells TEXT);
            INSERT INTO nist_ionization VALUES
                (103,14,'[Hg].5f8.6p'),
                (102,13,'[Hg].5f7.6p2');""")
        preferred = expand_nist_config("[Hg].5f8.6p", 89)
        nearby = expand_nist_config("[Hg].5f7.6p2", 89)
        neutral = expand_nist_config("6d1 7s2", 89)
        assert preferred and nearby and neutral
        candidates = {preferred: ("done", -100.0),
                      nearby: ("pending", None),
                      neutral: ("done", -90.0)}
        assert ground_configuration_with_dhf_candidates(
            source, 104, 15, has_ionization=True,
            candidates=candidates) == (preferred, "nist_isoelectronic_z103")
        candidates[nearby] = ("done", -101.0)
        assert ground_configuration_with_dhf_candidates(
            source, 104, 15, has_ionization=True,
            candidates=candidates) == (
                nearby, "dhf_lower_nist_isoelectronic_z102_over_z103")


def test_fresh_ledger_prepares_ion_ground_from_complete_gsie_shells(tmp_path):
    archive = tmp_path / "source.sqlite"
    ledger = tmp_path / "jobs.sqlite"
    isotopes = tmp_path / "isotopes.json"
    expected = expand_nist_config("[Xe].4f14.5d", 69)
    with sqlite3.connect(archive) as db:
        db.executescript("""CREATE TABLE species(z INTEGER,charge INTEGER);
            CREATE TABLE levels(z INTEGER,charge INTEGER,row_index INTEGER,
                configuration TEXT,term TEXT,j_text TEXT,energy_cm1 REAL);
            CREATE TABLE lines(z INTEGER,charge INTEGER,row_index INTEGER,
                lower_configuration TEXT,upper_configuration TEXT,
                observed_nm REAL,medium TEXT);
            CREATE TABLE nist_ionization(z INTEGER,charge INTEGER,
                ground_shells TEXT);
            INSERT INTO species VALUES (76,7);
            INSERT INTO levels VALUES (76,7,0,'5d1','2D','3/2',0);
            INSERT INTO nist_ionization VALUES (76,7,'[Xe].4f14.5d');""")
    isotopes.write_text(json.dumps({"source_sha256": "test", "selected": {
        "76": {"a": 192, "nuclear_mass_me": 350000.0}}}))
    result = prepare(archive, ledger, isotopes, ground_max_z=76)
    assert result["ground_selection_sources"] == {"nist_ground_shells": 1}
    with sqlite3.connect(ledger) as db:
        assert db.execute("""SELECT configuration,is_ground,status
            FROM config_jobs""").fetchone() == (expected, 1, "pending")


def test_core_hole_start_selection_detects_vacant_inner_shell():
    assert has_core_hole("1s1 3s1")
    assert has_core_hole("1s2 2s1 2p1")
    assert not has_core_hole("1s2 2s2 2p1")


def test_air_conversion_matches_nist_boundary():
    air = 200.0648 / refractive_index_air(200.0648)
    assert air == pytest.approx(200.0, abs=0.001)
    assert observed_vacuum_nm(air, "air") == pytest.approx(200.0648, abs=1e-6)


def test_completed_line_uses_corrected_level_gap(tmp_path):
    db = sqlite3.connect(tmp_path / "jobs.sqlite")
    db.executescript(SCHEMA)
    db.executemany("""INSERT INTO config_jobs
        (z,charge,configuration,status,energy_ha,isotope_a,nuclear_mass_me)
        VALUES (1,0,?,'done',?,1,?)""",
        [("1s1", -0.499729771705357, PROTON_ELECTRON_MASS_RATIO),
         ("2p1", -0.12493264921022461, PROTON_ELECTRON_MASS_RATIO)])
    db.executemany("""INSERT INTO config_state_counts VALUES (1,0,?,1)""",
                   [("1s1",), ("2p1",)])
    db.execute("""INSERT INTO line_jobs
        (z,charge,row_index,lower_configuration,upper_configuration,
         observed_nm,medium,status,isotope_a,mass_model)
        VALUES (1,0,74,'1s1','2p1',121.56699,'vacuum',
                'waiting_for_levels',1,'one-body normal mass shift')""")
    update_lines(db, 1, 0)
    status, predicted, error = db.execute("""SELECT status,calculated_nm,
        energy_error_percent FROM line_jobs""").fetchone()
    assert status == "configuration_average_only"
    assert predicted == pytest.approx(121.568042534)
    assert error == pytest.approx(-0.0008657983)
    db.close()


def test_single_level_gate_uses_physics_not_nist_row_count():
    assert configuration_has_single_level("1s2 2s2 2p6")
    assert configuration_has_single_level("1s2 2s2 2p6 3s1")
    assert not configuration_has_single_level("1s2 2s2 2p6 3s1 44p1")
    assert not configuration_has_single_level("2p1")
    assert not configuration_has_single_level("1s1 4s1")


def test_collector_reclassifies_old_worker_multiplet_score(tmp_path):
    archive = tmp_path / "archive.sqlite"
    ledger = tmp_path / "ledger.sqlite"
    with sqlite3.connect(archive) as db:
        db.executescript("""CREATE TABLE species(z INTEGER,charge INTEGER,
            symbol TEXT,label TEXT);
            CREATE TABLE lines(z INTEGER,charge INTEGER,row_index INTEGER,
            lower_configuration TEXT,upper_configuration TEXT,
            lower_term TEXT,lower_j TEXT,upper_term TEXT,upper_j TEXT,
            observed_nm REAL,observed_unc_nm REAL,ritz_nm REAL,medium TEXT);
            INSERT INTO species VALUES (12,0,'Mg','Mg I');
            INSERT INTO lines VALUES (12,0,91,'3s2','3s1 44p1',
                '1S','0','1P*','1',162.3069,NULL,NULL,'vacuum');""")
    with sqlite3.connect(ledger) as db:
        db.executescript(SCHEMA)
        db.execute("""INSERT INTO line_jobs
            (z,charge,row_index,lower_configuration,upper_configuration,
             observed_nm,medium,status,calculated_nm,isotope_a,mass_model)
            VALUES (12,0,91,'1s2 2s2 2p6 3s2',
                '1s2 2s2 2p6 3s1 44p1',162.3069,'vacuum',
                'compared',162.5,24,'one-body normal mass shift')""")
    assert collect_full_dhf(archive, ledger) == []
    rows = collect_full_dhf(archive, ledger, include_averages=True)
    assert len(rows) == 1
    assert rows[0]["comparison_status"] == "configuration_average_only"


def test_multiplet_line_keeps_prediction_but_is_not_scored(tmp_path):
    db = sqlite3.connect(tmp_path / "jobs.sqlite")
    db.executescript(SCHEMA)
    db.executemany("""INSERT INTO config_jobs
        (z,charge,configuration,status,energy_ha,isotope_a,nuclear_mass_me)
        VALUES (2,0,?,'done',?,4,7294)""",
        [("1s1 2s1", -2.1728), ("1s1 2p1", -2.1283)])
    db.executemany("INSERT INTO config_state_counts VALUES (2,0,?,?)",
                   [("1s1 2s1", 2), ("1s1 2p1", 4)])
    db.execute("""INSERT INTO line_jobs
        (z,charge,row_index,lower_configuration,upper_configuration,
         observed_nm,medium,status,isotope_a,mass_model)
        VALUES (2,0,570,'1s1 2s1','1s1 2p1',2058.692,'air',
                'waiting_for_levels',4,'one-body normal mass shift')""")
    update_lines(db, 2, 0)
    status, predicted = db.execute(
        "SELECT status,calculated_nm FROM line_jobs").fetchone()
    assert status == "configuration_average_only"
    assert predicted is not None


def test_core_hole_solver_preserves_occupied_radial_order():
    result = Atom(Z=4, config="1s1 2s2 2p1").solve(DiracFock())
    assert result.converged
    assert orbital_order_valid(result)
    assert result.orbital_energies["1s1/2"] < result.orbital_energies["2s1/2"]
    assert result.total_energy == pytest.approx(-10.3805, abs=0.02)


def test_sc_iv_core_p_node_guard_rejects_wrong_converged_branch():
    config = "1s2 2s2 2p6 3s2 3p5 4p1"
    mass = 81928.6
    bad = Atom(Z=21, config=config, engine="bspline", basis=Basis()).solve(
        DiracFock(core_hole_stationary=False, breit=True, qed=True, nuclear_mass_me=mass,
                  starting_potential="gsz"), max_iter=100)
    good = Atom(Z=21, config=config, engine="bspline", basis=Basis()).solve(
        DiracFock(core_hole_stationary=False, breit=True, qed=True, nuclear_mass_me=mass,
                  starting_potential="tfda"), max_iter=100)
    assert bad.converged and good.converged
    assert orbital_order_valid(bad) and orbital_order_valid(good)
    assert not orbital_node_valid(bad, 21, config)
    assert orbital_node_valid(good, 21, config)
    assert good.total_energy < bad.total_energy - 0.5


def test_kr_xxvii_core_p_node_guard_rejects_wrong_converged_branch():
    config = "1s2 2s2 2p5 3p1"
    mass = 157000.0
    bad = Atom(Z=36, config=config, engine="bspline", basis=Basis()).solve(
        DiracFock(core_hole_stationary=False, breit=True, qed=True, nuclear_mass_me=mass,
                  starting_potential="gsz"), max_iter=100)
    good = Atom(Z=36, config=config, engine="bspline", basis=Basis()).solve(
        DiracFock(core_hole_stationary=False, breit=True, qed=True, nuclear_mass_me=mass,
                  starting_potential="tfda"), max_iter=100)
    assert bad.converged and good.converged
    assert orbital_order_valid(bad) and orbital_order_valid(good)
    assert not orbital_node_valid(bad, 36, config)
    assert orbital_node_valid(good, 36, config)
    assert good.total_energy < bad.total_energy - 0.4


@pytest.mark.parametrize("z,mass,min_drop", [
    (54, 240392.56559839583, 0.4),
    (74, 335248.0340150634, 0.7),
])
def test_heavy_core_p_node_guard_rejects_reproduced_gsz_branch(
        z, mass, min_drop):
    config = "1s2 2s2 2p5 3p1"
    bad = Atom(Z=z, config=config, engine="bspline", basis=Basis()).solve(
        DiracFock(core_hole_stationary=False, breit=True, qed=True, nuclear_mass_me=mass,
                  starting_potential="gsz"), max_iter=160)
    good = Atom(Z=z, config=config, engine="bspline", basis=Basis()).solve(
        DiracFock(core_hole_stationary=False, breit=True, qed=True, nuclear_mass_me=mass,
                  starting_potential="tfda"), max_iter=160)
    assert bad.converged and good.converged
    assert orbital_order_valid(bad) and orbital_order_valid(good)
    assert not orbital_node_valid(bad, z, config)
    assert orbital_node_valid(good, z, config)
    assert good.total_energy < bad.total_energy - min_drop


def test_sc_x_core_p_node_guard_accepts_higher_energy_named_branch():
    # Reproduce the historical frozen-rotation branch for this guard regression.
    config = "1s2 2s2 2p6 3p1 4p1"
    mass = 81928.60759399581
    bad = Atom(Z=21, config=config, engine="bspline", basis=Basis()).solve(
        DiracFock(core_hole_stationary=False, breit=True, qed=True, nuclear_mass_me=mass,
                  starting_potential="gsz"), max_iter=100)
    good = Atom(Z=21, config=config, engine="bspline", basis=Basis()).solve(
        DiracFock(core_hole_stationary=False, breit=True, qed=True, nuclear_mass_me=mass,
                  starting_potential="tfda"), max_iter=100)
    assert bad.converged and good.converged
    assert orbital_order_valid(bad) and orbital_order_valid(good)
    assert not orbital_node_valid(bad, 21, config)
    assert orbital_node_valid(good, 21, config)
    assert good.total_energy > bad.total_energy + 0.04


def test_equal_occupation_core_hole_does_not_swap_radial_labels():
    result = Atom(Z=3, config="1s1 3s1").solve(DiracFock())
    assert result.converged and orbital_order_valid(result)
    assert result.orbital_energies["1s1/2"] < result.orbital_energies["3s1/2"]


def test_ground_state_rotation_still_optimizes_lithium():
    result = Atom(Z=3, config="1s2 2s1").solve(DiracFock())
    assert result.converged and orbital_order_valid(result)
    assert result.total_energy == pytest.approx(-7.433533, abs=0.002)


def test_explicit_scf_damping_preserves_lithium_ground_energy():
    plain = Atom(Z=3, config="1s2 2s1").solve(DiracFock())
    damped = Atom(Z=3, config="1s2 2s1").solve(
        DiracFock(scf_damping=0.35))
    assert damped.converged and orbital_order_valid(damped)
    assert damped.total_energy == pytest.approx(plain.total_energy, abs=1e-5)


def test_heavy_repair_uses_damping_if_seed_unavailable(monkeypatch):
    import scripts.generate_full_dhf_nist as jobs

    attempts = []

    class StubAtom:
        def __init__(self, **_kwargs):
            pass

        def solve(self, method, max_iter):
            attempts.append((method.starting_potential,
                             method.scf_damping, max_iter))
            return SimpleNamespace(converged=True, total_energy=-1.0,
                                   subshells=[], orbital_energies={})

    monkeypatch.setattr(jobs, "Atom", StubAtom)
    _result, source = solve_configuration(29, "1s2 2s2 2p6 3s2 3p6 3d10 4s1",
                                          115000.0, Basis(), retry_only=True)
    assert attempts == [("tfda", 0.35, 120), (None, 0.35, 200)]
    assert source == "tfda+damping=0.35"


def test_heavy_repair_uses_slater_orbitals_only_as_hf_seed(monkeypatch):
    import scripts.generate_full_dhf_nist as jobs

    attempts = []

    class StubAtom:
        def __init__(self, **_kwargs):
            pass

        def solve(self, method, max_iter):
            attempts.append((method.exchange, method.breit, method.qed,
                             method.nuclear_mass_me, method.seed_orbitals,
                             max_iter))
            return SimpleNamespace(converged=True, total_energy=-1.0,
                                   subshells=[], orbital_energies={},
                                   orbitals_raw={})

    monkeypatch.setattr(jobs, "Atom", StubAtom)
    _result, source = solve_configuration(103, "1s2 7s2 7p1",
                                          475000.0, Basis(), retry_only=True)
    assert [row[0] for row in attempts] == [Exchange.SLATER,
                                             Exchange.HARTREE_FOCK]
    assert all(row[1:4] == (True, True, 475000.0) for row in attempts)
    assert attempts[1][4] == {}
    assert source == "slater_seed+dhf_damping=0.35"


def test_light_rydberg_retry_tries_validated_damped_branch(monkeypatch):
    import scripts.generate_full_dhf_nist as jobs

    attempts = []

    class StubAtom:
        def __init__(self, **_kwargs):
            pass

        def solve(self, method, max_iter):
            attempts.append((method.starting_potential,
                             method.scf_damping, max_iter))
            return SimpleNamespace(converged=True, total_energy=-1.0,
                                   subshells=[], orbital_energies={})

    monkeypatch.setattr(jobs, "Atom", StubAtom)
    _result, source = solve_configuration(
        5, "1s2 2s2 39d1", 20063.73598271508, Basis(), retry_only=True)
    assert attempts == [(None, 0.35, 200)]
    assert source == "hydrogenic+damping=0.35"


def test_heavy_repair_reaches_stronger_damping_after_failed_start(monkeypatch):
    import scripts.generate_full_dhf_nist as jobs

    attempts = []

    class StubAtom:
        def __init__(self, **_kwargs):
            pass

        def solve(self, method, max_iter):
            attempts.append((method.scf_damping, max_iter))
            if method.scf_damping == 0.35:
                raise RuntimeError("SCF oscillation")
            return SimpleNamespace(converged=True, total_energy=-1.0,
                                   subshells=[], orbital_energies={})

    monkeypatch.setattr(jobs, "Atom", StubAtom)
    _result, source = solve_configuration(46,
        "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10",
        193004.0, Basis(), retry_only=True)
    assert attempts == [(0.35, 120), (0.35, 200), (0.7, 300)]
    assert source == "tfda+damping=0.7"


def test_calculation_scope_stops_at_requested_atomic_number(tmp_path):
    path = tmp_path / "jobs.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA)
        db.executemany("""INSERT INTO config_jobs
            (z,charge,configuration,status,isotope_a,nuclear_mass_me)
            VALUES (?,?,'1s1','pending',1,1836.152673426)""",
            [(1, 0), (21, 20)])
        db.execute("INSERT INTO metadata VALUES ('active_max_z','118')")
    run(path, max_calculations=1, refresh_site_every=0, max_z=20)
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT status FROM config_jobs WHERE z=1").fetchone()[0] == "done"
        assert db.execute("SELECT status FROM config_jobs WHERE z=21").fetchone()[0] == "pending"
        assert db.execute("SELECT value FROM metadata WHERE key='active_max_z'").fetchone()[0] == "118"


def test_calculation_scope_selects_requested_ion_charge(tmp_path):
    path = tmp_path / "jobs.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA)
        db.executemany("""INSERT INTO config_jobs
            (z,charge,configuration,status,isotope_a,nuclear_mass_me)
            VALUES (2,?,?,'pending',4,7294.3)""",
            [(0, "1s2"), (1, "1s1")])
    run(path, max_calculations=1, refresh_site_every=0,
        min_z=2, max_z=2, min_charge=1, max_charge=1)
    with sqlite3.connect(path) as db:
        assert db.execute("""SELECT charge,status FROM config_jobs
            ORDER BY charge""").fetchall() == [(0, "pending"), (1, "done")]


def test_repair_queue_retries_a_failed_configuration_once(tmp_path):
    path = tmp_path / "jobs.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA)
        db.execute("""INSERT INTO config_jobs
            (z,charge,configuration,status,isotope_a,nuclear_mass_me)
            VALUES (1,0,'1s1','failed',1,1836.152673426)""")
    assert stage_failed_jobs(path) == 1
    assert stage_failed_jobs(path) == 0
    run(path, max_calculations=1, refresh_site_every=0, max_z=1,
        recover_running=False, retry_only=True, queue_status="repair_pending")
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT status FROM config_jobs").fetchone()[0] == "done"
        db.execute("UPDATE config_jobs SET status='failed'")
    assert stage_failed_jobs(path) == 0


def test_heavy_repair_stages_only_new_failed_ground_jobs(tmp_path):
    path = tmp_path / "jobs.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA)
        db.executemany("""INSERT INTO config_jobs
            (z,charge,configuration,status,isotope_a,nuclear_mass_me,is_ground)
            VALUES (?,?,?,'failed',1,1836.152673426,?)""", [
                (2, 0, '1s2', 1), (47, 0, '1s2 2s1', 1),
                (47, 1, '1s2', 0)])
    assert stage_failed_jobs(path, min_z=21, max_z=118,
                             ground_only=True) == 1
    assert stage_failed_jobs(path, min_z=21, max_z=118,
                             ground_only=True) == 0
    with sqlite3.connect(path) as db:
        assert db.execute("""SELECT z,charge FROM config_jobs
            WHERE status='repair_pending'""").fetchall() == [(47, 0)]
        db.execute("""UPDATE config_jobs SET status='running'
            WHERE z=47 AND charge=0""")
        db.execute("""UPDATE config_jobs SET status='running'
            WHERE z=2 AND charge=0""")
    assert running_ground_count(path, min_z=1, max_z=118) == 1


def test_missing_line_endpoint_uses_same_ion_nist_level_id(tmp_path):
    archive, fresh, legacy, isotopes = (tmp_path / name for name in
        ("archive.sqlite", "fresh.sqlite", "legacy.sqlite", "isotopes.json"))
    with sqlite3.connect(archive) as db:
        db.executescript("""CREATE TABLE species(z INTEGER,charge INTEGER);
            CREATE TABLE levels(z INTEGER,charge INTEGER,configuration TEXT,
                term TEXT,j_text TEXT,energy_cm1 REAL,level_id TEXT,row_index INTEGER);
            CREATE TABLE lines(z INTEGER,charge INTEGER,row_index INTEGER,
                lower_configuration TEXT,upper_configuration TEXT,
                lower_level_id TEXT,upper_level_id TEXT,
                observed_nm REAL,medium TEXT);
            INSERT INTO species VALUES (3,0);
            INSERT INTO levels VALUES
                (3,0,'1s2 2s1','2S','1/2',0,'li-ground',0),
                (3,0,'1s2 3s1','2S','1/2',27000,'li-upper',1),
                (4,0,'1s2 3p2','3P','0',1000,'be-upper',0);
            INSERT INTO lines VALUES
                (3,0,7,'1s2 2s1','', 'li-ground','li-upper',370.0,'vacuum'),
                (3,0,8,'1s2 2s1','', 'li-ground','be-upper',371.0,'vacuum');""")
    isotopes.write_text(json.dumps({"source_sha256": "test", "selected": {
        "3": {"a": 7, "nuclear_mass_me": 12786.0}}}))
    prepare(archive, fresh, isotopes, ground_max_z=3)
    with sqlite3.connect(fresh) as db:
        assert db.execute("""SELECT upper_configuration,status,
            classification_source FROM line_jobs WHERE row_index=7""").fetchone() == (
                "1s2 3s1", "waiting_for_levels", "nist_level_id")
        assert db.execute("""SELECT status FROM line_jobs
            WHERE row_index=8""").fetchone()[0] == "missing_classification"
    with sqlite3.connect(legacy) as db:
        db.executescript(SCHEMA)
        db.execute("""INSERT INTO line_jobs(z,charge,row_index,
            observed_nm,medium,status,isotope_a,mass_model)
            VALUES (3,0,7,370.0,'vacuum','missing_classification',7,
                    'one-body normal mass shift')""")
    result = backfill_nist_level_ids(archive, legacy, isotopes)
    assert result["recovered_lines"] == 1
    assert result["new_configurations"] == 2
    with sqlite3.connect(legacy) as db:
        assert db.execute("""SELECT status,classification_source
            FROM line_jobs""").fetchone() == ("waiting_for_levels", "nist_level_id")
    assert backfill_nist_level_ids(archive, legacy, isotopes)["recovered_lines"] == 0


def test_failed_ground_repair_records_mass_and_start(tmp_path):
    path = tmp_path / "jobs.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA)
        db.execute("""INSERT INTO config_jobs
            (z,charge,configuration,status,isotope_a,nuclear_mass_me,is_ground)
            VALUES (1,0,'1s1','failed',1,1836.152673426,1)""")
    repaired = repair_ground(path, 1, 0, 0.7, max_iter=40)
    assert repaired["starting_potential"] == "hydrogenic+damping=0.7"
    with sqlite3.connect(path) as db:
        status, energy, nms, start = db.execute("""SELECT status,energy_ha,
            nms_ha,starting_potential FROM config_jobs""").fetchone()
    assert status == "done"
    assert energy == pytest.approx(-0.4997, abs=0.001)
    assert nms > 0
    assert start == "hydrogenic+damping=0.7"


def test_abandoned_worker_claim_returns_to_original_queue(tmp_path):
    path = tmp_path / "jobs.sqlite"
    live_ticks = process_start_ticks(os.getpid())
    assert live_ticks is not None
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA)
        db.executemany("""INSERT INTO config_jobs
            (z,charge,configuration,status,isotope_a,nuclear_mass_me,
             worker_pid,worker_start_ticks,claimed_from_status)
            VALUES (?,0,?,'running',1,1836,?,?,?)""", [
                (1, '1s1', 99999999, 1, 'retry_pending'),
                (2, '1s2', os.getpid(), live_ticks, 'pending')])
    assert recover_abandoned_claims(path, max_z=2) == 1
    with sqlite3.connect(path) as db:
        assert db.execute("""SELECT z,status,worker_pid FROM config_jobs
            ORDER BY z""").fetchall() == [
                (1, 'retry_pending', None), (2, 'running', os.getpid())]


def test_ground_jobs_cover_ions_without_observed_lines_in_scope(tmp_path):
    archive, output, isotopes = (tmp_path / name for name in
                                  ("archive.sqlite", "jobs.sqlite", "isotopes.json"))
    with sqlite3.connect(archive) as db:
        db.executescript("""CREATE TABLE species(z INTEGER,charge INTEGER);
            CREATE TABLE levels(z INTEGER,charge INTEGER,row_index INTEGER,
                configuration TEXT,term TEXT,j_text TEXT,energy_cm1 REAL);
            CREATE TABLE lines(z INTEGER,charge INTEGER,row_index INTEGER,
                lower_configuration TEXT,upper_configuration TEXT,
                observed_nm REAL,medium TEXT);
            INSERT INTO species VALUES (1,0),(2,1),(21,20);
            INSERT INTO levels VALUES
              (1,0,0,'1s1','2S','1/2',0),
              (2,1,0,'1s1','2S','1/2',0),
              (21,20,0,'1s1','2S','1/2',0);
        """)
    isotopes.write_text(json.dumps({"source_sha256": "test", "selected": {
        str(z): {"a": z, "nuclear_mass_me": 2000*z} for z in (1,2,21)}}))
    summary = prepare(archive, output, isotopes, ground_max_z=2)
    assert summary["ground_jobs_added"] == 2
    with sqlite3.connect(output) as db:
        assert list(db.execute("SELECT z,charge,configuration FROM config_jobs")) == [
            (1, 0, "1s1"), (2, 1, "1s1")]


def test_unresolved_one_electron_manifold_is_not_scored(tmp_path):
    db = sqlite3.connect(tmp_path / "jobs.sqlite")
    db.executescript(SCHEMA)
    db.execute("""INSERT INTO line_jobs
        (z,charge,row_index,lower_configuration,upper_configuration,
         observed_nm,medium,status,isotope_a,mass_model)
        VALUES (1,0,557,'6p1','6s1',7591000000,'vacuum',
                'compared',1,'one-body normal mass shift')""")
    update_lines(db, 1, 0)
    assert db.execute("SELECT status FROM line_jobs").fetchone()[0] == (
        "unresolved_hydrogenic_n_manifold")
    db.close()


def test_hydrogen_high_n_failed_basis_check():
    mass = PROTON_ELECTRON_MASS_RATIO
    assert one_electron_basis_valid(1, "2p1", -0.1249326492, mass)
    assert not one_electron_basis_valid(1, "4p1", -0.0312067033, mass)
    assert not one_electron_basis_valid(1, "6p1", -0.0040584122, mass)
    calcium_mass = 72947.0
    reference = hydrogenic_dirac_energy(1, -1, 20)
    reference *= calcium_mass / (calcium_mass + 1)
    assert one_electron_basis_valid(20, "1s1", reference, calcium_mass)
    assert not one_electron_basis_valid(20, "1s1", reference * 1.01,
                                        calcium_mass)


def test_radial_box_gate_flags_diffuse_neutral_rydberg_levels():
    assert radial_box_unresolved(0, "1s2 2s2 2p6 3s1 7i1")
    assert not radial_box_unresolved(4, "1s2 2s2 4f1")


def test_diffuse_unique_line_keeps_prediction_outside_accuracy_score(tmp_path):
    db = sqlite3.connect(tmp_path / "jobs.sqlite")
    db.executescript(SCHEMA)
    lower = "6s1"
    upper = "7s1"
    db.executemany("""INSERT INTO config_jobs
        (z,charge,configuration,status,energy_ha,isotope_a,nuclear_mass_me)
        VALUES (1,0,?,'done',?,1,1836)""",
        [(lower, -0.014), (upper, -0.010)])
    db.executemany("INSERT INTO config_state_counts VALUES (1,0,?,1)",
                   [(lower,), (upper,)])
    db.execute("""INSERT INTO line_jobs
        (z,charge,row_index,lower_configuration,upper_configuration,
         observed_nm,medium,status,isotope_a,mass_model)
        VALUES (1,0,1,?,?,12321.72,'vacuum',
                'waiting_for_levels',1,'one-body normal mass shift')""",
               (lower, upper))
    update_lines(db, 1, 0)
    status, predicted = db.execute(
        "SELECT status,calculated_nm FROM line_jobs").fetchone()
    assert status == "radial_box_unresolved"
    assert predicted > 0
    db.close()


def test_dirac_fock_mass_option_applies_level_correction():
    baseline = Atom(Z=1, config="1s1").solve(DiracFock())
    corrected = Atom(Z=1, config="1s1").solve(
        DiracFock(nuclear_mass_me=PROTON_ELECTRON_MASS_RATIO))
    assert baseline.converged and corrected.converged
    assert corrected.energy_components["nms"] > 0
    assert corrected.total_energy - baseline.total_energy == pytest.approx(
        corrected.energy_components["nms"], abs=1e-8)


def test_explicit_dirac_fock_method_uses_requested_radial_box():
    reference = hydrogenic_dirac_energy(6, -1, 1)
    small = Atom(Z=1, config="6s1", engine="bspline").solve(DiracFock())
    expanded = Atom(Z=1, config="6s1", engine="bspline",
                    basis=Basis(r_max=200)).solve(DiracFock())
    assert abs(expanded.total_energy - reference) < 1e-8
    assert abs(small.total_energy - reference) > 1e-3


def test_production_basis_contains_moderate_rydberg_state_and_caps_extreme_box():
    basis, radius, hmax = basis_for_job(
        12, 0, "1s2 2s2 2p6 3s1 7i1")
    assert basis.r_max == radius == 245
    assert basis.grid_hmax == hmax == 3.5
    _, hydrogen_radius, _ = basis_for_job(1, 0, "4s1")
    _, lithium_radius, _ = basis_for_job(3, 2, "8p1")
    assert hydrogen_radius == 80
    assert lithium_radius > 50
    magnesium_basis, magnesium_radius, magnesium_hmax = basis_for_job(
        12, 0, "1s2 2s2 2p6 3s1 59p1")
    assert magnesium_basis.r_max == magnesium_radius == 5 * 59**2
    assert magnesium_basis.grid_hmax == magnesium_hmax == 59 / 2
    with pytest.raises(RadialBasisLimitError):
        basis_for_job(37, 0, "77p1")
