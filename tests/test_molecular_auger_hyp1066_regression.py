"""HYP-1066 offline Ne KLL and H2O O KLL molecular Auger regressions."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from atomx.constants import HARTREE_EV
from atomx.providers.contract import (
    DensityMatrix,
    ElectronicStructurePayload,
    OrbitalCoefficients,
    OverlapBlock,
    Provenance,
    TwoElectronIntegrals,
)
from atomx.scattering.autoionization.molecular import (
    DiscreteMOContinuumChannel,
    SpectrumResult,
    compute_molecular_auger,
)

FIXTURE_PATH = Path(__file__).with_name("data") / (
    "molecular_auger_hyp1066_cleanroom.json"
)


class _CleanroomFixtureProvider:
    """Deterministic provider fixture exercising the public provider path."""

    def __init__(self, case: dict[str, Any]) -> None:
        self.case = case
        self.calls: list[tuple[Any, bool]] = []

    def build_core_hole_payload(
        self,
        core_hole_spec: str,
        *,
        with_eri: bool = False,
    ) -> ElectronicStructurePayload:
        self.calls.append((core_hole_spec, with_eri))
        provider = self.case["provider"]
        labels = list(provider["ao_labels"])
        energies = np.asarray(provider["orbital_energies_hartree"], dtype=float)
        occupations = np.asarray(provider["occupations"], dtype=float)
        n_mo = len(energies)
        core = int(provider["core_mo_index"])
        continuum = int(provider["continuum_mo_index"])
        eri = np.zeros((n_mo, n_mo, n_mo, n_mo), dtype=float)
        for key, raw_intensity in provider["raw_intensity_by_holes"].items():
            left, right = (int(item.removeprefix("MO")) for item in key.split(","))
            eri[core, continuum, left, right] = math.sqrt(float(raw_intensity))

        return ElectronicStructurePayload(
            Z=int(self.case["Z"]),
            provenance=Provenance(
                provider=provider["name"],
                code_version=provider["code_version"],
                method=provider["method"],
                basis=provider["basis"],
                theory_level=provider["theory_level"],
                approximations=[
                    "cleanroom-regression-fixture",
                    "nonrel-gto",
                ],
                core_hole=provider["core_hole"],
                energy_zero="delta_scf_total_energy",
            ),
            orbital_energies={f"MO{i}": float(e) for i, e in enumerate(energies)},
            orbitals=OrbitalCoefficients(
                coefficients=np.eye(n_mo),
                energies=energies,
                occupations=occupations,
                basis_labels=labels,
            ),
            overlap=OverlapBlock(np.eye(n_mo), basis_labels=labels),
            density=DensityMatrix(np.diag(occupations)),
            two_electron=TwoElectronIntegrals(eri, basis="mo"),
            metadata={
                "core_hole": {
                    "orbital_label": provider["core_hole"],
                    "occupation_before": 2.0,
                    "occupation_after": 1.0,
                    "tracked_mo_index": core,
                    "initial_state_energy_hartree": float(
                        provider["initial_state_energy_hartree"]
                    ),
                },
                "branch_by_holes": provider["branch_by_holes"],
                "mo_metadata": provider.get("mo_metadata", {}),
            },
        )


@pytest.fixture(scope="module")
def fixture() -> dict[str, Any]:
    with FIXTURE_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _run_case(case: dict[str, Any]) -> tuple[SpectrumResult, _CleanroomFixtureProvider]:
    provider = _CleanroomFixtureProvider(case)
    spectrum = compute_molecular_auger(
        provider,
        case["core_hole_spec"],
        continuum_channel=DiscreteMOContinuumChannel(
            mo_index=case["provider"]["continuum_mo_index"],
        ),
        population_threshold=0.0,
    )
    assert provider.calls == [(case["core_hole_spec"], True)]
    return spectrum, provider


def _hole_key(line) -> str:
    return ",".join(line.final_state.hole_orbitals)


def _branch_sums(spectrum: SpectrumResult, case: dict[str, Any]) -> dict[str, float]:
    branch_by_holes = case["provider"]["branch_by_holes"]
    sums: dict[str, float] = {}
    for line in spectrum.lines:
        branch = branch_by_holes[_hole_key(line)]
        sums[branch] = sums.get(branch, 0.0) + line.intensity
    return sums


def test_fixture_has_complete_cleanroom_provenance(fixture):
    provenance = fixture["provenance"]

    assert provenance["issue"] == "HYP-1066"
    assert "no NIST SRD" in provenance["data_policy"]
    assert provenance["energy_unit"] == "eV"
    assert provenance["intensity_convention"].startswith("global sum-rule")
    assert len(provenance["references"]) >= 2
    assert "sum-rule-total-stick-normalization" in provenance["approximations"]


def test_ne_kll_fixture_matches_registered_energy_and_branch_ratios(fixture):
    case = fixture["cases"]["ne_kll"]
    spectrum, _provider = _run_case(case)
    acceptance = case["acceptance"]
    branch_sums = _branch_sums(spectrum, case)
    main_branch = acceptance["main_branch"]
    main_lines = [
        line
        for line in spectrum.lines
        if case["provider"]["branch_by_holes"][_hole_key(line)] == main_branch
    ]

    assert main_lines
    for line in main_lines:
        kinetic_energy_eV = line.kinetic_energy * HARTREE_EV
        assert kinetic_energy_eV == pytest.approx(
            acceptance["main_line_kinetic_energy_eV"],
            abs=acceptance["main_line_tolerance_eV"],
        )
    for branch, expected in zip(
        acceptance["branch_order"],
        acceptance["branch_ratios"],
        strict=True,
    ):
        assert branch_sums[branch] == pytest.approx(
            expected,
            abs=acceptance["branch_tolerance_absolute"],
        )
    assert sum(spectrum.intensities) == pytest.approx(1.0)


def test_h2o_o_kll_fixture_matches_registered_main_envelope(fixture):
    case = fixture["cases"]["h2o_o_kll"]
    spectrum, _provider = _run_case(case)
    acceptance = case["acceptance"]
    branch_sums = _branch_sums(spectrum, case)
    main_branch = acceptance["main_branch"]
    lo, hi = acceptance["envelope_window_eV"]
    main_lines = [
        line
        for line in spectrum.lines
        if case["provider"]["branch_by_holes"][_hole_key(line)] == main_branch
    ]

    assert main_lines
    centroid = sum(
        line.kinetic_energy * HARTREE_EV * line.intensity for line in main_lines
    ) / branch_sums[main_branch]
    assert lo <= centroid <= hi
    assert centroid == pytest.approx(
        acceptance["main_envelope_kinetic_energy_eV"],
        abs=acceptance["main_envelope_tolerance_eV"],
    )
    assert sum(spectrum.intensities) == pytest.approx(1.0)


def test_energy_conservation_selection_rules_and_provenance(fixture):
    for case in fixture["cases"].values():
        spectrum, _provider = _run_case(case)
        assert any(
            item["reason"] == "spin-forbidden-non-soc-v1"
            for item in spectrum.dropped
        )
        for line in spectrum.lines:
            assert line.kinetic_energy == pytest.approx(
                line.initial_state.initial_energy - line.final_state.final_energy
            )
            assert line.provenance.provider == case["provider"]["name"]
            assert line.provenance.core_hole == case["provider"]["core_hole"]
            assert line.provenance.final_state
            assert line.provenance.energy_zero == "delta_scf_total_energy"
            assert "direct-minus-exchange-amplitude" in (
                line.provenance.approximations
            )
            assert "sum-rule-total-stick-normalization" in (
                line.provenance.approximations
            )


def test_ne_degenerate_sublevel_sum_is_m_independent(fixture):
    case = fixture["cases"]["ne_kll"]
    spectrum, _provider = _run_case(case)
    mo_metadata = case["provider"]["mo_metadata"]
    pp_sums: dict[int, float] = {}
    for line in spectrum.lines:
        holes = line.final_state.hole_orbitals
        if all(mo_metadata[hole]["subshell"] == "2p" for hole in holes):
            for hole in set(holes):
                m_value = int(mo_metadata[hole]["m"])
                pp_sums[m_value] = pp_sums.get(m_value, 0.0) + line.intensity

    assert set(pp_sums) == {-1, 0, 1}
    assert len({round(value, 12) for value in pp_sums.values()}) == 1


def test_spectrum_weight_sum_rule_uses_no_per_peak_scaling(fixture):
    for case in fixture["cases"].values():
        spectrum, _provider = _run_case(case)
        assert spectrum.intensity_unit == "sum_rule_normalized_probability"
        assert sum(line.intensity for line in spectrum.lines) == pytest.approx(1.0)
        scale = {
            round(line.intensity / line.raw_intensity, 12)
            for line in spectrum.lines
            if line.raw_intensity > 0.0
        }
        assert len(scale) == 1


@pytest.mark.parametrize(
    ("atom", "basis", "core_hole_spec", "continuum_mo_index"),
    [
        ("Ne 0 0 0", "sto-3g", "Ne 1s", 4),
        ("O 0 0 0; H 0 0 0.958; H 0 0.926 -0.239", "sto-3g", "O 1s", 6),
    ],
)
def test_public_path_reaches_real_pyscf_core_hole_provider_flow(
    atom,
    basis,
    core_hole_spec,
    continuum_mo_index,
):
    pyscf = pytest.importorskip("pyscf")
    from atomx.providers.pyscf_adapter import build_core_hole_delta_scf_payload

    mol = pyscf.gto.M(atom=atom, basis=basis, verbose=0)
    ground_mf = pyscf.scf.RHF(mol)
    ground_mf.conv_tol = 1e-8
    ground_mf.kernel()
    assert ground_mf.converged

    class _PySCFCoreHoleProvider:
        def __init__(self, mf) -> None:
            self.mf = mf
            self.calls: list[tuple[Any, bool]] = []

        def build_core_hole_payload(
            self,
            core_hole_spec,
            *,
            with_eri: bool = False,
        ) -> ElectronicStructurePayload:
            self.calls.append((core_hole_spec, with_eri))
            return build_core_hole_delta_scf_payload(
                self.mf,
                core_hole_spec,
                with_eri=with_eri,
                max_cycle=80,
            )

    provider = _PySCFCoreHoleProvider(ground_mf)
    spectrum = compute_molecular_auger(
        provider,
        core_hole_spec,
        continuum_channel=DiscreteMOContinuumChannel(mo_index=continuum_mo_index),
        population_threshold=0.0,
    )

    assert provider.calls == [(core_hole_spec, True)]
    assert spectrum.axis == "kinetic_energy"
    assert all(line.provenance.provider == "pyscf" for line in spectrum.lines)
