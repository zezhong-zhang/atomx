"""Derive case prose from the owning document and attach declared setup units.

The registry is a research design, not an executable universal Atomx input.
No version, data file, execution, or acceptance result is guessed.
"""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
text = (ROOT / "研究体系与案例设计.md").read_text(encoding="utf-8")
FIELDS = {
    "科学动机": "motivation", "体系与起始输入": "system_and_starting_input",
    "模型与软件": "model_and_software", "计算步骤": "procedure",
    "观测量与验证": "observables_and_validation", "Atomx 的位置与故事承接": "atomx_role_and_next_case",
}
PARAMS = {
    "S01": {"Z": [10, 18, 36, 54], "electron_counts": [1, 2, 3], "charge_rule": "Z-N", "light_ion_proposed_numerical_energy_stability_meV": 1, "isotope_mass_and_radius": None},
    "S02": {"Z": 10, "charge": 7, "log10_T_K_exploration": [5.4, 6.1], "log10_ne_cm_minus3_exploration": [7, 10], "reference_wavelength_angstrom_from_CHIANTI": 770.428, "proposed_synthetic_velocity_recovery_error_km_per_s": 5, "observational_data_id": None},
    "S03": {"Z": 10, "charges": [8, 9], "log10_T_K_exploration": [6, 7.5], "log10_ne_cm_minus3_exploration": [8, 13], "UV_stellar_spectrum_and_geometry": None, "proposed_refinement_line_ratio_change_max": 0.02, "observational_data_id": None},
    "S04": {"elements": ["Ne", "W"], "Te_eV_exploration": [5, 5000], "ne_cm_minus3_exploration": [1e12, 1e15], "proposed_population_sum_error_max": 1e-8, "proposed_refinement_output_change_max": 0.05, "device_discharge_and_transport_model": None, "atomic_collision_rate_sources": None},
    "S05": {"systems": ["CO", "CO2"], "absorber_shells": ["C 1s", "O 1s"], "CO_demo_bond_angstrom": 1.128, "bond_sensitivity_angstrom": [-0.02, 0.02], "proposed_basis_peak_drift_eV_max": 0.1, "proposed_relative_intensity_change_max": 0.05, "core_response_backend_version": None, "reference_spectra": None},
    "S06": {"system": "MgO", "demo_nearest_neighbor_distance_angstrom": 2.106, "demo_formal_point_charges_e": {"Mg": 2, "O": -2}, "demo_quantum_shells": 0, "demo_mm_shells": 1, "demo_evjen": True, "demo_basis": "sto-3g", "proposed_fixed_QM_MM_refinement_core_ionization_drift_eV_max": 0.1, "production_core_basis": None},
    "S07": {"system": "NiO", "formal_d_electrons": 8, "configurations": ["d8", "d9_Lhole", "d10_Lhole2"], "edrixs_tutorial_Udd_eV": 7.3, "edrixs_tutorial_Udp_eV": 8.5, "edrixs_tutorial_Delta_eV": 4.7, "hybridization_scale_control": [0, 0.5, 1], "parameter_source": "https://nsls-ii.github.io/edrixs/auto_examples/example_3_AIM_XAS.html", "proposed_matched_model_cross_code_peak_difference_eV_max": 0.05, "proposed_normalized_integral_difference_max": 0.02, "Atomx_CF_convention": "radial-free coefficients expressed in cm^-1; radial moment applied in operator, per CRYSTAL_FIELD_SPEC.md", "external_backend_version": None},
    "S08": {"systems": ["anatase TiO2", "rutile TiO2"], "primary_edge": "Ti K", "CIF_sources_and_hashes": None, "starting_supercell_controls": [[1,1,1], [2,2,2]], "proposed_XANES_refinement_peak_drift_eV_max": 0.1, "proposed_area_normalized_shape_change_max": 0.05, "pseudopotential_and_core_hole_identity": None, "EXAFS_k_R_windows": None},
    "S09": {"endmembers": ["LiFePO4", "FePO4"], "primary_edge": "Fe K", "formal_Fe_valence_endmembers": [2,3], "data_split_unit": "whole cell, particle, or nonoverlapping time block", "proposed_blind_mixture_fraction_absolute_error_max": 0.1, "operando_data_and_timestamps": None, "XRD_cross_validation": None},
    "S10": {"element": "Pd", "candidate_classes": ["supported isolated site", "Pd2 or small cluster", "nanoparticle reference"], "edges": ["Pd K"], "geometry_sources_and_hashes": None, "catalysis_and_independent_structural_data": None, "primary_gate": "reject candidates conflicting with independently measured structure; preserve non-identifiability", "predeclared_quantitative_inference_threshold": None},
    "S11": {"systems": ["Si", "SiO2", "Si/SiO2 interface"], "primary_observable": "Si 2p core ionization chemical shift", "auxiliary_absorption_edges": ["Si K", "O K"], "slab_orientation_termination_and_structure": None, "energy_reference_and_charged_state_correction": None, "proposed_relative_core_shift_refinement_drift_eV_max": 0.1},
    "S12": {"initial_system": "neutral Ne", "Z": 10, "initial_charge": 0, "photon_energy_eV_exploration": [800,2000], "pulse_FWHM_fs_exploration": [5,100], "fluence_photons_per_um2_exploration": [1e10,1e13], "proposed_population_sum_error_max": 1e-8, "proposed_refinement_yield_change_max": 0.02, "pulse_shape_spatial_profile_and_normalization": None, "rate_table_and_continuum_model_identity": None},
}

cases = []
for match in re.finditer(r"^### (S\d{2}) · ([^\n]+)\n(.*?)(?=^### S\d{2} · |^## 5\.|\Z)", text, re.M | re.S):
    case_id, title, body = match.groups()
    values = re.findall(r"\*\*([^*]+)：\*\*(.*?)(?=\n\n\*\*|\Z)", body.strip(), re.S)
    fields = {FIELDS[label]: value.strip() for label, value in values}
    assert len(fields) == len(FIELDS), (case_id, fields.keys())
    cases.append({"case_id": case_id, "title": title, "status": "planned_full_case_not_executed",
                  "scientific_acceptance_passed": None, "setup_parameters": PARAMS[case_id],
                  **fields, "reference_urls": sorted(set(re.findall(r"\]\((https?://[^)]+)\)", body)))})
assert [case["case_id"] for case in cases] == [f"S{i:02}" for i in range(1,13)]
registry = {
    "schema": "atomx-science-case-design-v1", "date": "2026-10-06",
    "owning_document": "研究体系与案例设计.md",
    "meaning": "research setup and proposed gates; not a universal executable Atomx API",
    "threshold_status": "proposed research targets; not measurements, industrial standards, or achieved accuracy",
    "null_policy": "required identity, data, backend, or threshold must be fixed before relevant scientific execution; never guessed",
    "global_execution_contract": ["source and input hashes", "backend versions", "state, Hamiltonian, basis, metric and units", "resource estimate and caps", "independent reference", "matched initial/final or lower/upper state controls", "failed results retained"],
    "software_demonstrations": {"count": 3, "results_directory": "foundation_results", "full_spectra_certified": False},
    "cases": cases,
}
(ROOT / "case_registry.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("Derived 12 cases, six prose fields each, declared units and proposed parameters.")
