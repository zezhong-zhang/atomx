"""Replay native Yb states and measure same-orbital correction differences."""
from __future__ import annotations

import hashlib
import json
import os
import resource
import signal
import sys
import time
import traceback
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    out = HERE / "same_orbital001"
    out.mkdir(exist_ok=False)
    contract_path = HERE / "contract.json"
    contract = json.loads(contract_path.read_text())
    fixed, gates = contract["fixed_conditions"], contract["acceptance"]
    inp = ROOT / fixed["input_run"]
    for name, expected in contract["inputs"].items():
        if sha(inp / name) != expected:
            raise ValueError(f"saved input identity changed: {name}")
    frozen = ROOT / fixed["frozen_source_root"]
    source = json.loads((inp / "source_identity.json").read_text())
    for name, expected in source["all_atomx_python_sha256"].items():
        path = frozen / Path(name).relative_to("src")
        if sha(path) != expected:
            raise ValueError(f"frozen scientific source changed: {name}")
    write(out / "preflight.json", {"source_files_verified": len(source["all_atomx_python_sha256"]),
          "source_map_sha256": source["source_map_sha256"], "contract_sha256": sha(contract_path),
          "input_hashes_verified": contract["inputs"], "load_average": os.getloadavg(),
          "cap_GiB": 4, "timeout_seconds": 900, "scope": "Same-orbital diagnostic, no certified spectrum"})
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    started = time.monotonic()

    def timeout(*_):
        raise TimeoutError("prospectively declared correction-budget 900 s wall limit exceeded")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(900)
    sys.path.insert(0, str(frozen))
    import numpy as np

    from atomx.config.csf import enumerate_jj_csfs, get_shared_orbital_basis
    from atomx.config.subshell import Subshell
    from atomx.constants import HARTREE_CM1
    from atomx.corrections.qed import (
        self_energy_radial_matrix,
        short_range_radial_matrix,
        uehling_potential,
    )
    from atomx.corrections.qedmod.flambaum_ginges import V_wichmann_kroll
    from atomx.corrections.recoil import (
        recoil_radial_matrix,
        recoil_state_parameters,
    )
    from atomx.corrections.specific_recoil import (
        momentum_recoil_matrix,
    )
    from atomx.methods.ci import ci_solve
    from atomx.methods.csf.variational import CSFOrbitalFunctional
    from atomx.physics.csf.matrix import build_ci_matrix
    from atomx.physics.csf.one_body import scalar_one_body_expectations
    from atomx.physics.csf.state_average import CIStateAverage
    from atomx.potentials.nuclear import make_nuclear_potential, nuclear_profile
    from atomx.properties.isotope_shift.mass_shift import default_nuclear_mass_me

    baseline = json.loads((inp / "result.json").read_text())
    saved_contract = json.loads((inp / "contract.json").read_text())
    arrays = np.load(inp / "orbitals_workspace.npz")
    saved_ci = np.load(inp / "ci_arrays.npz")
    workspace = {key.removeprefix("basis_"): arrays[key].item() if arrays[key].ndim == 0 else arrays[key]
                 for key in arrays.files if key.startswith("basis_")}
    keys = tuple(map(tuple, arrays["orbital_keys"].tolist()))
    orbitals = {key: (p, q) for key, p, q in zip(keys, arrays["P"], arrays["Q"], strict=True)}
    csfs = [csf for config in saved_contract["fixed_conditions"]["configs"]
            for csf in enumerate_jj_csfs(config)]
    if len(csfs) != 5 or len(keys) != 24 or workspace["ns"] != 123:
        raise ValueError("saved physical representation changed")
    nucleus = make_nuclear_potential(70, "fermi", A=175)
    if nuclear_profile(70, "fermi") != fixed["nucleus"]:
        raise ValueError("Fermi nuclear identity changed")
    common = dict(csfs=csfs, orbitals_dict=orbitals, orbital_basis=get_shared_orbital_basis(csfs),
                  bsp_data=workspace, Z=70, ns=123, nuclear_potential=nucleus)
    stage_seconds, checks, components, warnings_by_component, matrices = {}, [], {}, {}, {}

    def run_stage(name, call):
        start = time.monotonic()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = call()
        stage_seconds[name] = time.monotonic() - start
        warnings_by_component[name] = [str(item.message) for item in caught]
        print(json.dumps({"stage": name, "seconds": stage_seconds[name],
                          "elapsed_seconds": time.monotonic() - started}), flush=True)
        return value

    def record_check(name, difference, limit):
        checks.append({"name": name, "maximum_absolute_Ha": float(difference),
                       "limit_Ha": limit, "passed": bool(difference <= limit)})
        if difference > limit:
            raise AssertionError(f"{name}: {difference} > {limit}")

    def state_map(ci):
        result = {}
        for col in range(ci.mixing_coefficients.shape[1]):
            idx = int(np.argmax(np.abs(ci.mixing_coefficients[:, col])))
            if abs(ci.mixing_coefficients[idx, col]) < 1 - 1e-12:
                raise ValueError("expected singleton CI symmetry root lost identity")
            csf = csfs[idx]
            result[csf.total_J2, csf.parity] = col
        if len(result) != 5:
            raise ValueError("missing five target roots")
        return result

    target_symmetries = [(row["twoJ"], row["parity"]) for row in baseline["rows_dc"]]
    dc = run_stage("DC_replay", lambda: ci_solve(**common, solver="dense"))
    dc_map = state_map(dc)
    dc_order = [dc_map[sym] for sym in target_symmetries]
    dc_energies = dc.state_energies[dc_order]
    mixing = dc.mixing_coefficients[:, dc_order]
    record_check("saved_DC_roots", np.max(np.abs(dc_energies - saved_ci["replay_dc_energies_Ha"])),
                 gates["dc_replay_absolute_Ha"])
    matrix_dc = build_ci_matrix(**common)
    record_check("DC_CI_residual", np.max(np.linalg.norm(matrix_dc @ mixing - mixing * dc_energies, axis=0)),
                 gates["CI_residual_absolute_Ha"])
    static = run_stage("static_Breit", lambda: ci_solve(**common, include_breit=True, solver="dense"))
    static_map = state_map(static)
    static_energies = static.state_energies[[static_map[sym] for sym in target_symmetries]]
    record_check("saved_static_Breit_roots", np.max(np.abs(static_energies - saved_ci["rci_energies_Ha"])),
                 gates["static_replay_absolute_Ha"])
    components["static_Breit"] = static_energies - dc_energies
    zero = run_stage("dynamic_zero_frequency", lambda: ci_solve(**common, include_breit=True, solver="dense",
                                                                breit_orbital_energies=dict.fromkeys(keys, 0.)))
    zero_map = state_map(zero)
    zero_energies = zero.state_energies[[zero_map[sym] for sym in target_symmetries]]
    record_check("dynamic_zero_frequency_equals_static", np.max(np.abs(zero_energies - static_energies)),
                 gates["zero_frequency_static_absolute_Ha"])
    functional = CSFOrbitalFunctional(csfs, workspace, 70, nucleus,
        state_average=CIStateAverage(tuple(map(tuple, saved_contract["fixed_conditions"]["selected_levels"]))))
    state_weights = functional.state_average.weights(csfs, dc)
    radial_weights = functional.radial_density.contract(dc, state_weights)
    # Use the exact saved same-state gradient to reproduce the solver's public orbital-energy convention.
    gradient = {key: value for key, value in zip(map(tuple, saved_ci["gradient_keys"].tolist()),
                                                saved_ci["gradient"], strict=True)}
    eps = {}
    for key in keys:
        vector = np.concatenate(orbitals[key])
        occupation = radial_weights.get(("h", key[0], key[1], key[0]), 0.)
        if occupation <= 1e-10:
            eps[key] = float(vector @ functional.one_body[key[1]] @ vector)
        else:
            eps[key] = float(vector @ gradient[key] / (2 * occupation))
    write(out / "orbital_frequency_energies.json", {"prescription": contract["components"]["dynamic_Breit"],
          "same_orbital_gradient_sha256": contract["inputs"]["ci_arrays.npz"],
          "energy_map": [{"n": n, "kappa": k, "energy_Ha": eps[n, k]} for n, k in keys]})
    dynamic = run_stage("dynamic_Breit", lambda: ci_solve(**common, include_breit=True, solver="dense",
                                                        breit_orbital_energies=eps))
    dynamic_map = state_map(dynamic)
    dynamic_energies = dynamic.state_energies[[dynamic_map[sym] for sym in target_symmetries]]
    components["dynamic_Breit_total"] = dynamic_energies - dc_energies
    components["dynamic_minus_static_Breit"] = dynamic_energies - static_energies
    waves = [(*orbitals[key], key[1]) for key in keys]
    se_waves = [(*orbitals[key], key[1], Subshell(key[0], key[1], 1).label) for key in keys]

    def one_body(name, builder):
        radial = run_stage(name, builder)
        if not np.isfinite(radial).all():
            raise FloatingPointError(f"nonfinite radial correction {name}")
        matrices[name] = radial
        shift = scalar_one_body_expectations(csfs, mixing, keys, radial)
        independent = []
        key_index = {key: i for i, key in enumerate(keys)}
        for col in range(5):
            csf = csfs[int(np.argmax(np.abs(mixing[:, col])))]
            independent.append(sum(sh.occupation * radial[key_index[sh.n, sh.kappa], key_index[sh.n, sh.kappa]]
                                   for sh in csf.subshells))
        record_check(name + "_scalar_vs_independent_occupation", np.max(np.abs(shift - independent)),
                     gates["scalar_onebody_independent_occupation_absolute_Ha"])
        components[name] = shift

    one_body("Uehling_point_source", lambda: short_range_radial_matrix(waves, workspace,
                                                                     lambda r: uehling_potential(r, 70)))
    one_body("WK_local_model", lambda: short_range_radial_matrix(waves, workspace,
                                                                lambda r: V_wichmann_kroll(r, 70)))
    one_body("SE_indelicato_overlap_model", lambda: self_energy_radial_matrix(se_waves, 70, workspace,
                                                                             method="indelicato"))
    one_body("SE_shabaev_local_partial", lambda: self_energy_radial_matrix(se_waves, 70, workspace,
                                                                          method="shabaev_local"))
    unsupported = {"finite_charge_Uehling_and_WK": "local public potentials do not consume the A175 Fermi charge distribution",
                   "screened_many_body_SE_and_VP": "not supplied by these fixed one-body models",
                   "finite_size_radiative_higher_order_recoil": "point-Coulomb Breit recoil consumers omit these operators"}
    try:
        run_stage("SE_shabaev_full", lambda: self_energy_radial_matrix(se_waves, 70, workspace, method="shabaev"))
    except NotImplementedError as exc:
        unsupported["SE_shabaev_full"] = str(exc)
    mass = default_nuclear_mass_me(70, 175)
    nms_matrix = run_stage("NMS_Breit_radial", lambda: recoil_radial_matrix(waves, workspace, 70))
    matrices["NMS_Breit_radial"] = nms_matrix
    nms = recoil_state_parameters(csfs, mixing, orbitals, workspace, 70) / mass
    nms_expected = scalar_one_body_expectations(csfs, mixing, keys, nms_matrix) / mass
    record_check("NMS_public_state_vs_radial_consumer", np.max(np.abs(nms - nms_expected)),
                 gates["scalar_onebody_independent_occupation_absolute_Ha"])
    components["NMS_Breit_point_Coulomb"] = nms
    sms_matrix = run_stage("SMS_Breit_CSF", lambda: momentum_recoil_matrix(csfs, orbitals, workspace, Z=70))
    matrices["SMS_Breit_CSF"] = sms_matrix
    components["SMS_Breit_point_Coulomb"] = np.einsum("ik,ij,jk->k", mixing, sms_matrix, mixing, optimize=True) / mass
    for name, values in components.items():
        if not np.isfinite(values).all():
            raise FloatingPointError(f"nonfinite state correction {name}")
    ledgers = []
    for i, row in enumerate(baseline["rows_dc"]):
        entries = {name: {"absolute_Ha": float(values[i]),
                          "relative_to_6s_cm1": float((values[i] - values[0]) * HARTREE_CM1)}
                   for name, values in components.items()}
        ledgers.append({"label": row["label"], "twoJ": row["twoJ"], "parity": row["parity"],
                        "DC_excitation_cm1": row["computed_cm1"], "NIST_nominal_cm1": row["experimental_cm1"],
                        "DC_minus_NIST_cm1": row["error_cm1"], "components": entries})
    splittings = {}
    for name, lower, upper in (("5d_fine", 1, 2), ("6p_fine", 3, 4)):
        splittings[name] = {component: float((values[upper] - values[lower]) * HARTREE_CM1)
                           for component, values in components.items()}
    result = {"passed": all(c["passed"] for c in checks), "production_certified": False,
              "stationarity_passed": False, "experimental_gate_passed": False,
              "diagnostic_only": True, "no_complete_corrected_spectrum_claim": True,
              "input_run": fixed["input_run"], "source_map_sha256": fixed["source_map_sha256"],
              "contract_sha256": sha(contract_path), "input_hashes": contract["inputs"],
              "checks": checks, "state_correction_ledger": ledgers, "splitting_correction_cm1": splittings,
              "component_coverage": contract["components"], "unsupported": unsupported,
              "warnings_by_component": warnings_by_component,
              "nucleus": fixed["nucleus"], "nuclear_mass_me": mass,
              "mass_prescription": fixed["mass_proxy"], "dynamic_Breit_provenance": dynamic.provenance,
              "resource": {"total_seconds_uncontrolled": time.monotonic() - started,
                           "stage_seconds_uncontrolled": stage_seconds,
                           "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                           "controlled_timing_claim": False},
              "original_stationarity": {key: baseline["orbital_optimization"][key]
                                        for key in ("orbital_residual", "orbital_tolerance", "estimated_energy_error",
                                                    "energy_tolerance", "termination_reason")}}
    np.savez_compressed(out / "correction_arrays.npz", energies_DC_Ha=dc_energies,
                        energies_static_Breit_Ha=static_energies, energies_dynamic_Breit_Ha=dynamic_energies,
                        mixing_DC=mixing, **{name: values for name, values in components.items()},
                        **{"matrix_" + name: values for name, values in matrices.items()})
    result["correction_arrays_sha256"] = sha(out / "correction_arrays.npz")
    write(out / "result.json", result)
    print(json.dumps({"passed": result["passed"], "result": str(out / "result.json"),
                      "ledger": ledgers, "splittings": splittings}, ensure_ascii=False), flush=True)
    signal.alarm(0)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        out = HERE / "same_orbital001"
        if out.exists():
            write(out / "failure.json", {"exception": repr(exc), "traceback": traceback.format_exc(),
                                        "production_certified": False})
        raise
