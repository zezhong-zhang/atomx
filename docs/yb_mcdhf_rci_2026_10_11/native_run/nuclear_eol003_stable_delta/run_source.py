"""Prospectively bound native five-level Yb II MCDHF and same-orbital RCI."""

import argparse
import hashlib
import json
import os
import resource
import shutil
import signal
import sys
import threading
import time
import traceback
from pathlib import Path


def plain(value):
    import numpy as np

    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, float, int, bool)):
        return value
    return repr(value)


def write(path, value):
    path.write_text(json.dumps(plain(value), indent=2, ensure_ascii=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--trace-line-search", action="store_true")
    parser.add_argument("--numerical-successor-spec")
    args = parser.parse_args()
    owner = Path(__file__).resolve().parent
    repo = owner.parents[2]
    out = owner / args.label
    out.mkdir(exist_ok=False)
    core = "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f14"
    configs = [core + " " + tail for tail in ("6s1", "5d1", "6p1")]
    selectors = ((1, 1, 0, 1), (3, 1, 0, 1), (5, 1, 0, 1),
                 (1, -1, 0, 1), (3, -1, 0, 1))
    reference = repo / "docs/rare_earth_yb_spectrum_2026_10_08/experimental_reference.json"
    contract = {
        "hypothesis": "The current native coherent five-level EOL optimizer must reach its declared constrained stationarity before its same-orbital RCI spectrum is production evidence; a prior line-search failure cannot be repaired by declaring frozen DF orbitals MCDHF.",
        "fixed_conditions": {"Z": 70, "charge": 1, "electrons": 69,
            "configs": configs, "selected_levels": selectors,
            "weights": "equal 1/5 fixed by symmetry/local-root identity",
            "Hamiltonian": "finite Fermi nucleus Dirac-Coulomb with fixed nuclear positive-energy B-spline projector",
            "nuclear_mass_number": "public empirical default, not isotope-resolved",
            "engine": "bspline", "grid": {"ks": 9, "hi": 0.25,
                "he": 0.25, "hmax": 1.0, "r_max": 60.0},
            "radial_optimization": "mcdhf", "variational_projector": "nuclear",
            "max_iter": 12, "effective_gradient_max_iter": 384,
            "scf_max_iter": 40, "orbital_tolerance": 1e-7,
            "energy_tolerance_Ha": 1e-8,
            "all_24_radial_orbitals_optimized": True, "frozen_core_radials": False,
            "RCI": "same CSFs and stationary common orbitals, complete static Breit, no QED/recoil/empirical correction",
            "threads": 1},
        "acceptance": {"converged": True, "orbital_residual_lt": 1e-7,
            "estimated_energy_error_Ha_lt": 1e-8,
            "metric_orthogonality_max_error_lt": 1e-8,
            "dense_CI_residual_Ha_lt": 1e-8,
            "NIST_relative_excitation_error_goal": 0.01},
        "distinguishing_outcomes": "A stationary five-state solve establishes the variational mean-field/EOL baseline only. Failure identifies numerical optimization work before adding correlation; larger active spaces do not excuse failed residual gates.",
        "diagnostic_observation": {"line_search_trace": args.trace_line_search,
            "rule": "Python frame observation only; returned energies/gradients, optimization directions, line-search decisions and convergence gates are unchanged"},
        "stopping_condition": {"wall_seconds": args.timeout,
            "address_space_cap_GiB": 4, "retain_exception_and_unconverged_outputs": True},
        "correlation_limit": "Five CSFs in five different symmetry blocks: no within-block correlation mixing or core-hole excitations. The 69-electron core participates in common radial optimization but core-valence dynamical correlation is absent.",
        "reference": {"path": str(reference.relative_to(repo)), "sha256": sha(reference),
            "type": "NIST handbook measured level energies, uncertainty unspecified"},
        "historical_failed_same_model": [
            "docs/rare_earth_yb_spectrum_2026_10_08/results/dc_default_replay/structure.json",
            "docs/rare_earth_yb_spectrum_2026_10_08/results/dc_refined/structure.json"],
        "not_claimed": ["radial convergence", "full electron correlation", "isotope-resolved experiment", "highest atomic accuracy", "controlled performance speedup"]}
    if args.numerical_successor_spec:
        successor_path = Path(args.numerical_successor_spec).resolve()
        successor = json.loads(successor_path.read_text())
        if successor["fixed_conditions"] != plain(contract["fixed_conditions"]):
            raise ValueError("numerical successor changed fixed physical model or convergence gates")
        contract["numerical_successor"] = {"path": str(successor_path),
            "sha256": sha(successor_path), "specification": successor}
        shutil.copyfile(successor_path, out / "numerical_successor_spec.json")
    write(out / "contract.json", contract)
    shutil.copyfile(Path(__file__), out / "run_source.py")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    sources = {}
    for path in sorted((repo / "src/atomx").rglob("*.py")):
        relative = path.relative_to(repo / "src")
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        sources[str(path.relative_to(repo))] = sha(path)
    write(out / "source_identity.json", {"all_atomx_python_sha256": sources,
        "source_map_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        "script_sha256": sha(Path(__file__)), "import_root": str(out / "source")})
    sys.path.insert(0, str(out / "source"))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    import numpy as np

    from atomx.config.csf import enumerate_jj_csfs, get_shared_orbital_basis
    from atomx.constants import HARTREE_CM1
    from atomx.methods.ci import ci_solve
    from atomx.methods.csf.variational import CSFOrbitalFunctional
    from atomx.methods.mcdhf.solver import MCDHFSolver
    from atomx.physics.csf.matrix import build_ci_matrix
    from atomx.physics.csf.state_average import CIStateAverage
    from atomx.potentials.nuclear import nuclear_profile
    csfs = [csf for config in configs for csf in enumerate_jj_csfs(config)]
    keys = get_shared_orbital_basis(csfs)
    write(out / "preflight.json", {"CSFs": len(csfs), "radial_orbitals": len(keys),
        "electron_counts": [csf.n_electrons for csf in csfs],
        "symmetry_blocks": [(csf.total_J2, csf.parity) for csf in csfs],
        "each_symmetry_dimension": 1, "expected_actual_ns": 123,
        "resident_memory_estimate_GiB": 2.5, "peak_dense_radial_dimension": 246,
        "no_69_electron_determinant_enumeration": True,
        "load_average": os.getloadavg(), "logical_CPUs": os.cpu_count(),
        "nucleus": nuclear_profile(70, "fermi"), "orbitals": keys})
    started = time.monotonic()
    done = threading.Event()
    def heartbeat():
        while not done.wait(30):
            print(json.dumps({"heartbeat_seconds": time.monotonic() - started,
                "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}), flush=True)
    threading.Thread(target=heartbeat, daemon=True).start()
    def timeout(*_):
        raise TimeoutError("prospectively declared Yb MCDHF wall limit exceeded")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.timeout)
    trace_rows = []
    if args.trace_line_search:
        import inspect

        from atomx.numerics.bspline.orbital_optimization import optimize_orbitals

        optimize_file = inspect.getsourcefile(optimize_orbitals)
        file_lines = Path(optimize_file).read_text().splitlines()
        trial_line = next(index for index, line in enumerate(file_lines, 1)
            if line.strip().startswith("if np.isfinite(trial_energy)"))
        failed_line = next(index for index, line in enumerate(file_lines, 1)
            if line.strip() == 'termination_reason = "line_search_failed"')
        def trace_optimizer(frame, event, _argument):
            if frame.f_code.co_filename != optimize_file:
                return None
            if event != "line":
                return trace_optimizer
            values = frame.f_locals
            if frame.f_lineno == trial_line:
                trace_rows.append({"iteration": values["iteration"],
                    "attempt": values["attempt"], "step": values["step"],
                    "base_E_Ha": values["energy"],
                    "trial_E_Ha": values["trial_energy"],
                    "trial_minus_base_Ha": values["trial_energy"] - values["energy"],
                    "derivative_Ha": values["derivative"],
                    "linear_change_Ha": values["step"] * values["derivative"],
                    "armijo_change_Ha": 1e-4 * values["step"] * values["derivative"],
                    "base_orbital_residual": values["residual"],
                    "base_energy_error_Ha": values["energy_error"]})
            if frame.f_lineno == failed_line:
                write(out / "failed_line_search.json", trace_rows)
                coordinate_arrays = {name: values[name] for name in
                    ("x", "g", "direction", "preconditioned_gradient", "inverse_diagonal")}
                layout_metadata = []
                for index, (group_keys, projector, shape, where) in enumerate(values["layout"]):
                    coordinate_arrays[f"projector_{index}"] = projector
                    coordinate_arrays[f"inverse_blocks_{index}"] = np.array(values["inverse_blocks"][index])
                    layout_metadata.append({"keys": group_keys, "shape": shape,
                        "slice": [where.start, where.stop]})
                np.savez_compressed(out / "failed_coordinates.npz", **coordinate_arrays)
                write(out / "failed_coordinate_layout.json", layout_metadata)
            return trace_optimizer
        sys.settrace(trace_optimizer)
    try:
        solver = MCDHFSolver(70, csfs, ns=80, ks=9, r_max=60.0,
            grid_hi=0.25, grid_he=0.25, grid_hmax=1.0, nuclear="fermi",
            variational_projector="nuclear", radial_optimization="mcdhf",
            state_average=CIStateAverage(selectors), include_breit=False,
            include_qed=False)
        result = solver.solve(max_iter=12, scf_tol=1e-7, ci_tol=1e-8,
                              scf_max_iter=40)
        sys.settrace(None)
        if args.trace_line_search:
            write(out / "line_search_trials.json", trace_rows)
        solve_seconds = time.monotonic() - started
        arrays = {"orbital_keys": np.array(sorted(result.orbitals_dict)),
            "P": np.array([result.orbitals_dict[key][0] for key in sorted(result.orbitals_dict)]),
            "Q": np.array([result.orbitals_dict[key][1] for key in sorted(result.orbitals_dict)]),
            "energies_dc_Ha": result.state_energies,
            "mixing_dc": result.mixing_coefficients}
        arrays.update({"basis_" + key: value for key, value in result.bsp_data.items()
                       if isinstance(value, (np.ndarray, int, float))})
        np.savez_compressed(out / "orbitals_workspace.npz", **arrays)
        write(out / "mcdhf_diagnostics.json", {"converged": result.converged,
            "method_name": result.method_name,
            "orbital_optimization": result.orbital_optimization,
            "solver_diagnostics": result.solver_diagnostics,
            "calculation_spec": result.calculation_spec.to_dict(),
            "actual_ns": result.bsp_data["ns"], "solve_seconds_uncontrolled": solve_seconds})
        functional = CSFOrbitalFunctional(csfs, result.bsp_data, 70,
            result._nuclear_potential, state_average=CIStateAverage(selectors))
        objective, gradient, replay_ci = functional.evaluate(result.orbitals_dict)
        orthogonality_error = 0.0
        for kappa in {key[1] for key in keys}:
            group = [key for key in keys if key[1] == kappa]
            coefficients = np.column_stack([np.concatenate(result.orbitals_dict[key]) for key in group])
            orthogonality_error = max(orthogonality_error,
                float(np.max(abs(coefficients.T @ functional.metric[kappa] @ coefficients - np.eye(len(group))))))
        common = dict(csfs=csfs, orbitals_dict=result.orbitals_dict,
            orbital_basis=keys, bsp_data=result.bsp_data, Z=70,
            ns=result.bsp_data["ns"], nuclear_potential=result._nuclear_potential)
        matrix_dc = build_ci_matrix(**common)
        ci_residual = float(np.max(np.linalg.norm(matrix_dc @ replay_ci.mixing_coefficients
            - replay_ci.mixing_coefficients * replay_ci.state_energies, axis=0)))
        rci_started = time.monotonic()
        rci = ci_solve(**common, include_breit=True, solver="dense")
        rci_seconds = time.monotonic() - rci_started
        np.savez_compressed(out / "ci_arrays.npz", matrix_dc=matrix_dc,
            replay_dc_energies_Ha=replay_ci.state_energies,
            replay_dc_mixing=replay_ci.mixing_coefficients,
            rci_energies_Ha=rci.state_energies, rci_mixing=rci.mixing_coefficients,
            gradient_keys=np.array(sorted(gradient)),
            gradient=np.array([gradient[key] for key in sorted(gradient)]))
        nominal = json.loads(reference.read_text())
        experimental = {(level["two_J"], level["parity"]): level for level in nominal["levels"]}
        def rows(ci):
            output = []
            for index, energy in enumerate(ci.state_energies):
                dominant = int(np.argmax(abs(ci.mixing_coefficients[:, index])))
                csf = csfs[dominant]
                ref = experimental[(csf.total_J2, csf.parity)]
                calculated = float((energy - ci.state_energies[0]) * HARTREE_CM1)
                error = calculated - ref["excitation_cm1"]
                relative = error / ref["excitation_cm1"] if ref["excitation_cm1"] else None
                output.append({"label": ref["label"], "twoJ": csf.total_J2,
                    "parity": csf.parity, "energy_Ha": float(energy),
                    "computed_cm1": calculated, "experimental_cm1": ref["excitation_cm1"],
                    "error_cm1": error, "relative_error": relative,
                    "gate_pass": abs(relative) < .01 if relative is not None else None,
                    "dominant_CSF_weight": float(ci.mixing_coefficients[dominant, index] ** 2),
                    "reference_type": "NIST handbook measured level, uncertainty unspecified"})
            return output
        diag = result.orbital_optimization
        stationarity = bool(result.converged and diag["orbital_residual"] < 1e-7
                            and diag["estimated_energy_error"] < 1e-8)
        numeric = stationarity and orthogonality_error < 1e-8 and ci_residual < 1e-8
        dc_rows, rci_rows = rows(replay_ci), rows(rci)
        write(out / "result.json", {"computed_run": args.label,
            "method": "Atomx native MCDHF EOL -> same-orbital RCI static Breit",
            "production_certified": numeric, "stationarity_passed": stationarity,
            "experimental_gate_passed": numeric and all(row["gate_pass"] is not False for row in rci_rows),
            "highest_level_reached": False, "fitted_corrections": False,
            "rows_dc": dc_rows, "rows_rci": rci_rows,
            "orbital_optimization": diag,
            "independent_CI_residual_Ha": ci_residual,
            "independent_CI_energy_replay_max_error_Ha": float(np.max(abs(replay_ci.state_energies-result.state_energies))),
            "common_metric_orthogonality_max_error": orthogonality_error,
            "objective_replay_Ha": objective,
            "rcI_provenance": rci.provenance,
            "diagnostic_only_if_not_stationary": not numeric,
            "scope": contract["correlation_limit"],
            "resource": {"solve_seconds_uncontrolled": solve_seconds,
                "RCI_seconds_uncontrolled": rci_seconds,
                "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "controlled_performance": False},
            "artifact_sha256": {name: sha(out / name) for name in
                ("contract.json", "source_identity.json", "preflight.json",
                 "orbitals_workspace.npz", "ci_arrays.npz", "mcdhf_diagnostics.json")}})
        print(json.dumps({"finished": True, "production_certified": numeric,
            "orbital_residual": diag["orbital_residual"],
            "stationarity_termination": diag["termination_reason"],
            "result": str(out / "result.json")}), flush=True)
    except BaseException as exc:
        write(out / "failure.json", {"exception": repr(exc), "traceback": traceback.format_exc(),
            "elapsed_seconds": time.monotonic() - started,
            "peak_RSS_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "production_certified": False})
        traceback.print_exc()
        raise
    finally:
        sys.settrace(None)
        signal.alarm(0)
        done.set()


if __name__ == "__main__":
    main()
