"""One failed-kappa numerical discriminator; metadata default, explicit root binding."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent
ROOT = OLD.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def metadata(binding_path=None):
    c = load(HERE / "contract.json")
    for name, expected in c["inputs_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Immutable diagnostic input changed: " + name)
    source = load(ROOT / c["source_identity_path"])
    source_root = ROOT / c["frozen_source_root"]
    if len(source["all_atomx_python_sha256"]) != 353:
        raise ValueError("Full353 source identity required")
    digest = hashlib.sha256(json.dumps(source["all_atomx_python_sha256"], sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()
    if digest != c["source_map_sha256"]:
        raise ValueError("Wrong source epoch")
    for name, expected in source["all_atomx_python_sha256"].items():
        if sha(source_root / Path(name).relative_to("src")) != expected:
            raise ValueError("Frozen source bytes changed")
    if binding_path is not None:
        b = load(binding_path)
        if (b.get("execution_authorized_by_root") is not True
                or b.get("pending_exact_root_execution_contract") is not False
                or b.get("contract_sha256") != sha(HERE / "contract.json")
                or b.get("driver_sha256") != sha(__file__)
                or b.get("inputs_sha256") != c["inputs_sha256"]
                or b.get("source_map_sha256") != c["source_map_sha256"]
                or b.get("fixed_kappa") != -3):
            raise ValueError("NEW exact root diagnostic binding required; no physical imports")
    return c, source_root


def execute(binding_path):
    c, source_root = metadata(binding_path)
    if any((HERE / name).exists() for name in ("result.json", "failure.json", "progress.json", "operators.npz")):
        raise FileExistsError("Preserve prior diagnostic; require successor")
    started = time.monotonic()
    calls = {name: 0 for name in (
        "one_body_attempts", "one_body_completed", "public_Dirac_attempts", "public_Dirac_completed",
        "complement_SVD_attempts", "complement_SVD_completed", "condition_SVD_attempts", "condition_SVD_completed",
        "subset_values_attempts", "subset_values_completed", "subset_vectors_attempts", "subset_vectors_completed",
        "full_vectors_attempts", "full_vectors_completed", "bordered_Newton_attempts", "bordered_Newton_completed",
        "CI_solves", "many_electron_H", "virtual_generator_calls", "gradient_calls", "optimizer_calls")}
    rows = []
    try:
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
            os.environ[name] = "1"
        affinity, load_average = sorted(os.sched_getaffinity(0)), os.getloadavg()
        mem = {row.split(":")[0]: int(row.split()[1]) for row in Path("/proc/meminfo").read_text().splitlines()
               if len(row.split()) > 1 and row.split()[1].isdigit()}
        admitted = len(affinity) >= 3 and mem["MemAvailable"] / 1024**2 >= 6 and load_average[0] / max(1, len(affinity)) <= .8
        admission = dict(original_affinity=affinity, load_average=load_average,
                         host_available_GiB=mem["MemAvailable"] / 1024**2, admitted=admitted)
        write(HERE / "resource_admission.json", admission)
        if not admitted:
            raise RuntimeError("Finite diagnostic resource admission rejected")
        os.sched_setaffinity(0, {affinity[-2]})
        admission.update(pinned_affinity=sorted(os.sched_getaffinity(0)))
        write(HERE / "resource_admission.json", admission)
        byte_cap = c["resources"]["address_space_GiB"] * 1024**3
        resource.setrlimit(resource.RLIMIT_AS, (byte_cap, byte_cap))

        def timeout(*_):
            raise TimeoutError("One-kappa diagnostic wall cap")

        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(c["resources"]["wall_s"])
        sys.path.insert(0, str(source_root))
        import numpy as np
        from scipy import linalg

        from atomx.numerics.bspline.dirac_assembly import (
            build_dirac_matrices,
            solve_dirac,
        )
        from atomx.numerics.bspline.orbital_subspace import radial_projector_start
        from atomx.potentials.nuclear import make_nuclear_potential

        for name, module in tuple(sys.modules.items()):
            if ((name == "atomx" or name.startswith("atomx.")) and getattr(module, "__file__", None)
                    and not Path(module.__file__).resolve().is_relative_to(source_root.resolve())):
                raise ValueError("Physical consumer escaped frozen10d481 source")

        def progress(stage):
            write(HERE / "progress.json", dict(stage=stage, calls=calls, completed_evidence=rows,
                                              elapsed_s=time.monotonic() - started))

        def counted(attempted, completed, maximum, function, *args, **kwargs):
            if calls[attempted] >= maximum:
                raise RuntimeError("Explicit diagnostic call budget exhausted: " + attempted)
            calls[attempted] += 1
            progress("before_" + attempted)
            value = function(*args, **kwargs)
            calls[completed] += 1
            return value

        def save(name, **arrays):
            path = HERE / name
            if path.exists():
                raise FileExistsError("Preserve diagnostic arrays")
            np.savez_compressed(path, **arrays)
            return dict(path=name, sha256=sha(path))

        old_contract = load(OLD / "contract.json")
        roles = {name: ROOT / row["path"] for name, row in old_contract["input_roles"].items()}
        workspace = np.load(roles["geometry_workspace"], allow_pickle=False)
        data = {key[6:]: workspace[key].item() if workspace[key].ndim == 0 else workspace[key]
                for key in workspace.files if key.startswith("basis_")}
        checkpoint = np.load(roles["checkpoint_arrays"], allow_pickle=False)
        raw = np.load(OLD / "raw_native_virtuals.npz", allow_pickle=False)
        layout = load(roles["original_layout"])
        i = next(i for i, row in enumerate(layout) if row["keys"][0][1] == -3)
        original_p = checkpoint[f"projector_{i}"].copy()
        old_keys = [tuple(row) for row in checkpoint["orbital_keys"]]
        positions = [j for j, key in enumerate(old_keys) if key[1] == -3]
        frozen = np.column_stack([np.concatenate((checkpoint["P"][j], checkpoint["Q"][j])) for j in positions])
        raw_position = [tuple(row) for row in raw["radial_keys"]].index((6, -3))
        raw_vector = np.concatenate((raw["P"][raw_position], raw["Q"][raw_position]))
        nuclear = make_nuclear_potential(70, "fermi", A=175)
        raw_h, s = counted("one_body_attempts", "one_body_completed", 1, build_dirac_matrices,
                          123, -3, 70, data, nuclear)
        h = np.tril(raw_h) + np.tril(raw_h, -1).T
        spectral = counted("public_Dirac_attempts", "public_Dirac_completed", 1, solve_dirac,
                           h, s, 123, -3, nsp=data.get("nsp"), nsq=data.get("nsq"))
        first = radial_projector_start(-3, min(old_keys[j][0] for j in positions), data)
        fresh_p = spectral.coefficients[:, first:]
        operators = save("operators.npz", h=h, s=s, original_P=original_p, fresh_P=fresh_p,
                         frozen=frozen, raw_vector=raw_vector, original_x=checkpoint["x"],
                         physical_nuclear_eigenvalues=spectral.eigenvalues, first=np.array(first))
        if fresh_p.shape != original_p.shape:
            raise ValueError("Original/fresh retained complete subspace dimension differs")
        hld, sld = h.astype(np.longdouble), s.astype(np.longdouble)

        def rayleigh(v, high=False):
            if high:
                v = v.astype(np.longdouble)
                return float(v @ hld @ v / (v @ sld @ v))
            return float(v @ h @ v / (v @ s @ v))

        def residual(basis, vector, energy):
            r = basis.astype(np.longdouble).T @ (hld @ vector.astype(np.longdouble)
                                                  - np.longdouble(energy) * (sld @ vector.astype(np.longdouble)))
            return dict(complement_residual_norm_Ha=float(np.linalg.norm(r)),
                        complement_residual_max_Ha=float(np.max(abs(r))),
                        complement_residual_relative=float(np.linalg.norm(r)) / max(1., abs(energy)))

        for name, projector in (("original_P", original_p), ("fresh_P", fresh_p)):
            complement = counted("complement_SVD_attempts", "complement_SVD_completed", 2, linalg.null_space,
                                 frozen.T @ s @ projector)
            basis = projector @ complement
            a, b = basis.T @ h @ basis, basis.T @ s @ basis
            singular_s = counted("condition_SVD_attempts", "condition_SVD_completed", 2, linalg.svdvals, b)
            values = counted("subset_values_attempts", "subset_values_completed", 2, linalg.eigh,
                             a, b, subset_by_index=(0, 1), eigvals_only=True, driver="gvx")
            ev, cv = counted("subset_vectors_attempts", "subset_vectors_completed", 2, linalg.eigh,
                             a, b, subset_by_index=(0, 1), eigvals_only=False, driver="gvx")
            ef, cf = counted("full_vectors_attempts", "full_vectors_completed", 2, linalg.eigh,
                             a, b, eigvals_only=False, driver="gvd")
            archive = save(name + "_complement.npz", basis=basis, projected_h=a, projected_s=b,
                           subset_values=values, subset_vector_energies=ev, subset_vectors=cv,
                           full_energies=ef, full_vectors=cf, projected_s_singular_values=singular_s)
            vector_rows = []
            for method, energies, vectors in (("gvx_vectors", ev, cv), ("full_gvd", ef[:2], cf[:, :2])):
                for root in range(2):
                    vector = basis @ vectors[:, root]
                    energy, energy_ld = rayleigh(vector), rayleigh(vector, True)
                    vector_rows.append(dict(method=method, root=root, reported_energy_Ha=float(energies[root]),
                        float_Rayleigh_Ha=energy, longdouble_Rayleigh_Ha=energy_ld,
                        reported_minus_longdouble_Ha=float(energies[root] - energy_ld),
                        raw_minus_vector_Rayleigh_Ha=rayleigh(raw_vector, True) - energy_ld,
                        raw_vector_S_overlap_squared=float(raw_vector @ s @ vector)**2,
                        **residual(basis, vector, energy_ld)))
            raw_metrics = dict(float_Rayleigh_Ha=rayleigh(raw_vector), longdouble_Rayleigh_Ha=rayleigh(raw_vector, True),
                               raw_minus_reported_minimum_Ha=rayleigh(raw_vector) - float(values[0]),
                               **residual(basis, raw_vector, rayleigh(raw_vector, True)))
            row = dict(name=name, arrays=archive, complement_dimension=basis.shape[1],
                       original_minimum_gate_Ha=1e-8,
                       original_reported_minimum_gate_passed=abs(raw_metrics["raw_minus_reported_minimum_Ha"]) <= 1e-8,
                       projected_H_inf_norm_Ha=float(np.linalg.norm(a, np.inf)),
                       projected_H_maxabs_Ha=float(np.max(abs(a))),
                       projected_H_symmetry_error_Ha=float(np.max(abs(a-a.T))),
                       projected_S_inf_norm=float(np.linalg.norm(b, np.inf)),
                       projected_S_condition=float(singular_s.max() / singular_s.min()),
                       projected_S_Gram_error=float(np.max(abs(b-np.eye(len(b))))),
                       full_spectral_span_Ha=float(ef[-1]-ef[0]),
                       full_minimum_gap_Ha=float(ef[1]-ef[0]),
                       machine_epsilon_times_H_inf_norm_Ha=float(np.finfo(float).eps * np.linalg.norm(a, np.inf)),
                       raw_vector=raw_metrics, eigenvector_controls=vector_rows,
                       Newton_controls=[])
            rows.append(row)
            progress(name + "/spectral_evidence_saved")
            if name == "original_P":
                y = cv[:, 0].copy()
                energy = rayleigh(basis @ y, True)
                original_y = y.copy()
                for iteration in range(2):
                    r = np.asarray(a.astype(np.longdouble) @ y.astype(np.longdouble)
                                   - np.longdouble(energy) * (b.astype(np.longdouble) @ y.astype(np.longdouble)), dtype=float)
                    by = b @ y
                    border = np.block([[a - energy * b, -by[:, None]],
                                       [-by[None, :], np.zeros((1, 1))]])
                    delta = counted("bordered_Newton_attempts", "bordered_Newton_completed", 2, linalg.solve,
                                    border, np.r_[-r, 0.], assume_a="sym", check_finite=True)
                    y += delta[:-1]
                    y /= np.sqrt(y @ b @ y)
                    vector = basis @ y
                    energy = rayleigh(vector, True)
                    control = dict(iteration=iteration+1, longdouble_Rayleigh_Ha=energy,
                                   raw_minus_refined_Rayleigh_Ha=rayleigh(raw_vector, True)-energy,
                                   original_subset_vector_overlap_squared=float(original_y @ b @ y)**2,
                                   **residual(basis, vector, energy))
                    row["Newton_controls"].append(control)
                    save("Newton" + str(iteration+1) + ".npz", coordinates=y.copy(), vector=vector)
                progress("original_P/refinement_diagnostic_saved")
        result = dict(status="one_failed_kappa_saved_operator_spectral_diagnostic_completed",
                      contract_sha256=sha(HERE / "contract.json"), driver_sha256=sha(__file__),
                      execution_binding_sha256=sha(binding_path), source_map_sha256=c["source_map_sha256"],
                      inherited_failure_sha256=sha(OLD / "failure.json"), operators=operators,
                      calls=calls, complements=rows, raw_vector_unchanged=True,
                      no_repaired_generator_or_threepoint_claim=True,
                      original_scientific_gate_Ha=1e-8, original_stationarity_passed=False,
                      experimental_gate_passed=False, new_state_or_boundary=False,
                      independent_saved_array_audit_pending=True, elapsed_uncontrolled_s=time.monotonic()-started,
                      peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        write(HERE / "result.json", result)
        print(json.dumps(dict(status=result["status"], calls=calls, complements=rows)), flush=True)
    except Exception as error:
        write(HERE / "failure.json", dict(error_type=type(error).__name__, message=str(error),
            traceback=traceback.format_exc(), calls=calls, completed_evidence=rows,
            contract_sha256=sha(HERE / "contract.json"), driver_sha256=sha(__file__),
            execution_binding_sha256=sha(binding_path), scientific_success_claim=False))
        raise
    finally:
        signal.alarm(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorize", action="store_true")
    parser.add_argument("--binding", type=Path)
    args = parser.parse_args()
    if args.execute or args.authorize:
        if args.binding is None:
            parser.error("NEW root diagnostic binding required")
        if args.execute:
            execute(args.binding)
        else:
            metadata(args.binding)
            print(json.dumps(dict(metadata_authorized=True, physical_calls=0, heavy_imports=0)))
    else:
        metadata()
        print(json.dumps(dict(status="one_failed_kappa_prepared_only", physical_calls=0, heavy_imports=0)))


if __name__ == "__main__":
    main()
