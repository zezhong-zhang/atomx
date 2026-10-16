"""Independent corrected EXP-T comparison of the unchanged N3/seven-spinor toy."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import signal
import sys
import time
from collections.abc import Mapping
from pathlib import Path

HERE = Path(__file__).resolve().parent
THREADS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def json_default(value):
    if isinstance(value, Mapping):
        return dict(value)
    if type(value).__module__ == "numpy" and hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Unsupported evidence scalar: {type(value).__name__}")


def source(contract):
    for path, digest in contract["sources"].items():
        if sha(path) != digest:
            raise RuntimeError(f"source/input binding changed: {path}")
    identity = json.loads((HERE / "source_snapshot_identity.json").read_text())
    for path, digest in identity["current_files"].items():
        if (
            sha(HERE / "source_snapshot/atomx" / Path(path).relative_to("src/atomx"))
            != digest
        ):
            raise RuntimeError(f"private source snapshot changed: {path}")
    backend = json.loads(Path(contract["backend_manifest"]).read_text())
    if (
        backend["noncanonical_attachment_equation"] != "ATOMX_FOV_T2_001"
        or backend["source_sha256"]["src/rcc/models/sector01.c"]
        != "f73ee35aead1a0f856012db090d50892e0503691aa0551edd36ffa6b6f32204c"
        or sha(contract["executable"]) != backend["executable_sha256"]
    ):
        raise RuntimeError("corrected tensor equation capability binding rejected")
    for path, digest in backend["source_sha256"].items():
        if sha(Path(contract["backend_source"]) / path) != digest:
            raise RuntimeError(f"tensor source binding changed: {path}")
    return dict(
        status="SOURCE_ready",
        private_python_sources=len(identity["current_files"]),
        tensor_sources=len(backend["source_sha256"]),
        numerical_calls=0,
        scientific_model="N3/seven-spinor finite synthetic two-body toy; not a closed atomic core",
    )


def execute(contract):
    if (HERE / "actual").exists():
        raise RuntimeError("existing execution evidence; no automatic retry")
    cap = contract["resources"]
    if any(os.environ.get(t) != "1" for t in THREADS) or os.sched_getaffinity(0) != {
        cap["CPU"]
    }:
        raise RuntimeError("declared CPU/serial-thread admission failed")
    available = next(
        int(s.split()[1]) * 1024
        for s in Path("/proc/meminfo").read_text().splitlines()
        if s.startswith("MemAvailable:")
    )
    if os.getloadavg()[0] > 16 or available < 8 * 1024**3:
        raise RuntimeError("host resource admission failed")
    resource.setrlimit(resource.RLIMIT_AS, (cap["AS_bytes"], cap["AS_bytes"]))
    signal.signal(
        signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("inner wall cap"))
    )
    signal.alarm(cap["inner_seconds"])
    actual = HERE / "actual"
    actual.mkdir()
    start = time.monotonic()
    report = dict(
        status="RUNNING",
        checks=[],
        phases_seconds={},
        host=dict(
            load=os.getloadavg(),
            available_bytes=available,
            affinity=sorted(os.sched_getaffinity(0)),
        ),
        scope=contract["scope"],
        counts={},
    )
    arrays = {}

    def checkpoint():
        (actual / "partial_ledger.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=json_default) + "\n"
        )

    def gate(name, value, limit=None):
        if type(value).__module__ == "numpy":
            value = value.item()
        passed = bool(value) if limit is None else bool(value <= limit)
        report["checks"].append(
            dict(name=name, observed=value, limit=limit, PASS=passed)
        )
        if not passed:
            checkpoint()
            raise RuntimeError(f"qualification failed: {name}")

    def maximum(value):
        return float(np.max(np.abs(value), initial=0))

    def act(mask, operations):
        # Independent ascending occupation-list signs, not Atomx fermion helpers.
        occupied = [p for p in range(7) if mask & (1 << p)]
        sign = 1
        for p, create in reversed(operations):
            if (p in occupied) == create:
                return None
            sign *= (-1) ** sum(q < p for q in occupied)
            if create:
                occupied.append(p)
            else:
                occupied.remove(p)
        return sum(1 << p for p in occupied), sign

    def lift(masks, terms):
        lookup = {m: i for i, m in enumerate(masks)}
        result = np.zeros((len(masks), len(masks)), complex)
        for operations, coefficient in terms:
            for col, mask in enumerate(masks):
                action = act(mask, operations)
                if action is not None:
                    target, phase = action
                    result[lookup[target], col] += phase * coefficient
        return result

    try:
        sys.path.insert(0, str(HERE / "source_snapshot"))
        sys.path.insert(0, str(HERE))
        import model_helpers as model
        import numpy as np
        from scipy.linalg import expm

        from atomx.methods import fscc
        from atomx.methods.eom_attachment import compute

        decoder = load_module("bound_tensor_decoder", contract["decoder"])
        obj, h, eri = model.integrals(spinors=7)
        gate("literal_counts", obj.noccupied == 3 and len(obj.labels) == 7)
        gate("literal_frozen_origin", obj.frozen_energy_Ha == -2.3)
        fov = maximum(obj.fock_Ha[:3, 3:])
        gate("noncanonical_Fov_preserved", fov > 1e-6)
        estimate = fscc.resource_estimate(
            obj, model="CCSDT", sector="0h1p", active_count=4
        )
        gate("tensor_prospective_work", estimate == contract["tensor_work"])
        t = time.monotonic()
        masks_c, hc = model.independent_hamiltonian(obj, h, eri, 3)
        masks_a, ha = model.independent_hamiltonian(obj, h, eri, 4)
        report["counts"]["independent_complete_H_builds"] = 2
        ec, vc = np.linalg.eigh(hc)
        efull, vfull = np.linalg.eigh(ha)
        report["counts"]["independent_complete_eigh"] = 2
        ref = masks_c.index(7)
        gate("reference_ground_branch_overlap", abs(float(vc[ref, 0])) >= 0.5)
        native = compute(
            obj,
            core_method="CCSDT",
            attachment_particle_rank=3,
            action_representation="dense",
            determinant_cap=256,
            amplitude_cap=128,
            generator_memory_MiB=128,
            logical_term_action_cap=2_000_000,
            max_evaluations=2000,
            residual_tolerance_Ha=1e-10,
            imaginary_energy_tolerance_Ha=1e-10,
        )
        report["counts"]["native_CCSDT_EOM_composite"] = 1
        report["phases_seconds"]["native_with_independent_FCI"] = time.monotonic() - t
        gate(
            "complete_core35_attachment35_R34",
            len(masks_c) == 35
            and len(masks_a) == 35
            and len(native.projected_masks) == 34,
        )
        gate(
            "native_core_FCI_energy_Ha", abs(native.core.total_energy_Ha - ec[0]), 1e-10
        )
        gate(
            "native_core_full_residual_Ha",
            native.core.full_eigenstate_residual_Ha,
            1e-10,
        )
        gate(
            "native_core_FCI_projector",
            maximum(
                np.outer(native.core.coefficients, native.core.coefficients)
                - np.outer(vc[:, 0], vc[:, 0])
            ),
            1e-10,
        )
        gate(
            "native_EOM_all_R_projected_residual_Ha",
            maximum(native.projected_residuals_Ha),
            1e-10,
        )
        native_terms, native_t3 = [], []
        for amp, (holes, particles) in zip(
            native.core.amplitudes, native.core.provenance["excitations"], strict=True
        ):
            operations = tuple((a, True) for a in particles) + tuple(
                (i, False) for i in reversed(holes)
            )
            _, sign = act(7, operations)
            term = operations, amp / sign
            native_terms.append(term)
            if len(holes) == 3:
                native_t3.append(term)
        tc = lift(masks_c, native_terms)
        ta = lift(masks_a, native_terms)
        triple = lift(masks_a, native_t3)
        t3max = maximum(triple)
        gate("genuine_nonzero_T3_action_in_Nplus1", t3max > 1e-6)
        p_rows = [masks_a.index(7 | (1 << v)) for v in range(3, 7)]
        r_rows = [masks_a.index(m) for m in native.projected_masks]
        rp = [r_rows.index(p) for p in p_rows]
        chosen = native.projected_right_vectors[:, :4]
        condition = float(np.linalg.cond(chosen[rp]))
        gate("chosen_four_model_condition", condition, 1e6)
        gate(
            "chosen_four_root_gap",
            float(native.attachment_energies_Ha[4] - native.attachment_energies_Ha[3])
            >= 1e-6,
        )
        omega_native = np.zeros((35, 4), complex)
        omega_native[r_rows] = chosen @ np.linalg.inv(chosen[rp])
        hbar_native = expm(-ta) @ ha @ expm(ta)
        heff_native = (hbar_native @ omega_native)[
            p_rows
        ] - native.core.total_energy_Ha * np.eye(4)
        arrays.update(
            one_body_Ha=obj.one_body_Ha,
            Fock_Ha=obj.fock_Ha,
            packed_eri_Ha=obj.packed_eri_Ha,
            core_masks=np.asarray(masks_c),
            attachment_masks=np.asarray(masks_a),
            full_core_H=hc,
            full_attachment_H=ha,
            core_FCI_E=ec,
            core_FCI_C=vc,
            attachment_FCI_E=efull,
            attachment_FCI_C=vfull,
            native_core_amplitudes=native.core.amplitudes,
            native_core_wave=native.core.coefficients,
            native_core_T=tc,
            native_attachment_T=ta,
            native_attachment_T3=triple,
            native_EOM_H=native.projected_operator_Ha,
            native_EOM_E=native.attachment_energies_Ha,
            native_EOM_right=native.projected_right_vectors,
            native_physical_wave=native.coefficients,
            native_projected_residuals=native.projected_residuals_Ha,
            native_full_residuals=native.full_eigenstate_residuals_Ha,
            native_four_model_omega=omega_native,
            native_four_model_Heff=heff_native,
            model_rows=np.asarray(p_rows),
            retained_R_rows=np.asarray(r_rows),
        )
        np.savez_compressed(actual / "native_complete.npz", **arrays)
        report.update(
            native_core_energy_Ha=native.core.total_energy_Ha,
            FCI_core_energy_Ha=float(ec[0]),
            native_T3_Nplus1_max=t3max,
            native_four_attachment_energies_Ha=native.attachment_energies_Ha[
                :4
            ].tolist(),
            native_core_provenance=dict(native.core.provenance),
            native_EOM_provenance=dict(native.provenance),
            tensor_work=estimate,
            Fov_max_Ha=fov,
            model_condition=condition,
            native_omitted_rank_full_H_residual_Ha=maximum(
                native.full_eigenstate_residuals_Ha
            ),
        )
        checkpoint()
        environment = dict(os.environ)
        environment["LD_LIBRARY_PATH"] = contract["runtime_library_directory"]
        t = time.monotonic()
        backend = fscc.compute(
            obj,
            executable=contract["executable"],
            directory=actual / "fscc",
            model="CCSDT",
            sector="0h1p",
            active_indices=(3, 4, 5, 6),
            tolerance=1e-13,
            residual_bound_tolerance_Ha=1e-11,
            maxiter=200,
            memory_bytes=512 * 1024**2,
            tile_size=4,
            threads=1,
            timeout_seconds=60,
            environment=environment,
            tensor_symmetry="C1",
        )
        report["counts"]["external_tensor_FSCC_launches"] = 1
        report["phases_seconds"]["external_FSCC_call"] = time.monotonic() - t
        tensors, metadata = {}, {}
        for name in ("t1c", "t2c", "t3c", "s1c", "s2c", "s3c", "hp", "ph", "pphh"):
            tensors[name], metadata[name] = decoder.decode_diagram(
                actual / "fscc/scratch" / (name + ".dg"), 7
            )
        report["counts"]["complete_tensor_diagrams_decoded"] = 9
        terms = decoder.cluster_terms([tensors[n] for n in ("t1c", "t2c", "t3c")])
        external_tc, external_ta = lift(masks_c, terms), lift(masks_a, terms)
        external_t3 = lift(
            masks_a,
            decoder.cluster_terms(
                [
                    np.zeros_like(tensors["t1c"]),
                    np.zeros_like(tensors["t2c"]),
                    tensors["t3c"],
                ]
            ),
        )
        s = lift(
            masks_a,
            decoder.cluster_terms(
                [tensors[n] for n in ("s1c", "s2c", "s3c")], attachment=True
            ),
        )
        omega = (np.eye(35) + s)[:, p_rows]
        core_hbar = expm(-external_tc) @ hc @ expm(external_tc)
        external_core_energy = core_hbar[ref, ref]
        core_residual = core_hbar[:, ref].copy()
        core_residual[ref] -= external_core_energy
        external_hbar = expm(-external_ta) @ ha @ expm(external_ta)
        target = backend.model_space_matrix_Ha
        wave_heff = (external_hbar @ omega)[p_rows] - external_core_energy * np.eye(4)
        bloch = external_hbar @ omega - omega @ (
            target + external_core_energy * np.eye(4)
        )
        full_roots, model_vectors = np.linalg.eig(target)
        order = np.argsort(full_roots.real, kind="stable")
        physical = expm(external_ta) @ omega @ model_vectors[:, order]
        physical /= np.linalg.norm(physical, axis=0)
        full_residual = ha @ physical - physical * (
            external_core_energy + full_roots[order]
        )
        loaded = fscc.load_integrals(actual / "fscc")
        metrics = dict(
            core_FCI_Ha=abs(float(external_core_energy.real) - ec[0]),
            core_exported_energy_Ha=abs(external_core_energy - backend.core_energy_Ha),
            core_complete_residual_Ha=maximum(core_residual),
            core_native_cluster_max=maximum(external_tc - tc),
            Nplus1_native_cluster_max=maximum(external_ta - ta),
            external_T3_Nplus1_max=maximum(external_t3),
            hp_Fock_decode_Ha=maximum(tensors["hp"][:3, 3:] - obj.fock_Ha[:3, 3:]),
            ph_Fock_decode_Ha=maximum(tensors["ph"][3:, :3] - obj.fock_Ha[3:, :3]),
            exported_one_body_replay_Ha=maximum(loaded.one_body_Ha - obj.one_body_Ha),
            exported_Fock_replay_Ha=maximum(loaded.fock_Ha - obj.fock_Ha),
            exported_packed_ERI_replay_Ha=maximum(
                loaded.packed_eri_Ha - obj.packed_eri_Ha
            ),
            intermediate_normalization=maximum(omega[p_rows] - np.eye(4)),
            retained_Bloch_residual_Ha=maximum(bloch[r_rows]),
            omitted_Bloch_residual_Ha=maximum(
                bloch[[i for i in range(35) if i not in r_rows]]
            ),
            full_physical_H_residual_Ha=maximum(full_residual),
            stored_Heff_minus_wave_projection_Ha=maximum(target - wave_heff),
            FSCC_minus_native_four_model_Heff_Ha=maximum(target - heff_native),
            FSCC_minus_native_four_roots_Ha=maximum(
                np.sort(full_roots.real) - native.attachment_energies_Ha[:4]
            ),
            FSCC_minus_native_model_omega=maximum(omega - omega_native),
            FSCC_vs_selected_physical_projectors=max(
                maximum(
                    np.outer(physical[:, i], physical[:, i].conj())
                    - np.outer(
                        native.coefficients[:, i], native.coefficients[:, i].conj()
                    )
                )
                for i in range(4)
            ),
            selected_roots_FCI_method_difference_Ha=maximum(
                np.sort(full_roots.real + external_core_energy.real) - efull[:4]
            ),
            backend_raw_Jacobi_bound_Ha=backend.equation_residual_bound_Ha,
        )
        report.update(
            tensor_comparison=metrics,
            tensor_decoder_metadata=metadata,
            external_provenance=dict(backend.provenance),
            FSCC_core_energy_Ha=backend.core_energy_Ha,
            FSCC_attachment_energies_Ha=np.sort(full_roots.real).tolist(),
        )
        arrays.update(
            external_core_T=external_tc,
            external_attachment_T=external_ta,
            external_attachment_T3=external_t3,
            external_core_hbar=core_hbar,
            external_core_residual=core_residual,
            external_attachment_hbar=external_hbar,
            external_model_omega=omega,
            external_model_Heff=target,
            external_wave_Heff=wave_heff,
            external_complete_Bloch=bloch,
            external_physical_wave=physical,
            external_full_H_residual=full_residual,
            **tensors,
        )
        np.savez_compressed(actual / "complete_comparison.npz", **arrays)
        checkpoint()
        for name in (
            "core_FCI_Ha",
            "core_exported_energy_Ha",
            "core_complete_residual_Ha",
            "core_native_cluster_max",
            "Nplus1_native_cluster_max",
            "stored_Heff_minus_wave_projection_Ha",
            "retained_Bloch_residual_Ha",
        ):
            gate("independent_tensor:" + name, float(metrics[name]), 1e-10)
        for name in (
            "hp_Fock_decode_Ha",
            "ph_Fock_decode_Ha",
            "exported_one_body_replay_Ha",
            "exported_Fock_replay_Ha",
            "exported_packed_ERI_replay_Ha",
            "intermediate_normalization",
        ):
            gate("literal_input_and_metric:" + name, float(metrics[name]), 1e-12)
        gate("external_genuine_T3_action", metrics["external_T3_Nplus1_max"] > 1e-6)
        # These are measured conditional matches after the independent retained
        # equations/normalization above; no general FSCC/EOM identity is assumed.
        for name in (
            "FSCC_minus_native_four_model_Heff_Ha",
            "FSCC_minus_native_four_roots_Ha",
            "FSCC_minus_native_model_omega",
            "FSCC_vs_selected_physical_projectors",
        ):
            gate("finite_conditional_match:" + name, float(metrics[name]), 1e-10)
        report["loaded_native_sources"] = {
            str(Path(module.__file__).resolve()): sha(module.__file__)
            for name, module in tuple(sys.modules.items())
            if name.startswith("atomx")
            and getattr(module, "__file__", None)
            and Path(module.__file__).suffix == ".py"
        }
        gate(
            "all_native_modules_from_private_snapshot",
            all(
                Path(p).is_relative_to(HERE / "source_snapshot")
                for p in report["loaded_native_sources"]
            ),
        )
        source(contract)
        report["output_files"] = {
            str(p.relative_to(actual)): sha(p)
            for p in sorted(actual.rglob("*"))
            if p.is_file()
        }
        report["status"] = "PASS_finite_N3_tensor_comparison_only"
    except BaseException as error:
        report.update(
            status="FAILED_preserved",
            failure=dict(type=type(error).__name__, message=str(error)),
        )
        raise
    finally:
        signal.alarm(0)
        report.update(
            elapsed_seconds=time.monotonic() - start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            run_sha256=sha(__file__),
            contract_sha256=sha(HERE / "contract.json"),
        )
        (actual / "result.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=json_default) + "\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    contract = json.loads((HERE / "contract.json").read_text())
    admission = source(contract)
    if args.execute:
        execute(contract)
    else:
        (HERE / "preparation.json").write_text(
            json.dumps(admission, indent=2, sort_keys=True) + "\n"
        )
