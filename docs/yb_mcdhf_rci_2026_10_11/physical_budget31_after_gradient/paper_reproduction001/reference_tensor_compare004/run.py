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
        raise RuntimeError("closed evidence directory; no retry")
    cap = contract["resources"]
    if any(os.environ.get(t) != "1" for t in THREADS) or os.sched_getaffinity(0) != {
        cap["CPU"]
    }:
        raise RuntimeError("CPU/threads admission")
    available = next(
        int(s.split()[1]) * 1024
        for s in Path("/proc/meminfo").read_text().splitlines()
        if s.startswith("MemAvailable:")
    )
    if os.getloadavg()[0] > 16 or available < 8 * 1024**3:
        raise RuntimeError("host resource admission")
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
        counts={"native_complex_CC_calls": 0},
        scope=contract["scope"],
        phases_seconds={},
        host=dict(
            load=os.getloadavg(),
            available_bytes=available,
            affinity=sorted(os.sched_getaffinity(0)),
        ),
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
            raise RuntimeError("qualification failed: " + name)

    def maximum(value):
        return float(np.max(np.abs(value), initial=0))

    def action(mask, operations):
        occupied = [i for i in range(7) if mask & (1 << i)]
        phase = 1
        for i, create in reversed(operations):
            if (i in occupied) == create:
                return None
            phase *= (-1) ** sum(v < i for v in occupied)
            if create:
                occupied.append(i)
            else:
                occupied.remove(i)
        return sum(1 << i for i in occupied), phase

    def lift(masks, terms):
        index = {mask: i for i, mask in enumerate(masks)}
        result = np.zeros((len(masks), len(masks)), complex)
        for operations, value in terms:
            for col, mask in enumerate(masks):
                item = action(mask, operations)
                if item is not None:
                    target, sign = item
                    result[index[target], col] += value * sign
        return result

    def matrix(electrons, h, eri):
        # Full ordered 1/4 two-body strings, independent occupation-list signs.
        from itertools import combinations, product

        masks = tuple(
            sum(1 << i for i in row) for row in combinations(range(7), electrons)
        )
        terms = [
            (h[p, q], ((p, True), (q, False))) for p, q in product(range(7), repeat=2)
        ]
        terms += [
            (0.25 * eri(p, q, r, s), ((p, True), (q, True), (s, False), (r, False)))
            for p, q, r, s in product(range(7), repeat=4)
        ]
        result = np.eye(len(masks), dtype=complex) * (-2.3)
        result += lift(masks, [(ops, value) for value, ops in terms if value != 0])
        return masks, result

    try:
        sys.path.insert(0, str(HERE / "source_snapshot"))
        sys.path.insert(0, str(HERE))
        import model_helpers as model
        import numpy as np
        from scipy.linalg import expm

        from atomx.methods import fscc
        from atomx.physics.csf.spinor_integrals import SpinorIntegrals

        decoder = load_module("bound_tensor_decoder", contract["decoder"])
        base, h0, eri0 = model.integrals(spinors=7)
        with np.load(contract["baseline_arrays"], allow_pickle=False) as archive:
            baseline = {key: archive[key] for key in archive.files}
        report["counts"]["complete_baseline_archive_decoded"] = 1
        gate(
            "baseline_all_decoded_bytes",
            sum(v.nbytes for v in baseline.values()),
            64 * 1024**2,
        )
        gate(
            "baseline_literal_one_body",
            maximum(base.one_body_Ha - baseline["one_body_Ha"]),
            0.0,
        )
        gate("baseline_literal_Fock", maximum(base.fock_Ha - baseline["Fock_Ha"]), 0.0)
        gate(
            "baseline_literal_complete_packed_ERI",
            maximum(base.packed_eri_Ha - baseline["packed_eri_Ha"]),
            0.0,
        )
        phases = np.exp(1j * np.pi * np.asarray(contract["phase_fractions_pi"]))
        gate("fixed_phase_unitarity", maximum(abs(phases) ** 2 - 1), 1e-14)

        def eri(p, q, r, s):
            return (
                phases[p].conjugate()
                * phases[q].conjugate()
                * eri0(p, q, r, s)
                * phases[r]
                * phases[s]
            )

        def phased_Hermitian(value):
            output = np.empty(value.shape, complex)
            for p in range(7):
                output[p, p] = value[p, p]
                for q in range(p + 1, 7):
                    output[p, q] = phases[p].conjugate() * value[p, q] * phases[q]
                    output[q, p] = output[p, q].conjugate()
            return output

        h = phased_Hermitian(h0)
        fock = phased_Hermitian(base.fock_Ha)
        gate(
            "full_onebody_phase_covariance_after_ABI_storage",
            maximum(h - phases.conj()[:, None] * h0 * phases[None, :]),
            1e-12,
        )
        gate(
            "full_Fock_phase_covariance_after_ABI_storage",
            maximum(fock - phases.conj()[:, None] * base.fock_Ha * phases[None, :]),
            1e-12,
        )
        packed = np.asarray(
            [
                eri(p, q, r, s)
                for p in range(6)
                for q in range(p + 1, 7)
                for r in range(p, 6)
                for s in range(r + 1, 7)
            ]
        )
        obj = SpinorIntegrals(
            base.labels,
            3,
            h,
            fock,
            packed,
            -2.3,
            base.reference_energy_Ha,
            dict(
                base.identity,
                phase_case="fixed orbital basis rephasing; same complete toy operator",
            ),
        )
        gate(
            "noncanonical_Fov_magnitude_preserved",
            abs(maximum(obj.fock_Ha[:3, 3:]) - maximum(base.fock_Ha[:3, 3:])),
            1e-14,
        )
        gate("complex_operator_distinguishes_conjugation", maximum(h.imag) > 1e-3)
        mc, hc = matrix(3, h, eri)
        ma, ha = matrix(4, h, eri)
        gate(
            "complete_core_and_attachment_masks",
            tuple(baseline["core_masks"]) == mc
            and tuple(baseline["attachment_masks"]) == ma,
        )
        dc = np.asarray(
            [np.prod([phases[i] for i in range(7) if mask & (1 << i)]) for mask in mc]
        )
        da = np.asarray(
            [np.prod([phases[i] for i in range(7) if mask & (1 << i)]) for mask in ma]
        )
        p_rows = baseline["model_rows"]
        r_rows = baseline["retained_R_rows"]
        dm = da[p_rows]
        hc_expected = dc.conj()[:, None] * baseline["full_core_H"] * dc[None, :]
        ha_expected = da.conj()[:, None] * baseline["full_attachment_H"] * da[None, :]
        gate(
            "independent_complete_complex_core_H_covariance",
            maximum(hc - hc_expected),
            1e-12,
        )
        gate(
            "independent_complete_complex_attachment_H_covariance",
            maximum(ha - ha_expected),
            1e-12,
        )
        gate("complete_core_H_hermiticity", maximum(hc - hc.conj().T), 1e-12)
        gate("complete_attachment_H_hermiticity", maximum(ha - ha.conj().T), 1e-12)
        ec, vc = np.linalg.eigh(hc)
        ea, va = np.linalg.eigh(ha)
        gate(
            "all_complex_core_FCI_roots_covariant",
            maximum(ec - baseline["core_FCI_E"]),
            1e-10,
        )
        gate(
            "all_complex_attachment_FCI_roots_covariant",
            maximum(ea - baseline["attachment_FCI_E"]),
            1e-10,
        )
        report["counts"].update(
            independent_complete_complex_H=2, independent_complete_complex_FCI=2
        )
        ref = mc.index(7)
        gate("unchanged_reference_ground_overlap", abs(vc[ref, 0]) >= 0.5)
        tc_expected = dc.conj()[:, None] * baseline["native_core_T"] * dc[None, :]
        ta_expected = da.conj()[:, None] * baseline["native_attachment_T"] * da[None, :]
        t3_expected = (
            da.conj()[:, None] * baseline["native_attachment_T3"] * da[None, :]
        )
        omega_expected = (
            da.conj()[:, None] * baseline["native_four_model_omega"] * dm[None, :]
        )
        heff_expected = (
            dm.conj()[:, None] * baseline["native_four_model_Heff"] * dm[None, :]
        )
        wave_expected = da.conj()[:, None] * baseline["native_physical_wave"][:, :4]
        gate("same_nonzero_T3_Nplus1", maximum(t3_expected) > 1e-6)
        rp = [list(r_rows).index(p) for p in p_rows]
        original_chosen = baseline["native_EOM_right"][:, :4]
        phased_chosen = da[r_rows].conj()[:, None] * original_chosen
        original_condition = float(np.linalg.cond(original_chosen[rp]))
        phased_condition = float(np.linalg.cond(phased_chosen[rp]))
        gate(
            "same_four_model_condition",
            phased_condition,
            1e6,
        )
        gate(
            "four_model_projector_condition_covariant",
            abs(phased_condition - original_condition),
            1e-12,
        )
        gate(
            "same_four_root_gap",
            float(baseline["native_EOM_E"][4] - baseline["native_EOM_E"][3]) >= 1e-6,
        )
        arrays.update(
            phases=phases,
            core_determinant_phases=dc,
            attachment_determinant_phases=da,
            model_determinant_phases=dm,
            one_body_Ha=h,
            Fock_Ha=fock,
            packed_eri_Ha=packed,
            full_core_H=hc,
            full_attachment_H=ha,
            core_FCI_E=ec,
            core_FCI_C=vc,
            attachment_FCI_E=ea,
            attachment_FCI_C=va,
            expected_core_T=tc_expected,
            expected_attachment_T=ta_expected,
            expected_attachment_T3=t3_expected,
            expected_model_omega=omega_expected,
            expected_ket_Heff=heff_expected,
            expected_physical_wave=wave_expected,
            core_masks=np.asarray(mc),
            attachment_masks=np.asarray(ma),
            model_rows=p_rows,
            retained_R_rows=r_rows,
        )
        np.savez_compressed(actual / "phase_operator_reference.npz", **arrays)
        report.update(
            common_model_reference_condition=phased_condition,
            original_saved_model_projector_condition=original_condition,
            independent_core_energy_Ha=float(ec[0]),
            frozen_roots_Ha=baseline["native_EOM_E"][:4].tolist(),
        )
        checkpoint()
        environment = dict(os.environ)
        environment["LD_LIBRARY_PATH"] = contract["runtime_library_directory"]
        t = time.monotonic()
        external = fscc.compute(
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
        report["counts"]["external_complex_tensor_launches"] = 1
        report["phases_seconds"]["external_FSCC_call"] = time.monotonic() - t
        tensors, metadata = {}, {}
        for name in ("t1c", "t2c", "t3c", "s1c", "s2c", "s3c", "hp", "ph", "pphh"):
            tensors[name], metadata[name] = decoder.decode_diagram(
                actual / "fscc/scratch" / (name + ".dg"), 7
            )
        terms = decoder.cluster_terms([tensors[n] for n in ("t1c", "t2c", "t3c")])
        tc, ta = lift(mc, terms), lift(ma, terms)
        t3 = lift(
            ma,
            decoder.cluster_terms(
                [
                    np.zeros_like(tensors["t1c"]),
                    np.zeros_like(tensors["t2c"]),
                    tensors["t3c"],
                ]
            ),
        )
        sop = lift(
            ma,
            decoder.cluster_terms(
                [tensors[n] for n in ("s1c", "s2c", "s3c")], attachment=True
            ),
        )
        omega = (np.eye(35) + sop)[:, p_rows]
        core_hbar = expm(-tc) @ hc @ expm(tc)
        core_energy = core_hbar[ref, ref]
        core_residual = core_hbar[:, ref].copy()
        core_residual[ref] -= core_energy
        hbar = expm(-ta) @ ha @ expm(ta)
        wave_heff = (hbar @ omega)[p_rows] - core_energy * np.eye(4)
        raw = external.model_space_matrix_Ha
        orientations = {
            "raw": raw,
            "transpose": raw.T,
            "conjugate": raw.conj(),
            "adjoint": raw.conj().T,
        }
        diagnostics = {}
        physical_by_map = {}
        for name, mapped in orientations.items():
            bloch = hbar @ omega - omega @ (mapped + core_energy * np.eye(4))
            values, columns = np.linalg.eig(mapped)
            order = np.argsort(values.real, kind="stable")
            values, columns = values[order], columns[:, order]
            psi = expm(ta) @ omega @ columns
            psi /= np.linalg.norm(psi, axis=0)
            full = ha @ psi - psi * (core_energy + values)
            physical_by_map[name] = psi
            diagnostics[name] = dict(
                mapped_minus_independent_expected_ket_Heff=maximum(
                    mapped - heff_expected
                ),
                mapped_minus_actual_wave_ket_Heff=maximum(mapped - wave_heff),
                retained_Bloch_Ha=maximum(bloch[r_rows]),
                omitted_Bloch_Ha=maximum(
                    bloch[[i for i in range(35) if i not in r_rows]]
                ),
                four_roots_minus_fixed_native_Ha=maximum(
                    values.real - baseline["native_EOM_E"][:4]
                ),
                imaginary_roots_Ha=maximum(values.imag),
                physical_full_H_residual_Ha=maximum(full),
                physical_projector_covariance=max(
                    maximum(
                        np.outer(psi[:, i], psi[:, i].conj())
                        - np.outer(wave_expected[:, i], wave_expected[:, i].conj())
                    )
                    for i in range(4)
                ),
            )
            arrays["orientation_" + name + "_Bloch"] = bloch
        # Source LAPACK_COL_MAJOR treats raw row-major matrix as raw.T. Test the
        # model-vector conjugation independently when converting to ket Ω.
        values, cbackend = np.linalg.eig(raw.T)
        order = np.argsort(values.real, kind="stable")
        cbackend = cbackend[:, order]
        psi_conjugated = expm(ta) @ omega @ cbackend.conj()
        psi_conjugated /= np.linalg.norm(psi_conjugated, axis=0)
        conjugated_coeff_projector = max(
            maximum(
                np.outer(psi_conjugated[:, i], psi_conjugated[:, i].conj())
                - np.outer(wave_expected[:, i], wave_expected[:, i].conj())
            )
            for i in range(4)
        )
        metrics = dict(
            core_complete_residual_Ha=maximum(core_residual),
            core_energy_FCI_Ha=abs(core_energy - ec[0]),
            core_tensor_phase_covariance=maximum(tc - tc_expected),
            Nplus1_tensor_phase_covariance=maximum(ta - ta_expected),
            T3_Nplus1_phase_covariance=maximum(t3 - t3_expected),
            T3_Nplus1_max=maximum(t3),
            model_omega_phase_covariance=maximum(omega - omega_expected),
            model_omega_normalization=maximum(omega[p_rows] - np.eye(4)),
            hp_Fock_decode_Ha=maximum(tensors["hp"][:3, 3:] - fock[:3, 3:]),
            ph_Fock_decode_Ha=maximum(tensors["ph"][3:, :3] - fock[3:, :3]),
            raw_nonHermitian_max=maximum(raw - raw.conj().T),
            raw_imaginary_max=maximum(raw.imag),
            backend_column_major_coefficients_conjugated_physical_projector=conjugated_coeff_projector,
            ket_wave_Heff_phase_covariance=maximum(wave_heff - heff_expected),
        )
        report.update(
            tensor_phase_metrics=metrics,
            all_orientation_diagnostics=diagnostics,
            tensor_decoder_metadata=metadata,
            external_provenance=dict(external.provenance),
        )
        arrays.update(
            external_core_T=tc,
            external_attachment_T=ta,
            external_attachment_T3=t3,
            external_model_omega=omega,
            external_raw_Heff=raw,
            external_wave_ket_Heff=wave_heff,
            external_core_residual=core_residual,
            external_core_hbar=core_hbar,
            external_attachment_hbar=hbar,
            external_backend_conjugated_coeff_wave=psi_conjugated,
            **{("physical_" + n): v for n, v in physical_by_map.items()},
            **tensors,
        )
        np.savez_compressed(actual / "complete_phase_comparison.npz", **arrays)
        checkpoint()
        for name in (
            "core_complete_residual_Ha",
            "core_energy_FCI_Ha",
            "core_tensor_phase_covariance",
            "Nplus1_tensor_phase_covariance",
            "T3_Nplus1_phase_covariance",
            "model_omega_phase_covariance",
            "ket_wave_Heff_phase_covariance",
        ):
            gate("complex_tensor:" + name, float(metrics[name]), 1e-10)
        for name in (
            "model_omega_normalization",
            "hp_Fock_decode_Ha",
            "ph_Fock_decode_Ha",
        ):
            gate("complex_input:" + name, float(metrics[name]), 1e-12)
        gate("genuine_T3_not_Pauli_blocked", metrics["T3_Nplus1_max"] > 1e-6)
        winners = [
            name
            for name, data in diagnostics.items()
            if max(
                data["mapped_minus_independent_expected_ket_Heff"],
                data["mapped_minus_actual_wave_ket_Heff"],
                data["retained_Bloch_Ha"],
                data["physical_projector_covariance"],
            )
            <= 1e-10
        ]
        gate("unique_complete_matrix_wave_Bloch_orientation", len(winners) == 1)
        winner = winners[0]
        gate(
            "other_complete_orientation_maps_distinct",
            min(
                data["mapped_minus_actual_wave_ket_Heff"]
                for name, data in diagnostics.items()
                if name != winner
            )
            >= 1e-6,
        )
        report["qualified_orientation"] = winner
        if winner == "adjoint":
            gate(
                "qualified_adjoint_requires_conjugated_backend_model_coefficients",
                conjugated_coeff_projector,
                1e-10,
            )
            report["qualified_backend_model_coefficient_conversion"] = (
                "conjugate raw.T right coefficients before ket Omega lift"
            )
        else:
            report["backend_model_coefficient_conjugation_scope"] = (
                "diagnostic only; adjoint-specific conversion not qualified"
            )
        report["loaded_native_sources"] = {
            str(Path(m.__file__).resolve()): sha(m.__file__)
            for n, m in tuple(sys.modules.items())
            if n.startswith("atomx")
            and getattr(m, "__file__", None)
            and Path(m.__file__).suffix == ".py"
        }
        gate(
            "native_import_origin_is_private_snapshot",
            all(
                Path(p).is_relative_to(HERE / "source_snapshot")
                for p in report["loaded_native_sources"]
            ),
        )
        source(contract)
        report["status"] = "PASS_same_toy_complex_matrix_convention_discriminator_only"
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
            driver_sha256=sha(__file__),
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
