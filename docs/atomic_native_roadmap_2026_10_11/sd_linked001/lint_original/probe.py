"""Bounded native complete scalar linked SD, fixed pre-existing Hamiltonians.

No external SD/CC executable or library is loaded. The independent oracle
uses elementary Fock operations on the complete fixed-N space.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import resource
import signal
import time
import traceback


OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[2]
INPUT = ROOT / "docs/atomic_pci_sd_native_2026_10_09/parallel_six_sd_diagrams001/inputs.npz"
MODES = ("dc", "static", "dynamic")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_bindings():
    # Source-only traversal excludes archived run trees and compiler caches.
    paths = sorted((ROOT / "src/atomx").rglob("*.py"))
    paths += [Path(__file__).resolve()]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def preparation():
    import numpy as np
    from atomx.methods.linearized_sd import plan_scalar_sd

    path = OWNER / "contract.json"
    if path.exists():
        raise FileExistsError("premeasurement scientific contract is immutable")
    data = np.load(INPUT, allow_pickle=False)
    actual = {}
    for mode in MODES:
        required = [f"{mode}_{name}" for name in (
            "labels", "constants", "one_body_Ha", "fock_Ha", "packed_eri_Ha")]
        if any(name not in data.files for name in required):
            raise ValueError(f"complete original {mode} payload absent")
        labels = tuple(tuple(map(int, row)) for row in data[f"{mode}_labels"])
        occupied = int(data[f"{mode}_constants"][0])
        occupied_keys = tuple(dict.fromkeys(row[:2] for row in labels[:occupied]))
        virtual_keys = tuple(dict.fromkeys(row[:2] for row in labels[occupied:]))
        plan = plan_scalar_sd(occupied_keys, virtual_keys)
        actual[mode] = dict(spinors=len(labels), electrons=occupied,
            determinants=math.comb(len(labels), occupied), occupied_keys=occupied_keys,
            virtual_keys=virtual_keys, scalar_plan=plan.resources(),
            one_body_shape=list(data[f"{mode}_one_body_Ha"].shape),
            packed_eri_elements=int(data[f"{mode}_packed_eri_Ha"].size))
        if (len(labels), occupied, actual[mode]["determinants"], plan.singles,
                plan.doubles) != (12, 4, 495, 2, 10):
            raise ValueError("fixed scientific model differs from authorized counts")
        if any(np.any(data[name].imag) for name in required if
                data[name].dtype.kind == "c"):
            raise ValueError("authorized DC/static/dynamic payload must be real")
    # Deliberately conservative numeric scratch and Python dictionary budget.
    numeric_bound = 8 * (8 * 495**2 + 20 * 495 * 13 + 20 * 13**2)
    python_bound = 64 * 1024**2
    contract = dict(status="predeclared", hypothesis=
        "The native complete Q(H+[H,T1+T2]) scalar equations agree with independent full Fock action; existing source+PP+HH is a strict incomplete subgraph even at this fixed space.",
        fixed_conditions="Original immutable 4-electron/12-spinor payload; every scalar T1/T2 and all fixed-N determinants retained; no orbital reconstruction, physical truncation, fit, source change or external solver.",
        actual=actual, modes=list(MODES), input_path=str(INPUT.relative_to(ROOT)),
        input_sha256=digest(INPUT), sources=source_bindings(),
        resources=dict(blas_threads=1, omp_threads=1, numba_threads=1,
            address_space_bytes=3 * 1024**3, cpu_seconds=300,
            single_case_wall_seconds=120, total_wall_seconds=300,
            numerical_scratch_bound_bytes=numeric_bound,
            additional_python_payload_allowance_bytes=python_bound,
            scope="Conservative experiment allocation bound, not measured RSS; source copying excluded from science time."),
        gates=dict(signed_source_matrix_energy_Ha=1e-11, scalar_metric=1e-12,
            solved_full_projected_residual_Ha=1e-12,
            producer_hermitian_roundoff_Ha=1e-11,
            real_roundoff_imaginary_Ha=1e-14),
        distinguishing_outcomes=dict(
            correctness="Every signed affine source/K column and linked energy passes independent Fermi action; root passes full retained equation.",
            missing_physics="Quantify full K minus PP/HH, and residual after including independently identified full Fock action. Report actual nonzero Fov*T2 without inventing a shift.",
            limitations="Finite native equation discriminator; no production heavy-atom solver, triples, new NIST roots or performance speedup."),
        stopping_condition="Any unchanged gate, source/input identity, 3 GiB, per-case 120 s or total 300 s failure is preserved and stops execution; no model shrink or tolerance relaxation.")
    write_json(path, contract)
    print(json.dumps(dict(contract=str(path), input_sha256=contract["input_sha256"],
                         actual=actual, resources=contract["resources"])))


def elementary(mask, operations):
    """Independent rightmost-first sign; never calls Atomx _ops_sign."""
    phase = 1
    for index, creation in reversed(operations):
        bit = 1 << index
        if bool(mask & bit) == creation:
            return None, 0
        phase *= (-1) ** (mask & (bit - 1)).bit_count()
        mask ^= bit
    return mask, phase


def action(vector, terms):
    out = {}
    for operations, coefficient in terms:
        for mask, amplitude in vector.items():
            target, phase = elementary(mask, operations)
            if phase:
                out[target] = out.get(target, 0.0) + phase * coefficient * amplitude
    return out


def dense(vector, masks, np):
    return np.asarray([vector.get(mask, 0.0) for mask in masks], dtype=np.complex128)


def check(label, value, limit):
    if not math.isfinite(value) or value > limit:
        raise AssertionError(f"{label}: {value} > {limit}")


def terms_matrix(terms, masks, np):
    lookup = {mask: i for i, mask in enumerate(masks)}
    h = np.zeros((len(masks), len(masks)), dtype=np.complex128)
    for column, mask in enumerate(masks):
        for operations, coefficient in terms:
            target, phase = elementary(mask, operations)
            if phase:
                h[lookup[target], column] += phase * coefficient
    return h


def run_mode(data, mode, contract, arrays):
    import numpy as np
    from atomx.angular.recoupling import twoj_of_kappa
    from atomx.methods.linearized_sd import solve
    from atomx.methods.sd_pair_diagrams import CorePairLadders
    from atomx.methods.sd_transverse import build_j_coupled_sd_interaction
    from atomx.physics.csf.spinor_integrals import SpinorIntegrals

    started = time.monotonic()
    occupied, frozen, reference_energy = data[f"{mode}_constants"]
    labels = tuple(tuple(map(int, row)) for row in data[f"{mode}_labels"])
    a = SpinorIntegrals(labels, int(occupied), data[f"{mode}_one_body_Ha"],
        data[f"{mode}_fock_Ha"], data[f"{mode}_packed_eri_Ha"], float(frozen),
        float(reference_energy), json.loads(str(data["identities_json"]))[mode])
    n, o, ref = len(labels), a.noccupied, (1 << a.noccupied) - 1
    masks = tuple(sum(1 << i for i in row) for row in itertools.combinations(range(n), o))
    ref_vector = {ref: 1.0}
    bridge = build_j_coupled_sd_interaction(a)
    ladders = CorePairLadders(bridge)
    excitations, descriptions = [], []
    # Complete scalar T1, normalized common-j magnetic sums.
    occupied_keys = tuple(dict.fromkeys(row[:2] for row in labels[:o]))
    virtual_keys = tuple(dict.fromkeys(row[:2] for row in labels[o:]))
    for occupied_key in occupied_keys:
        for virtual_key in virtual_keys:
            if occupied_key[1] == virtual_key[1]:
                terms = []
                for hole in range(o):
                    for particle in range(o, n):
                        if (labels[hole][:2] == occupied_key and
                                labels[particle][:2] == virtual_key and
                                labels[particle][2] == labels[hole][2]):
                            terms.append((((particle, True), (hole, False)),
                                1.0 / math.sqrt(twoj_of_kappa(occupied_key[1]) + 1)))
                excitations.append(tuple(terms))
                descriptions.append(dict(rank=1, occupied=occupied_key, virtual=virtual_key))
    ns = len(excitations)
    for (twoj, parity), virtual_pair, occupied_pair in ladders.coordinates:
        rows = bridge.channels[twoj, parity]
        v, h = rows.index(virtual_pair), rows.index(occupied_pair)
        terms = []
        for twom in range(-twoj, twoj + 1, 2):
            vectors = bridge.pair_vectors(twoj, parity, twom=twom)
            for p, q, cv in vectors[v]:
                for r, s, co in vectors[h]:
                    terms.append((((p, True), (q, True), (s, False), (r, False)),
                        cv * co / math.sqrt(twoj + 1)))
        excitations.append(tuple(terms))
        descriptions.append(dict(rank=2, twoj=twoj, parity=parity,
                                 occupied=occupied_pair, virtual=virtual_pair))
    columns = [dense(action(ref_vector, terms), masks, np) for terms in excitations]
    c = np.column_stack([dense(ref_vector, masks, np)] + columns)
    metric_error = float(np.max(abs(c.conj().T @ c - np.eye(13))))
    check("complete scalar metric", metric_error, contract["gates"]["scalar_metric"])
    if c.shape != (495, 13) or ns != 2 or len(excitations) != 12:
        raise AssertionError("whole 2 T1 + 10 T2 scalar space lost")
    pairs = tuple(itertools.combinations(range(n), 2))
    h_terms = [((), a.frozen_energy_Ha)]
    h_terms += [(((p, True), (q, False)), a.one_body_Ha[p, q])
                for p, q in itertools.product(range(n), repeat=2)
                if a.one_body_Ha[p, q] != 0]
    h_terms += [(((p, True), (q, True), (s, False), (r, False)), value)
                for p, q in pairs for r, s in pairs
                if (value := a.eri(p, q, r, s)) != 0]
    h_raw = terms_matrix(h_terms, masks, np)
    hermitian_error = float(np.max(abs(h_raw - h_raw.conj().T)))
    imaginary_error = float(np.max(abs(h_raw.imag)))
    check("full producer Hermiticity", hermitian_error,
          contract["gates"]["producer_hermitian_roundoff_Ha"])
    check("full producer real input", imaginary_error,
          contract["gates"]["real_roundoff_imaginary_Ha"])
    # Explicit bounded producer roundoff projection, not a changed physical model.
    h = (h_raw.real + h_raw.real.T) / 2
    hc = c.real.T @ h @ c.real
    hc_roundoff = float(np.max(abs(hc - hc.T)))
    check("projected producer symmetry", hc_roundoff,
          contract["gates"]["producer_hermitian_roundoff_Ha"])
    hc = (hc + hc.T) / 2
    vectors = tuple(tuple((mask, float(c[row, column].real))
                    for row, mask in enumerate(masks) if c[row, column] != 0)
                    for column in range(13))
    public = solve(hc, vectors, ref, reference_identity=f"fixed-native-{mode}-4electron-full-scalar-SD",
        residual_tolerance_Ha=contract["gates"]["solved_full_projected_residual_Ha"],
        memory_bytes=32 * 1024**2)
    # Independent full Fock action of H T_mu - T_mu H on the actual reference.
    h_reference = action(ref_vector, h_terms)
    source = c[:, 1:].conj().T @ dense(h_reference, masks, np)
    k_columns = []
    for terms, column in zip(excitations, columns, strict=True):
        state = {mask: column[row] for row, mask in enumerate(masks) if column[row] != 0}
        ht = dense(action(state, h_terms), masks, np)
        th = dense(action(h_reference, terms), masks, np)
        k_columns.append(c[:, 1:].conj().T @ (ht - th))
    k = np.column_stack(k_columns)
    energy_row = c[:, 0].conj().T @ h_raw @ c[:, 1:]
    source_error = float(np.max(abs(source - hc[1:, 0])))
    k_error = float(np.max(abs(k - public.equation_matrix_Ha)))
    energy_error = float(abs(energy_row @ public.amplitudes - public.correlation_energy_Ha))
    raw_residual = float(np.max(abs(source + k @ public.amplitudes)))
    doubled_source_error = float(np.max(abs(source[ns:] - ladders.source_Ha)))
    pp = np.column_stack([ladders.evaluate(t).particle_particle_Ha
                           for t in np.eye(10)])
    hh = np.column_stack([ladders.evaluate(t).hole_hole_Ha for t in np.eye(10)])
    fock_terms = [((), -np.trace(a.fock_Ha[:o, :o]))]
    fock_terms += [(((p, True), (q, False)), a.fock_Ha[p, q])
                  for p, q in itertools.product(range(n), repeat=2)
                  if a.fock_Ha[p, q] != 0]
    f = terms_matrix(fock_terms, masks, np)
    fock_k = c[:, 1:].conj().T @ f @ c[:, 1:]
    missing_dd = k[ns:, ns:] - pp - hh
    missing_after_fock = missing_dd - fock_k[ns:, ns:]
    offdiag_missing = missing_after_fock.copy()
    np.fill_diagonal(offdiag_missing, 0)
    fov_terms = [(((p, True), (q, False)), a.fock_Ha[p, q])
                 for p in range(o) for q in range(o, n) if a.fock_Ha[p, q] != 0]
    fov = terms_matrix(fov_terms, masks, np)
    fov_t2 = c[:, 1:1+ns].conj().T @ fov @ c[:, 1+ns:]
    # PP/HH plus known Fock is a coverage control, never a substitute SD solve.
    incomplete = fock_k.copy()
    incomplete[ns:, ns:] += pp + hh
    incomplete_root_residual = float(np.max(abs(source + incomplete @ public.amplitudes)))
    gates = dict(scalar_metric_error=metric_error, complete_source_error_Ha=source_error,
        complete_K_all_12_columns_error_Ha=k_error, linked_energy_error_Ha=energy_error,
        independent_full_projected_root_residual_Ha=raw_residual,
        shared_solver_residual_Ha=public.equation_residual_Ha,
        doubles_source_vs_existing_native_error_Ha=doubled_source_error,
        reference_energy_identity_error_Ha=float(abs(h_raw[masks.index(ref), masks.index(ref)] - a.reference_energy_Ha)))
    for name, value in gates.items():
        if name.endswith("Ha"):
            limit = (contract["gates"]["solved_full_projected_residual_Ha"]
                if name in ("independent_full_projected_root_residual_Ha", "shared_solver_residual_Ha")
                else contract["gates"]["signed_source_matrix_energy_Ha"])
            check(name, value, limit)
    for name, value in dict(H_raw=h_raw, scalar_vectors=c, source=source, K=k,
        shared_K=public.equation_matrix_Ha, amplitudes=public.amplitudes,
        energy_row=energy_row, pp=pp, hh=hh, fock_K=fock_k,
        missing_after_Fock_PP_HH=missing_after_fock, Fov_T2=fov_t2,
        shared_singles_subtraction=public.singles_product_subtraction_Ha).items():
        arrays[f"{mode}_{name}"] = value
    return dict(status="passed", gates=gates, interaction_identity=dict(a.identity),
        scalar_descriptions=descriptions, counts=dict(spinors=n, electrons=o,
            determinants=len(masks), singles=ns, doubles=10, scalar_amplitudes=12),
        total_energy_Ha=public.total_energy_Ha,
        correlation_energy_Ha=public.correlation_energy_Ha,
        numerical_producer_projection=dict(full_H_hermitian_error_Ha=hermitian_error,
            full_H_roundoff_projection_max_Ha=hermitian_error/2,
            scalar_H_roundoff_projection_max_Ha=hc_roundoff/2),
        coverage_discriminator=dict(
            core_virtual_Fock_max_Ha=float(np.max(abs(a.fock_Ha[:o, o:]))),
            actual_Fov_T2_to_T1_max_Ha=float(np.max(abs(fov_t2))),
            actual_noncanonical_T1_product_subtraction_max_Ha=float(np.max(abs(public.singles_product_subtraction_Ha))),
            missing_T1_source_Ha=float(np.max(abs(source[:ns]))),
            full_DD_minus_existing_PP_HH_max_Ha=float(np.max(abs(missing_dd))),
            missing_DD_after_full_Fock_PP_HH_max_Ha=float(np.max(abs(missing_after_fock))),
            missing_offdiagonal_DD_after_full_Fock_PP_HH_max_Ha=float(np.max(abs(offdiag_missing))),
            missing_all_blocks_after_full_Fock_PP_HH_max_Ha=float(np.max(abs(k-incomplete))),
            full_root_residual_if_only_Fock_PP_HH_Ha=incomplete_root_residual,
            interpretation="Measured omitted complete linked SD action, including mixed/ring terms. No claim each omitted entry is uniquely a named diagram; no incomplete-map root substitutes the accepted root."),
        elapsed_wall_seconds=time.monotonic()-started)


def timeout_handler(signum, frame):
    raise TimeoutError("predeclared case or total wall-time limit")


def execute():
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        if os.environ.get(variable) != "1":
            raise RuntimeError(f"explicit {variable}=1 required before imports")
    contract = json.loads((OWNER / "contract.json").read_text())
    if digest(INPUT) != contract["input_sha256"] or source_bindings() != contract["sources"]:
        raise RuntimeError("contract source or input identity changed before execution")
    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
    signal.signal(signal.SIGALRM, timeout_handler)
    import numpy as np
    import threadpoolctl
    import numba
    runtime = dict(threadpools=threadpoolctl.threadpool_info(), numba_threads=numba.get_num_threads(),
                   environment={v: os.environ[v] for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")})
    if any(x["num_threads"] != 1 for x in runtime["threadpools"]) or runtime["numba_threads"] != 1:
        raise RuntimeError("actual runtime worker count differs from contract")
    data = np.load(INPUT, allow_pickle=False)
    arrays, cases = {}, {}
    started = time.monotonic()
    outcome = dict(status="running", contract_sha256=digest(OWNER / "contract.json"),
                   input_sha256=digest(INPUT), runtime=runtime, cases=cases,
                   highest_level_reached=False, fitted_corrections=False)
    try:
        for mode in MODES:
            remaining = 300 - (time.monotonic()-started)
            if remaining <= 0:
                raise TimeoutError("total wall-time limit")
            signal.setitimer(signal.ITIMER_REAL, min(120, remaining))
            cases[mode] = run_mode(data, mode, contract, arrays)
            signal.setitimer(signal.ITIMER_REAL, 0)
            write_json(OWNER / f"{mode}.json", cases[mode])
            print(json.dumps(dict(mode=mode, status=cases[mode]["status"], gates=cases[mode]["gates"],
                coverage_discriminator=cases[mode]["coverage_discriminator"])), flush=True)
        if digest(INPUT) != contract["input_sha256"] or source_bindings() != contract["sources"]:
            raise RuntimeError("scientific input/source changed during actual calculation")
        outcome["status"] = "passed"
    except BaseException as exc:
        outcome.update(status="failed", exception_type=type(exc).__name__, error=str(exc),
                       traceback=traceback.format_exc())
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        np.savez_compressed(OWNER / "arrays.npz", **arrays)
        outcome.update(elapsed_wall_seconds=time.monotonic()-started,
            peak_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            arrays_sha256=digest(OWNER / "arrays.npz"),
            source_identity_after=source_bindings(), input_sha256_after=digest(INPUT),
            scope="Actual complete finite native linked SD equation discriminator; no external solver, heavy-atom production roots, triples, speedup or experimental validation.")
        write_json(OWNER / "result.json", outcome)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    preparation() if args.prepare else execute()
