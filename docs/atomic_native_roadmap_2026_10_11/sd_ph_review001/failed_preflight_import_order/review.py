"""Independent core-normal-ordered PH review of immutable native SD arrays.

The PH action is built from elementary Fermi signs, rather than importing
the linked probe's generator or pair-ladder implementation.
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
LINKED = OWNER.parent / "sd_linked001"
MODES = ("dc", "static", "dynamic")
THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def bindings():
    old = json.loads((LINKED / "contract.json").read_text())
    paths = [ROOT / name for name in old["sources"]]
    paths += [Path(__file__).resolve()]
    sources = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    inputs = [LINKED / name for name in (
        "contract.json", "result.json", "arrays.npz", "inputs.npz", "verify.py",
        "dc.json", "static.json", "dynamic.json",
    )]
    inputs += [ROOT / old["input_path"]]
    return dict(sources=sources, inputs={str(path.relative_to(ROOT)): sha(path) for path in inputs})


def prepare():
    if (OWNER / "contract.json").exists():
        raise FileExistsError("premeasurement review contract is immutable")
    original = json.loads((LINKED / "contract.json").read_text())
    result = json.loads((LINKED / "result.json").read_text())
    if result["status"] != "passed":
        raise AssertionError("complete native finite source result must have passed")
    for mode in MODES:
        counts = original["actual"][mode]
        if (counts["spinors"], counts["electrons"], counts["determinants"],
                counts["scalar_plan"]["singles"], counts["scalar_plan"]["doubles"]) != (12, 4, 495, 2, 10):
            raise AssertionError("original complete physical space differs")
    contract = dict(
        status="predeclared", bindings=bindings(), modes=MODES,
        fixed_space=dict(spinors=12, electrons=4, determinants=495,
                         scalar_singles=2, scalar_doubles=10, scalar_coordinates=12),
        hypothesis="The complete normal-ordered occupied-virtual two-body PH action reproduces every signed DD element of K minus projected Fock, PP and HH in this fixed finite model.",
        operator="sum_(a,b virtual;i,j occupied) g(a i,b j) [a_a^dag a_i^dag a_j a_b - delta_ij a_a^dag a_b]",
        gates=dict(signed_complete_DD_max_error_Ha=1e-11, scalar_metric=1e-12,
                   reconstructed_saved_decomposition_Ha=1e-11),
        resources=dict(blas_threads=1, omp_threads=1, numba_threads=1,
                       address_space_bytes=2 * 1024**3, cpu_seconds=120,
                       total_wall_seconds=120, numerical_PH_action_bytes=16 * 495 * 10,
                       scope="Bounded review; full original fixed-N basis retained. No controlled timing or performance claim."),
        distinguishing_outcomes=dict(
            accepted="All 100 signed DD elements in each of three unchanged Hamiltonians match the independently normal-ordered PH action under the unchanged 1e-11 Ha gate.",
            rejected="A failed element or identity gate stops the review with failure preserved; no rescaling, model shrink or tolerance change.",
            limits="Identifies a finite PP/HH-only subgraph gap, not a missing contribution in the existing complete pCI-based production map or a Yb energy error."),
    )
    save(OWNER / "contract.json", contract)
    print(json.dumps({"status": "predeclared", "fixed_space": contract["fixed_space"]}))


def fermi(mask, operators):
    """Independent rightmost-first Fermi sign, without native sign helpers."""
    phase = 1
    for index, creation in reversed(operators):
        bit = 1 << index
        if bool(mask & bit) == creation:
            return None, 0
        if (mask & (bit - 1)).bit_count() % 2:
            phase = -phase
        mask ^= bit
    return mask, phase


def timeout(signum, frame):
    raise TimeoutError("predeclared 120-second bounded PH review")


def execute():
    if (OWNER / "result.json").exists():
        raise FileExistsError("original review evidence must not be overwritten")
    if any(os.environ.get(name) != "1" for name in THREAD_VARIABLES):
        raise RuntimeError("explicit one-thread environment required before imports")
    contract = json.loads((OWNER / "contract.json").read_text())
    if bindings() != contract["bindings"]:
        raise AssertionError("source/input bindings changed after premeasurement contract")
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 120)
    import numpy as np
    import threadpoolctl
    from atomx.physics.csf.spinor_integrals import SpinorIntegrals

    runtime = dict(environment={name: os.environ[name] for name in THREAD_VARIABLES},
                   threadpools=threadpoolctl.threadpool_info())
    if any(pool["num_threads"] != 1 for pool in runtime["threadpools"]):
        raise RuntimeError("actual numerical thread count differs from contract")
    started = time.monotonic()
    outcome = dict(status="running", contract_sha256=sha(OWNER / "contract.json"),
                   runtime=runtime, cases={}, external_solver_invoked=False,
                   fitted_corrections=False, highest_level_reached=False)
    matrices = {}
    try:
        original = json.loads((LINKED / "contract.json").read_text())
        for name, expected in original["sources"].items():
            for path in (ROOT / name, LINKED / "source" / name):
                if sha(path) != expected:
                    raise AssertionError(f"original source or snapshot changed: {path}")
        inp = np.load(LINKED / "inputs.npz", allow_pickle=False)
        arr = np.load(LINKED / "arrays.npz", allow_pickle=False)
        masks = tuple(sum(1 << i for i in row) for row in itertools.combinations(range(12), 4))
        lookup = {mask: row for row, mask in enumerate(masks)}
        if len(masks) != 495:
            raise AssertionError("complete fixed-N basis lost")
        for mode in MODES:
            occupied, frozen, reference_energy = inp[f"{mode}_constants"]
            occupied = int(occupied)
            labels = tuple(tuple(map(int, row)) for row in inp[f"{mode}_labels"])
            integrals = SpinorIntegrals(
                labels, occupied, inp[f"{mode}_one_body_Ha"], inp[f"{mode}_fock_Ha"],
                inp[f"{mode}_packed_eri_Ha"], float(frozen), float(reference_energy),
                json.loads(str(inp["identities_json"]))[mode],
            )
            scalar = arr[f"{mode}_scalar_vectors"]
            metric = float(np.max(abs(scalar.conj().T @ scalar - np.eye(13))))
            if scalar.shape != (495, 13) or occupied != 4 or len(labels) != 12:
                raise AssertionError("complete reference + 12 scalar columns lost")
            doubles = scalar[:, 3:]
            action = np.zeros((495, 10), dtype=np.complex128)
            nonzero_terms = 0
            for a, b, i, j in itertools.product(range(4, 12), range(4, 12), range(4), range(4)):
                coefficient = integrals.eri(a, i, b, j)
                if coefficient == 0:  # Exact algebraic zero only.
                    continue
                nonzero_terms += 1
                terms = [(((a, True), (i, True), (j, False), (b, False)), coefficient)]
                if i == j:
                    terms.append((((a, True), (b, False)), -coefficient))
                for column in range(10):
                    for row in np.flatnonzero(doubles[:, column]):
                        for operators, value in terms:
                            target, phase = fermi(masks[row], operators)
                            if phase:
                                action[lookup[target], column] += value * phase * doubles[row, column]
            ph = doubles.conj().T @ action
            missing = arr[f"{mode}_missing_after_Fock_PP_HH"]
            reconstructed = (arr[f"{mode}_K"][2:, 2:]
                             - arr[f"{mode}_fock_K"][2:, 2:]
                             - arr[f"{mode}_pp"] - arr[f"{mode}_hh"])
            errors = dict(signed_complete_DD_max_error_Ha=float(np.max(abs(ph - missing))),
                          reconstructed_saved_decomposition_Ha=float(np.max(abs(reconstructed - missing))),
                          scalar_metric=metric)
            for key, error in errors.items():
                if not math.isfinite(error) or error > contract["gates"][key]:
                    raise AssertionError(f"unchanged {mode}/{key} gate: {error}")
            diagonal = np.diag(np.diag(ph))
            outcome["cases"][mode] = dict(
                status="passed", errors=errors, signed_entries_checked=100,
                nonzero_PH_integral_terms=nonzero_terms,
                normal_ordered_PH_diagonal_max_Ha=float(np.max(abs(diagonal))),
                normal_ordered_PH_offdiagonal_max_Ha=float(np.max(abs(ph - diagonal))),
                saved_partial_projected_Fock_PP_HH_action_residual_Ha=
                    json.loads((LINKED / f"{mode}.json").read_text())["coverage_discriminator"]["full_root_residual_if_only_Fock_PP_HH_Ha"],
            )
            matrices[f"{mode}_normal_ordered_PH_DD"] = ph
            matrices[f"{mode}_saved_missing_DD"] = missing
        if bindings() != contract["bindings"]:
            raise AssertionError("original sources or evidence changed during review")
        outcome["status"] = "passed"
    except BaseException as exc:
        outcome.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        np.savez_compressed(OWNER / "matrices.npz", **matrices)
        outcome.update(elapsed_wall_seconds=time.monotonic() - started,
                       peak_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                       source_and_input_bindings_unchanged=bindings() == contract["bindings"],
                       matrices_sha256=sha(OWNER / "matrices.npz"),
                       scope="Finite unchanged native normal-ordered DD PH discriminator; no production solver change, Yb spectrum, experimental error, nonlinear CC, triples or performance comparison.",
                       all_blocks_caveat="Saved Fock_K is the projected normal-ordered Fock matrix. In DD this equals its linked commutator action; in the DS block it excludes the noncanonical T1 Fock|reference> product subtraction. The archived all-block partial residual is not a full-Fock-commutator residual.")
        save(OWNER / "result.json", outcome)
    print(json.dumps(outcome))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else execute()
