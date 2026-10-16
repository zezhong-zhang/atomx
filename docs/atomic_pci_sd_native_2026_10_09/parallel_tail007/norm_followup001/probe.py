"""Full finite l4 magnetic norm versus independent fermion amplitudes."""

import argparse
import difflib
import hashlib
import importlib.util
import json
import os
import resource
import signal
import sys
import time
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
OWNER = HERE.parents[1]
ROOT = OWNER.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(OWNER), str(HERE)]
for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[name] = "1"
resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
signal.signal(
    signal.SIGALRM,
    lambda *_: (_ for _ in ()).throw(TimeoutError("norm finite120s cap")),
)
signal.alarm(120)

ORIGINAL = (
    HERE
    / "run002/source/docs/atomic_pci_sd_native_2026_10_09/monovalent_norm_control.py"
)
specification = importlib.util.spec_from_file_location("frozen_original_norm", ORIGINAL)
original = importlib.util.module_from_spec(specification)
specification.loader.exec_module(original)
import numpy as np  # noqa: E402
from candidate import BoundGrams, plan_scratch, specialize  # noqa: E402
from test_reduced_monovalent_sd import expanded  # noqa: E402

from atomx.methods.core_sd import plan_core_sd  # noqa: E402
from atomx.methods.monovalent_sd import FiniteMonovalentSD, _core_terms  # noqa: E402
from atomx.methods.reduced_monovalent_sd import plan_reduced_monovalent_sd  # noqa: E402
from atomx.numerics.fermion import apply_terms  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def square(state):
    return sum(value**2 for value in state.values())


def main(out):
    out.mkdir(exist_ok=False)
    paths = [
        Path(__file__),
        HERE / "candidate.py",
        ORIGINAL,
        ROOT / "tests/test_reduced_monovalent_sd.py",
        ROOT / "src/atomx/angular/sd_rho.py",
        ROOT / "src/atomx/angular/wigner.py",
        ROOT / "src/atomx/angular/recoupling.py",
        ROOT / "src/atomx/numerics/fermion.py",
        ROOT / "src/atomx/methods/core_sd.py",
        ROOT / "src/atomx/methods/valence_sd.py",
        ROOT / "src/atomx/methods/monovalent_sd.py",
        ROOT / "src/atomx/methods/reduced_monovalent_sd.py",
    ]
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    input_paths = [OWNER / "parallel_tail007/run004/result.json"] + sorted(
        (OWNER / "parallel_tail007/run004").glob("*_band.json")
    )
    input_hashes = {str(path.relative_to(ROOT)): sha(path) for path in input_paths}
    for path in paths:
        target = out / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())

    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + "\n")

    save(
        "contract.json",
        dict(
            hypothesis="Owned256-entry/4MiB Gram cache and quartet work bound preserve exact complete l4 norm while avoiding global kappa^4 residency estimate.",
            fixed="Synthetic complete1s2 core plus every s/p/d/f/g kappa multiplet, two distinct radial s orbitals, random complete signed redundant rho; no Hamiltonian/root/experimental result.",
            norm_payload_cap_bytes=64 * 1024**2,
            address_cap_bytes=4 * 1024**3,
            seconds_cap=120,
            physical_metric="explicit full-magnetic fermionic (1+Tcore+Sv)|Phiv> overlap",
            component_and_total_gate=1e-11,
            complete_M_gate=1e-11,
            random_seed=719230,
            stop="Any coordinate/source/component/sign/M/cap mismatch rejects; original shared helper unchanged; no longYb root or raised64MiB cap.",
            source_sha256=hashes,
            input_sha256=input_hashes,
            original_helper_sha256=sha(ORIGINAL),
            original_helper_scope="frozen run002 pre-adoption helper; current shared helper intentionally not consumed",
        ),
    )
    save("source_sha256.json", hashes)
    save("input_sha256.json", input_hashes)
    source = Path(original.__file__).read_text()
    candidate_source = specialize(source)
    (out / "candidate_norm.py").write_text(candidate_source)
    (out / "proposed_helper.patch").write_text(
        "".join(
            difflib.unified_diff(
                source.splitlines(True),
                candidate_source.splitlines(True),
                fromfile="monovalent_norm_control.py",
                tofile="owned-cache/monovalent_norm_control.py",
            )
        )
    )
    namespace = dict(BoundGrams=BoundGrams, plan_scratch=plan_scratch)
    exec(compile(candidate_source, str(out / "candidate_norm.py"), "exec"), namespace)
    occupied = ((1, -1),)
    virtual = (
        (2, -1),
        (3, -1),
        (2, 1),
        (2, -2),
        (3, 2),
        (3, -3),
        (4, 3),
        (4, -4),
        (5, 4),
        (5, -5),
    )
    feature = dict(
        auxiliary_nodes=20,
        quadrature_nodes=(2, 3),
        same_cell_edges=12,
        coefficient_sizes=(4, 4),
        component_orders=(3, 3),
        payload_estimate_bytes=10000,
    )
    cp = plan_core_sd(occupied, virtual, feature)
    vp = plan_reduced_monovalent_sd(cp, virtual)
    labels = tuple(
        (n, k, m) for n, k in occupied + virtual for m in original.projections(k)
    )
    integrals = SimpleNamespace(labels=labels, noccupied=2)
    core = SimpleNamespace(plan=cp, integrals=integrals)
    problem = SimpleNamespace(core=core, plan=vp)
    rng = np.random.default_rng(719230)
    crho = rng.normal(size=cp["raw_amplitudes"]) * 0.013
    vt = {key: rng.normal(size=vp["amplitudes"][key]) * 0.017 for key in virtual}
    np.savez(
        out / "signed_inputs.npz",
        core=crho,
        **{str(key): value for key, value in vt.items()},
    )
    ct = _core_terms(core, crho)
    cs, cd = (
        tuple(t for t in ct if len(t[1]) == 2),
        tuple(t for t in ct if len(t[1]) == 4),
    )
    core_double = square(apply_terms(cd, {3: 1.0}))
    # Original cap failure is retained, without executing it under a new cap.
    try:
        original.norm_components(cp, vp, crho, vt[virtual[0]], virtual[0])
    except MemoryError as error:
        save(
            "original_cap_failure.json",
            dict(type=type(error).__name__, message=str(error), cap_bytes=64 * 1024**2),
        )
    else:
        raise AssertionError("original l4 norm cap should reject")
    rows, max_error, max_peak, began = [], 0.0, 0, time.perf_counter()
    for key in virtual:
        covariance = {1: [], -1: []}
        for m in original.projections(key[1]):
            v = labels.index((*key, m))
            finite = SimpleNamespace(integrals=integrals, valence_index=v)
            finite.excitations = FiniteMonovalentSD._excitations(finite)
            attached = {3 | (1 << v): 1.0}
            ts, td = apply_terms(cs, attached), apply_terms(cd, attached)
            for sign in (1, -1):
                packed = sign * vt[key]
                decoded = expanded(problem, key, packed, finite)
                valence = FiniteMonovalentSD._valence_terms(finite, decoded)
                vs = apply_terms(tuple(t for t in valence if len(t[1]) == 2), attached)
                vd = apply_terms(tuple(t for t in valence if len(t[1]) == 4), attached)
                full = apply_terms(ct + valence, attached)
                mask = next(iter(attached))
                full[mask] = full.get(mask, 0) + 1
                expected = dict(
                    valence_singles=square(vs),
                    valence_doubles=square(vd),
                    core_singles_unblocked=square(ts),
                    core_singles_valence_doubles_interference=2
                    * sum(value * vd.get(mask, 0) for mask, value in ts.items()),
                    core_doubles_total=core_double,
                    core_doubles_pauli_blocked=core_double - square(td),
                    core_doubles_unblocked=square(td),
                    norm_squared=square(full),
                )
                tracemalloc.start()
                actual = namespace["norm_components"](cp, vp, crho, packed, key, twoM=m)
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                error = max(
                    abs(actual[name] - value) for name, value in expected.items()
                )
                max_error = max(max_error, error)
                max_peak = max(max_peak, peak)
                if (
                    error > 1e-11
                    or peak > actual["extra_scratch_bound_bytes"]
                    or actual["extra_scratch_bound_bytes"] > 64 * 1024**2
                ):
                    save(
                        "failure.json",
                        dict(
                            key=key,
                            twoM=m,
                            sign=sign,
                            actual=actual,
                            expected=expected,
                            component_error=error,
                            traced_peak_bytes=peak,
                        ),
                    )
                    raise AssertionError("complete norm/sign/memory gate failed")
                rows.append(
                    dict(
                        key=key,
                        twoM=m,
                        sign=sign,
                        max_component_error=error,
                        traced_peak_bytes=peak,
                        extra_scratch_bound_bytes=actual["extra_scratch_bound_bytes"],
                        cache=actual["angular_cache"],
                        actual_components={name: actual[name] for name in expected},
                        independent_components=expected,
                    )
                )
                covariance[sign].append(actual["norm_squared"])
        if any(max(values) - min(values) > 1e-11 for values in covariance.values()):
            raise AssertionError("complete magnetic covariance gate failed")
    # Force eviction independently, and compare fresh signed Gram arrays
    # elementwise; cache policy may affect reuse, never physical values.
    cache = BoundGrams(original.grams.__wrapped__, max_entries=2, max_bytes=4096)
    controls = [
        (-1, -1, -1, -1, (0, 1), (0, 1), "none", 0),
        (-5, -5, -4, -4, (0, 1), (0, 1), "none", 0),
        (4, -5, -1, -1, (0, 1), (0, 1), "hole", 1),
    ]
    for index in (0, 1, 2, 0):
        arrays = cache(*controls[index])
        oracle = original.grams.__wrapped__(*controls[index])
        if any(not np.array_equal(a, b) for a, b in zip(arrays, oracle, strict=True)):
            raise AssertionError("cache eviction changed signed Gram")
    assert cache.evictions > 0
    # Worst complete l4 quartet, with all allowed ranks present. This
    # independently measures construction temporaries beyond the resident
    # cache arrays; no physical magnetic row or signed Gram is discarded.
    ranks = tuple(range(cp["dimensions"]["KK"] + 1))
    tracemalloc.start()
    dense = original.grams.__wrapped__(-5, -5, -5, -5, ranks, ranks, "none", 0)
    _, angular_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    row_count = sum(
        mi + mj - mm in original.projections(-5)
        for mm in original.projections(-5)
        for mi in original.projections(-5)
        for mj in original.projections(-5)
    )
    angular_bound = plan_scratch(cp, vp, virtual[0])[
        "angular_quartet_transient_bound_bytes"
    ]
    if angular_peak > angular_bound:
        raise AssertionError("complete l4 angular construction exceeds declared bound")
    actual_yb = json.loads((OWNER / "parallel_tail007/run004/result.json").read_text())
    yb = []
    for task in actual_yb["tasks"]:
        band = json.loads(
            (
                OWNER / "parallel_tail007/run004" / (task["task_id"] + "_band.json")
            ).read_text()
        )
        c = plan_core_sd(
            tuple(map(tuple, band["occupied_keys"])),
            tuple(map(tuple, band["virtual_keys"])),
            feature,
        )
        p = plan_reduced_monovalent_sd(c, ((6, -1), (6, 1), (6, -2), (5, 2), (5, -3)))
        states = [dict(key=key, **plan_scratch(c, p, key)) for key in p["valence_keys"]]
        yb.append(
            dict(
                task_id=task["task_id"],
                original_physical_identity=task["physical_identity"],
                lmax=task["lmax"],
                emax_Ha=task["energy_max_Ha"],
                all_five_states=states,
                max_candidate_norm_scratch_bytes=max(
                    state["required_bytes"] for state in states
                ),
                no_large_norm_executed=True,
            )
        )
    assert all(sha(ROOT / name) == value for name, value in hashes.items())
    assert all(sha(ROOT / name) == value for name, value in input_hashes.items())
    save(
        "source_and_input_postcheck.json",
        dict(sources_unchanged=True, inputs_unchanged=True),
    )
    save(
        "signed_input_sha256.json",
        {"signed_inputs.npz": sha(out / "signed_inputs.npz")},
    )
    save(
        "result.json",
        dict(
            status="pass",
            gates=dict(
                signed_full_magnetic_oracle=True,
                complete_M_covariance=True,
                original_64MiB_failure_preserved=True,
                cache_eviction_preserves_all_Grams=True,
                all_five_Yb_complete_resource_plans=True,
                sources_unchanged=True,
                inputs_unchanged=True,
            ),
            card_summary="独立完整 l4 的 104 个 signed/全部 M 范数控制通过，原64MiB预算不变；候选有界Gram缓存待整合并复核完整Yb，未求解新能级。",
            cases=len(rows),
            component_gate=1e-11,
            max_component_error=max_error,
            max_traced_peak_bytes=max_peak,
            rows=rows,
            eviction_gate=dict(
                evictions=cache.evictions, peak_array_bytes=cache.peak_bytes
            ),
            complete_dense_angular_transient=dict(
                magnetic_rows=row_count,
                rank_count=len(ranks),
                f_and_r_array_bytes=16 * row_count * len(ranks),
                three_Gram_output_bytes=sum(array.nbytes for array in dense),
                traced_peak_bytes=angular_peak,
                declared_transient_bound_bytes=angular_bound,
                all_rows_and_ranks_retained=True,
            ),
            yb_counting_only=yb,
            seconds=time.perf_counter() - began,
            shared_helper_modified=False,
            highest_level_reached=False,
            scope="independent finite complete l4 signed norm and cache bounds only; candidate not adopted, no actual fullYb norm or controlled timing certificate",
        ),
    )
    signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        main(arguments.output)
    except Exception as error:
        if arguments.output.is_dir():
            (arguments.output / "failure.json").write_text(
                json.dumps(
                    dict(type=type(error).__name__, message=str(error)), indent=2
                )
                + "\n"
            )
        raise
