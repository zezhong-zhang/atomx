"""Actually consume complete saved five-state/all18M amplitudes with current norm."""

import argparse
import hashlib
import json
import os
import resource
import shutil
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWNER = HERE.parent
REPO = OWNER.parents[1]
STATES = ((6, -1), (6, 1), (6, -2), (5, 2), (5, -3))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def plain(value):
    if hasattr(value, "items"):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def main(out):
    began = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (out / name).write_text(
            json.dumps(plain(value), ensure_ascii=False, indent=2) + "\n"
        )

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
        lambda *_: (_ for _ in ()).throw(TimeoutError("complete norm120s wall cap")),
    )
    signal.alarm(120)
    run = OWNER / "yb_full_monovalent006"
    seed = OWNER / "yb_full_run008/yb_full_solution.npz"
    inputs = OWNER / "yb_full_run008/yb_full_inputs.npz"
    historical = OWNER / "yb_monovalent_norm001"
    adoption = OWNER / "parallel_tail007/norm_integration001"
    source_paths = [
        Path(__file__),
        HERE / "finite_anchor.py",
        OWNER / "monovalent_norm_control.py",
    ] + [
        REPO / name
        for name in (
            "src/atomx/methods/core_sd.py",
            "src/atomx/methods/linearized_sd.py",
            "src/atomx/methods/reduced_monovalent_sd.py",
            "src/atomx/methods/monovalent_sd.py",
            "src/atomx/methods/valence_sd.py",
            "src/atomx/angular/csf_hamiltonian.py",
            "src/atomx/angular/sd_rho.py",
            "src/atomx/angular/wigner.py",
            "src/atomx/angular/recoupling.py",
            "src/atomx/numerics/fermion.py",
            "src/atomx/numerics/bspline/coulomb_features.py",
            "src/atomx/physics/csf/spinor_integrals.py",
            "src/atomx/methods/_pci_monovalent.f90",
            "src/atomx/methods/_pci_valence.f90",
            "src/atomx/methods/_pci_valence_term2_streamed.f90",
            "tests/test_monovalent_sd.py",
            "tests/test_atomic_complete_norm_consumer.py",
            "tests/test_reduced_monovalent_sd.py",
        )
    ]
    input_paths = (
        [seed, inputs]
        + [
            run / name
            for name in (
                "contract.json",
                "resource.json",
                "core_replay.json",
                "result.json",
                "source_sha256.json",
                "source_postcheck.json",
                "gpu_source_postcheck.json",
                "gpu_result.json",
                "numeric64_result.json",
            )
        ]
        + [
            historical / name
            for name in (
                "result.json",
                "core_replay.json",
                "payload_sha256.json",
                "source_and_payload_postcheck.json",
            )
        ]
        + [adoption / "adoption_identity.json"]
    )
    for sub in ("finite004", "multicore002"):
        input_paths += [
            adoption / sub / name for name in ("result.json", "source_sha256.json")
        ]
    for n, k in STATES:
        input_paths += [
            run / f"root_{n}_{k}.json",
            run / f"root_{n}_{k}.npz",
            run / f"cpu_root_recheck_{n}_{k}.json",
        ]
    sources = {str(p.relative_to(REPO)): digest(p) for p in source_paths}
    payloads = {str(p.relative_to(REPO)): digest(p) for p in input_paths}
    producer = json.loads((run / "contract.json").read_text())
    saved_resource = json.loads((run / "resource.json").read_text())
    core_plan = saved_resource["core_plan"]
    counts = saved_resource["standalone_plan"]["amplitudes"]
    payload_bound = (
        16 * core_plan["raw_amplitudes"] + 16 * max(counts.values()) + 64 * 1024**2
    )
    disk_bound = (
        sum(p.stat().st_size for p in source_paths + input_paths) + 16 * 1024**2
    )
    save(
        "contract.json",
        dict(
            hypothesis="Current adopted bounded helper actually preserves full original five-state/all18M physical norm and signed components on the unchanged complete original amplitude payload.",
            fixed=producer["fixed"],
            states=STATES,
            helper_sha256=sources[
                str((OWNER / "monovalent_norm_control.py").relative_to(REPO))
            ],
            source_sha256=sources,
            input_sha256=payloads,
            raw_core_coordinates=core_plan["raw_amplitudes"],
            all_state_coordinates=counts,
            allocated_payload_bound_bytes=payload_bound,
            runtime_reserve_bytes=1024**3,
            disk_archive_reservation_bytes=disk_bound,
            address_cap_bytes=4 * 1024**3,
            wall_cap_seconds=120,
            threads=1,
            GPU=False,
            component_and_complete_M_gate=1e-11,
            distinguishing="Fresh live-helper104-case fermion anchor and freshly computed18 complete Yb contractions; old values only secondary parity oracle.",
            stop="Any full source/input/coordinate/state/signed finite/current helper/cap/M gate failure rejects, preserves failure and never truncates.",
            scope="New metric consumption of original certified saved wavefunctions, without new CoreSD/root/full-H/property/tail convergence/experimental accuracy certificate",
        ),
    )
    if (
        payload_bound + 1024**3 > 4 * 1024**3
        or shutil.disk_usage(out).free < disk_bound
    ):
        raise MemoryError(
            "complete norm payload/runtime/input archive exceeds budget; no model shrink"
        )
    save("source_sha256.json", sources)
    save("input_sha256.json", payloads)
    for category, paths in (("source", source_paths), ("input", input_paths)):
        for path in paths:
            target = out / category / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    sys.path[:0] = [str(REPO / "src"), str(REPO / "tests"), str(OWNER), str(HERE)]
    import monovalent_norm_control as norm
    import numpy as np
    from finite_anchor import run as anchor

    from atomx.methods.core_sd import plan_core_sd
    from atomx.methods.reduced_monovalent_sd import plan_reduced_monovalent_sd

    identity = json.loads((adoption / "adoption_identity.json").read_text())
    if (
        digest(OWNER / "monovalent_norm_control.py")
        != identity["adopted_source_sha256"]
        or not identity["all_function_and_class_AST_equal"]
    ):
        raise ValueError("adopted helper differs from independently validated source")
    for directory, checks in (("finite004", 672), ("multicore002", 168)):
        proof = json.loads((adoption / directory / "result.json").read_text())
        frozen = json.loads((adoption / directory / "source_sha256.json").read_text())
        if (
            proof["status"] != "pass"
            or proof["checks"] != checks
            or not proof["source_match"]
            or frozen[str((OWNER / "monovalent_norm_control.py").relative_to(REPO))]
            != identity["adopted_source_sha256"]
        ):
            raise ValueError("adopted finite/multicore anchor source gate failed")
    save("fresh_finite_anchor.json", anchor(norm))
    if (
        digest(inputs) != producer["input_sha256"]
        or digest(seed) != producer["seed_sha256"]
    ):
        raise ValueError("original complete wavefunction inputs/seed changed")
    for name in (
        "source_postcheck.json",
        "gpu_source_postcheck.json",
        "gpu_result.json",
        "numeric64_result.json",
    ):
        if json.loads((run / name).read_text())["status"] != "pass":
            raise ValueError("original complete root execution audit failed")
    before = json.loads((historical / "result.json").read_text())
    core_replay = json.loads((historical / "core_replay.json").read_text())
    if (
        before["status"] != "pass"
        or not core_replay["seed_bitwise_identical"]
        or core_replay["residual_max_Ha"] > 1e-10
    ):
        raise ValueError("original actual complete core amplitude provenance failed")
    occupied, virtual = (
        tuple(map(tuple, core_plan[name])) for name in ("occupied_keys", "virtual_keys")
    )
    feature = json.loads((run / "core_replay.json").read_text())["provenance"][
        "feature_identity"
    ]
    cp = plan_core_sd(
        occupied,
        virtual,
        feature,
        term21_threads=8,
        term21_mode="streamed",
        core_singles_blas=True,
        core_single_to_doubles_blas=True,
    )
    vp = plan_reduced_monovalent_sd(
        cp, STATES, valence_term2_blas=True, valence_term2_threads=8
    )
    if (
        plain(cp) != saved_resource["core_plan"]
        or plain(vp) != saved_resource["standalone_plan"]
    ):
        raise ValueError("all original complete physical rho coordinates differ")
    save(
        "resource.json",
        dict(
            allocated_payload_bound_bytes=payload_bound,
            runtime_reserve_bytes=1024**3,
            norm_state_scratch={str(k): norm.plan_scratch(cp, vp, k) for k in STATES},
            machine_load=os.getloadavg(),
            disk_archive_reservation_bytes=disk_bound,
            scope="Only full saved core+one full valence, NPZ loading copy reserve, norm64MiB and separate runtime; no native constructor/backend/root/GPU allocation",
        ),
    )
    with np.load(seed, allow_pickle=False) as packed:
        crho = packed["amplitudes"]
    raw_hash = hashlib.sha256(memoryview(crho).cast("B")).hexdigest()
    if (
        raw_hash != core_replay["actual_amplitude_sha256"]
        or crho.shape != (cp["raw_amplitudes"],)
        or crho.dtype != np.float64
    ):
        raise ValueError("complete actual core amplitude raw hash differs")
    rows = []
    component_names = (
        "reference",
        "valence_singles",
        "valence_doubles",
        "core_singles_unblocked",
        "core_singles_valence_doubles_interference",
        "core_doubles_total",
        "core_doubles_pauli_blocked",
        "core_doubles_unblocked",
        "norm_squared",
        "wavefunction_norm",
    )
    common = None
    for key in STATES:
        n, k = key
        root = json.loads((run / f"root_{n}_{k}.json").read_text())
        audit = json.loads((run / f"cpu_root_recheck_{n}_{k}.json").read_text())
        provenance = root["provenance"]
        fields = (
            "hamiltonian",
            "binding",
            "occupied_keys",
            "virtual_keys",
            "feature_identity",
            "equation",
            "amplitude_representation",
            "physical_source_sha256",
        )
        physical = {name: provenance[name] for name in fields}
        if common is None:
            common = physical
        if (
            physical != common
            or root["column_key"] != list(key)
            or root["channels"] != plain(vp["channels"][key])
            or root["single_keys"] != plain(vp["single_keys"][key])
        ):
            raise ValueError(
                "same-H/same-core/same-space complete five-state identity failed"
            )
        if (
            audit["status"] != "pass"
            or max(audit["raw_residual_Ha"], audit["energy_residual_Ha"]) > 1e-10
        ):
            raise ValueError("original actual CPU residual certificate failed")
        with np.load(run / f"root_{n}_{k}.npz", allow_pickle=False) as packed:
            rho = packed["amplitudes"]
        if rho.shape != (vp["amplitudes"][key],):
            raise ValueError("complete valence rho vector required")
        actual, started = [], time.monotonic()
        previous = next(row for row in before["rows"] if row["key"] == list(key))
        for m in norm.projections(k):
            value = norm.norm_components(cp, vp, crho, rho, key, twoM=m)
            old = next(v for v in previous["components"] if v["twoM"] == m)
            error = max(abs(value[name] - old[name]) for name in component_names)
            if error > 1e-11:
                raise ValueError(
                    "fresh bounded full-array norm differs from prior physical norm"
                )
            actual.append(
                dict(twoM=m, **value, max_historical_component_difference=error)
            )
        spread = max(v["norm_squared"] for v in actual) - min(
            v["norm_squared"] for v in actual
        )
        if spread > 1e-11:
            raise ValueError("all complete Yb magnetic projections disagree")
        rows.append(
            dict(
                key=key,
                components=actual,
                complete_M_spread=spread,
                seconds=time.monotonic() - started,
                raw_valence_amplitude_sha256=hashlib.sha256(
                    memoryview(rho).cast("B")
                ).hexdigest(),
            )
        )
        save(
            "progress.json",
            dict(
                rows=rows,
                full_array_contractions_completed=sum(
                    len(r["components"]) for r in rows
                ),
            ),
        )
        print(
            json.dumps(
                dict(
                    stage="new_full_bounded_norm_consumption",
                    key=key,
                    all_M=len(actual),
                    norm_squared=actual[0]["norm_squared"],
                    seconds=rows[-1]["seconds"],
                )
            ),
            flush=True,
        )
        del rho
    unchanged = all(
        digest(REPO / name) == value for name, value in {**sources, **payloads}.items()
    )
    if not unchanged:
        raise ValueError(
            "consuming source or whole-amplitude input changed during new norm execution"
        )
    save(
        "source_and_input_postcheck.json",
        dict(status="pass", sources=len(sources), inputs=len(payloads)),
    )
    result = dict(
        status="pass",
        gates=dict(
            current_adopted_source=True,
            fresh_independent_signed_finite_anchor=True,
            original_complete_core_raw_amplitudes=True,
            original_complete_five_valence_columns=True,
            original_certified_same_H_core_space=True,
            all18_actual_new_norm_contractions=True,
            original64MiB_scratch_cap=True,
            original1e11_component_and_M_gates=True,
            sources_and_inputs_unchanged=True,
        ),
        rows=rows,
        core_actual_amplitude_sha256=raw_hash,
        common_physical_identity=common,
        full_Yb_norm_executed=True,
        new_root_or_tail_spectrum_computed=False,
        seconds=time.monotonic() - began,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        card_summary="当前有界helper已实际消费完整28,218,500核rho与五态全部18M，范数/符号/泡利/干涉和同空间门通过；未重新求根或产生新的尾部收敛谱，实验精度门仍未提升。",
        scope="New full saved-wavefunction metric contractions plus fresh finite anchor; no new CoreSD replay or new eigenstate/property/experimental certificate",
        highest_level_reached=False,
    )
    save("result.json", result)
    signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.output)
    except Exception as error:
        if args.output.is_dir():
            (args.output / "failure.json").write_text(
                json.dumps(
                    dict(type=type(error).__name__, message=str(error)), indent=2
                )
                + "\n"
            )
            (args.output / "result.json").write_text(
                json.dumps(
                    dict(
                        status="fail",
                        gates={"all_declared_gates": False},
                        card_summary=str(error),
                        highest_level_reached=False,
                    ),
                    indent=2,
                )
                + "\n"
            )
        raise
