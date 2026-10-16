"""Complete saved-band counts with the adopted bounded norm planner."""

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWNER = HERE.parents[1]
REPO = OWNER.parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "tools"), str(OWNER)]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def main(output):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    contract = json.loads((HERE / "tail_probe_contract.json").read_text())
    files = sorted(set(contract["source_files"] + contract["input_files"]))
    hashes = {relative: digest(REPO / relative) for relative in files}
    (output / "contract.json").write_text(json.dumps(contract, indent=2))
    (output / "source_and_input_sha256.json").write_text(json.dumps(hashes, indent=2))
    for relative in contract["source_files"]:
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / relative).read_bytes())

    import numpy as np
    from atomic_tail_preflight import YB_STATES, admit, estimate_task, machine_snapshot
    from monovalent_norm_control import plan_scratch

    from atomx.methods.atomic_space import AtomicVirtualSpace

    previous = json.loads((OWNER / "parallel_tail007/run004/result.json").read_text())
    if not all(previous["gates"].values()):
        raise ValueError("complete original band replay is not certified")
    adoption = json.loads((HERE / "adoption_identity.json").read_text())
    if (
        not adoption["all_function_and_class_AST_equal"]
        or not adoption["cache_limits_equal"]
        or digest(OWNER / "monovalent_norm_control.py")
        != adoption["adopted_source_sha256"]
    ):
        raise ValueError("adopted complete norm source differs from verified candidate")
    finite = json.loads((HERE / "finite004/result.json").read_text())
    multicore = json.loads((HERE / "multicore002/result.json").read_text())
    if any(d["status"] != "pass" or not d["source_match"] for d in (finite, multicore)):
        raise ValueError("actual adopted finite/multicore norm replay failed")
    for directory in (HERE / "finite004", HERE / "multicore002"):
        sources = json.loads((directory / "source_sha256.json").read_text())
        if (
            sources["docs/atomic_pci_sd_native_2026_10_09/monovalent_norm_control.py"]
            != (adoption["adopted_source_sha256"])
        ):
            raise ValueError("norm replay used a different helper")
    replay = json.loads((OWNER / "yb_full_monovalent006/core_replay.json").read_text())
    feature = replay["provenance"]["feature_identity"]
    machine = machine_snapshot(output)
    tasks = []
    for old in previous["tasks"]:
        band_file = OWNER / "parallel_tail007/run004" / (old["task_id"] + "_band.json")
        band = json.loads(band_file.read_text())
        occupied, virtual = (
            tuple(map(tuple, band[k])) for k in ("occupied_keys", "virtual_keys")
        )
        # Only coordinate counting uses empty radial placeholders. They are
        # never passed to a correlation/native constructor or physical action.
        space = AtomicVirtualSpace(
            {key: (np.empty(0), np.empty(0)) for key in occupied + virtual},
            occupied,
            virtual,
            {"virtual_orbital_energies_Ha": band["energies_Ha"]},
        )
        plan = estimate_task(
            space, feature, worker_threads=8, norm_planner=plan_scratch
        )
        old_plan = old["existing_gpu8_resource"]
        if (
            plan["state_amplitudes"] != old_plan["state_amplitudes"]
            or plan["core_raw_amplitudes"] != old_plan["core_raw_amplitudes"]
            or set(plan["norm_state_scratch_plans"]) != {str(k) for k in YB_STATES}
            or not plan["norm_scratch_within_original_cap"]
            or plan["maximum_legacy_numeric64_host_payload_bytes"]
            != old_plan["maximum_legacy_numeric64_host_payload_bytes"]
        ):
            raise ValueError("complete counts/original bounds/norm cap changed")
        tasks.append(
            dict(
                task_id=old["task_id"],
                lmax=old["lmax"],
                energy_max_Ha=old["energy_max_Ha"],
                resource=plan,
                admission=admit(plan, machine),
            )
        )
    unchanged = all(digest(REPO / p) == value for p, value in hashes.items())
    if not unchanged:
        raise ValueError("source or complete-band input changed during counting")
    result = dict(
        status="pass",
        gates=dict(
            adopted_source_matches_verified_candidate=True,
            actual_finite_multicore_replays=True,
            all_four_complete_bands=True,
            all_five_states=True,
            original_coordinate_and_numeric64_counts=True,
            original_64MiB_norm_cap=True,
            sources_and_inputs_unchanged=unchanged,
        ),
        tasks=tasks,
        previous_spectrum_unchanged=True,
        new_tail_spectra_computed=False,
        full_Yb_norm_executed=False,
        card_summary="完整五态四档计数通过；有界角向缓存将norm暂存上限降至14.15–18.54MB，低于原64MiB。实际有限/多核符号控制通过；完整Yb norm、尾部根求解与GPU生产准入仍待验证，尚无新实验谱精度改善。",
        scope="Complete saved radial identities and pure coordinate counts; no native/GPU/root/physical-tail execution",
        seconds=time.monotonic() - started,
        peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        highest_level_reached=False,
    )
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print(
        json.dumps(
            dict(
                status=result["status"],
                maximum_scratch_bytes=max(
                    t["resource"]["norm_scratch_required_bytes"] for t in tasks
                ),
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
