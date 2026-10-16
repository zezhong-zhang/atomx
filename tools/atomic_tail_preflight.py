"""Complete atomic SD tail task planning without correlation allocation.

The public radial and SD planners own the physics and coordinate counts.
This diagnostic adds numerical64, norm replay, archive and CUDA staging
reservations. A planning pass never authorizes a solve or reduces a space.
"""

import hashlib
import json
import math
import os
import shutil
from pathlib import Path

GIB = 1024**3
MIB = 1024**2
YB_STATES = ((6, -1), (6, 1), (6, -2), (5, 2), (5, -3))
YB_PANEL = ((3, 5.0), (3, 10.0), (4, 5.0), (4, 10.0))


def digest(path):
    """Hash a file with bounded memory, including large scientific inputs."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(MIB):
            hasher.update(block)
    return hasher.hexdigest()


def identity_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def plain(value):
    if hasattr(value, "items"):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def retained_band(space, *, lmax, energy_max_Ha):
    """Select a complete declared band from one maximal public radial solve.

    The public builder filters eigenvalues with strict '<', after solving
    each kappa. Selecting subsets of those same ordered roots avoids solving
    the identical Fock matrix four times and preserves phases and numbering.
    """
    from atomx.angular.recoupling import l_of_kappa
    from atomx.methods.atomic_space import AtomicVirtualSpace

    if (
        isinstance(lmax, bool)
        or not isinstance(lmax, int)
        or lmax < 0
        or lmax > space.identity["virtual_lmax"]
        or isinstance(energy_max_Ha, bool)
        or not isinstance(energy_max_Ha, (int, float))
        or not math.isfinite(energy_max_Ha)
        or energy_max_Ha > space.identity["virtual_energy_max_Ha"]
    ):
        raise ValueError("finite contained partial-wave/energy band required")
    energies = space.identity["virtual_orbital_energies_Ha"]
    if set(energies) != {str(key) for key in space.virtual_keys}:
        raise ValueError("complete virtual key/energy identity required")
    if any(not math.isfinite(float(value)) for value in energies.values()):
        raise ValueError("finite complete virtual energies required")
    keys = tuple(
        key
        for key in space.virtual_keys
        if l_of_kappa(key[1]) <= lmax and float(energies[str(key)]) < energy_max_Ha
    )
    if not keys:
        raise ValueError("nonempty complete retained band required")
    return AtomicVirtualSpace(
        {key: space.orbitals[key] for key in space.occupied_keys + keys},
        space.occupied_keys,
        keys,
        {
            **space.identity,
            "virtual_lmax": lmax,
            "virtual_energy_max_Ha": float(energy_max_Ha),
            "virtual_orbital_energies_Ha": {
                str(key): energies[str(key)] for key in keys
            },
            "selection": "complete strict energy band of same maximal public VN solve",
        },
    )


def estimate_task(space, feature_identity, *, worker_threads=1, norm_planner=None):
    """Reuse complete rho planners and bound additional diagnostic storage.

    CPU and CUDA bounds describe the existing private numerical64 protocol.
    They exclude interpreter/JIT/compiler RSS: an explicit separate runtime
    reservation is added for admission. CUDA angular rows are bounded by
    their original 128 MiB cap and must still be counted before native load.
    """
    from atomx.methods.core_sd import plan_core_sd
    from atomx.methods.reduced_monovalent_sd import plan_reduced_monovalent_sd

    if (
        isinstance(worker_threads, bool)
        or not isinstance(worker_threads, int)
        or worker_threads < 1
    ):
        raise ValueError("positive explicit worker count required")
    core = plan_core_sd(
        space.occupied_keys,
        space.virtual_keys,
        feature_identity,
        term21_threads=worker_threads,
        term21_mode="streamed",
        core_singles_blas=True,
        core_single_to_doubles_blas=True,
    )
    valence = plan_reduced_monovalent_sd(
        core,
        YB_STATES,
        valence_term2_blas=True,
        valence_term2_threads=worker_threads,
    )
    d = core["dimensions"]
    counts = valence["amplitudes"]
    largest = max(counts.values())
    # Original public40-vector/restart20 preflight, then private numeric64
    # correction from probe_yb_monovalent_subspace.py; no physics selection.
    cpu40 = max(
        core["complete_payload_bound_bytes"] + 40 * core["krylov_vector_bytes"],
        valence["complete_payload_bound_bytes"] + 8 * (40 * largest + 4 * 21**2),
    )
    term31 = 16 * d["NHF"] * d["NXX"] + 8 * d["NHF"] * (d["NX"] + 16)
    term_a4 = 32 * d["NHF"] * d["NX"]
    cuda_outputs = 16 * d["NXX"] ** 2 * valence["nvh"]
    numeric64 = 8 * (96 * largest + 8 * (65**2 - 17**2))
    legacy_gpu64 = cpu40 + term31 + term_a4 + cuda_outputs + numeric64
    kappas = {k for _, k in space.occupied_keys + space.virtual_keys}
    maxradial = max(
        sum(key in space.virtual_keys for key in group) for group in core["groups"]
    )
    maxrank = d["KK"] + 1
    # Existing complete norm helper's own Gram reservation is a distinct
    # cap, even when the whole task fits host memory. Keep its64MiB cap.
    legacy_norm_scratch = (
        2 * MIB + 128 * len(kappas) ** 4 * maxrank**2 + 64 * maxrank * maxradial**2
    )
    norm_scratch = legacy_norm_scratch
    norm_plans = None
    if norm_planner is not None:
        if not callable(norm_planner):
            raise ValueError("explicit complete norm planner required")
        norm_plans = {str(key): norm_planner(core, valence, key) for key in YB_STATES}
        values = [p["required_bytes"] for p in norm_plans.values()]
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values):
            raise ValueError("positive integer complete norm scratch bounds required")
        norm_scratch = max(values)
    norm_replay = (
        24 * core["raw_amplitudes"] + 16 * largest + max(64 * MIB, norm_scratch)
    )
    # prepare() holds rows/coeff and their copies together with phi fields;
    # density construction also stages geometry/CSR on the host. These were
    # absent as explicit additions in the historical numeric64 host bound.
    geometry = 8 * (
        8 * d["NHF"]
        + 3 * (2 * d["NV"] * d["NG"])
        + 12 * d["NE"]
        + 6 * d["NV"] * d["NG"]
    )
    angular_staging = 2 * 128 * MIB
    field_staging = 16 * (
        d["NHF"] * d["NX"] * d["NK"] + d["NX"] * d["NK"] + d["NN"] + d["NK"]
    )
    host_staging = angular_staging + field_staging + geometry
    density = geometry + 8 * d["NXX"] ** 2 * (2 * d["NHF"] + 4 * d["NV"])
    device = density + 8 * (
        d["NHF"] * d["NX"] * d["NK"]
        + 6 * d["NXX"] ** 2 * valence["nvh"]
        + 8 * d["NHF"] * d["NXX"]
        + 3 * d["NXX"] ** 2
    )
    host_payload = legacy_gpu64 + norm_replay + host_staging
    runtime_reserve = 1 * GIB
    # Explicit artifact protocol reserve: at most16 complete core vectors
    # plus32 vectors for every joint state, uncompressed, with a128MiB build
    # and small metadata reserve. An executor must enforce this write budget.
    disk = 8 * (16 * core["raw_amplitudes"] + 32 * sum(counts.values())) + 128 * MIB
    return {
        "dimensions": dict(d),
        "occupied_radials": len(space.occupied_keys),
        "occupied_electrons": sum(2 * abs(k) for _, k in space.occupied_keys),
        "virtual_radials": len(space.virtual_keys),
        "virtual_magnetic_spinors": sum(2 * abs(k) for _, k in space.virtual_keys),
        "core_raw_amplitudes": core["raw_amplitudes"],
        "independent_core_scalar_amplitudes": core["independent_scalar_amplitudes"],
        "state_amplitudes": {str(key): int(counts[key]) for key in YB_STATES},
        "per_state_core_reuse": "same source/space/core solution only; all five states mandatory",
        "maximum_legacy_numeric64_host_payload_bytes": int(legacy_gpu64),
        "norm_replay_bytes": int(norm_replay),
        "cuda_host_staging_bytes": int(host_staging),
        "cuda_host_angular_plan_staging_bytes": angular_staging,
        "maximum_complete_host_payload_bytes": int(host_payload),
        "runtime_reserve_bytes": runtime_reserve,
        "host_admission_reservation_bytes": int(host_payload + runtime_reserve),
        "device_payload_bound_bytes": int(device),
        "device_reservation_bytes": 2 * GIB,
        "angular_plan_before_native_load_required": True,
        "native_cuda_bridge_worker_requirement": 8,
        "native_cuda_matches_worker_policy": worker_threads == 8,
        "artifact_write_reservation_bytes": int(disk),
        "complete_valence_vector_entries_per_160_actions": 160 * sum(counts.values()),
        "full_native_flop_and_wall_bound": None,
        "work_scope": "vector traffic only; diagram FLOPs/end-to-end wall not bounded by coordinate count",
        "worker_threads": worker_threads,
        "inner_blas_threads": 1,
        "norm_scratch_cap_bytes": 64 * MIB,
        "norm_scratch_required_bytes": norm_scratch,
        "legacy_norm_scratch_required_bytes": legacy_norm_scratch,
        "norm_scratch_strategy": "explicit complete per-state planner"
        if norm_plans is not None
        else "historical global-quartet reservation",
        "norm_state_scratch_plans": norm_plans,
        "norm_scratch_within_original_cap": norm_scratch <= 64 * MIB,
        "scope": "complete allocated payload plus explicit reservations; actual source-bound constructors repeat preflight",
    }


def machine_snapshot(output_directory):
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("MemAvailable", "MemTotal"):
            info[key] = int(value.split()[0]) * 1024
    return {
        "available_host_bytes": info["MemAvailable"],
        "total_host_bytes": info["MemTotal"],
        "available_disk_bytes": shutil.disk_usage(output_directory).free,
        "cpu_count": os.cpu_count(),
        "load": os.getloadavg(),
        "gpu_free_bytes": None,
        "gpu_free_scope": "not queried through CUDA; device presence/free memory requires live admission check",
    }


def admit(resource, machine, *, host_cap_bytes=32 * GIB, device_cap_bytes=2 * GIB):
    """Fail closed; a successful count is separate from task admission."""
    reasons = []
    if resource["maximum_complete_host_payload_bytes"] > host_cap_bytes:
        reasons.append("complete_host_payload_exceeds_original_cap")
    if resource["host_admission_reservation_bytes"] > machine["available_host_bytes"]:
        reasons.append("insufficient_live_host_memory")
    if resource["artifact_write_reservation_bytes"] > machine["available_disk_bytes"]:
        reasons.append("insufficient_live_disk_reservation")
    if resource["device_payload_bound_bytes"] > device_cap_bytes:
        reasons.append("complete_device_payload_exceeds_original_cap")
    if not resource["norm_scratch_within_original_cap"]:
        reasons.append("complete_norm_scratch_exceeds_original_cap")
    if not resource["native_cuda_matches_worker_policy"]:
        reasons.append("existing_cuda_bridge_requires_eight_workers")
    free_gpu = machine["gpu_free_bytes"]
    if free_gpu is None:
        reasons.append("live_device_free_memory_unverified")
    elif resource["device_reservation_bytes"] > free_gpu:
        reasons.append("insufficient_live_device_memory")
    # The current accounting probe has no tail root executor and no signed
    # native angular plan build. Passing memory cannot waive those contracts.
    reasons.extend(
        ("source_bound_tail_executor_required", "native_angular_plan_count_required")
    )
    return {
        "status": "deferred",
        "reasons": reasons,
        "physical_model_reduced": False,
        "solve_authorized": False,
    }
