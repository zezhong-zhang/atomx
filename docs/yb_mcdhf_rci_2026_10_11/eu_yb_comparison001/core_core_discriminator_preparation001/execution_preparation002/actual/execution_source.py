"""Once-only native core-core differential RCI; default SOURCE/JSON checks only.

Numerical archive hashing, science imports and execution require a NEW exact
root seal and a fresh actual directory. No numerical work happens in preparation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import signal
import sys
import threading
import time
import traceback
import zipfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path("/data/code/atomx")
HERE = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    return value


def digest_json(value):
    return hashlib.sha256(
        json.dumps(plain(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write(path, value):
    temp = path.with_name(path.name + ".part")
    temp.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def metadata():
    """Check SOURCE/JSON only. All numerical files are deferred without access."""
    contract = read(HERE / "contract.json")
    roles = {key: ROOT / row["path"] for key, row in contract["input_roles"].items()}
    for key, row in contract["input_roles"].items():
        if not row["deferred_numerical"] and sha(roles[key]) != row["sha256"]:
            raise ValueError("SOURCE/JSON input changed: " + key)
    identity = read(roles["source_identity"])
    source = ROOT / contract["source_archive"]
    if (
        identity["source_map_sha256"] != contract["source_map_sha256"]
        or len(identity["all_atomx_python_sha256"]) != contract["full_source_count"]
    ):
        raise ValueError("Frozen f043 source-map identity changed")
    for name, expected in identity["all_atomx_python_sha256"].items():
        if sha(source / name.removeprefix("src/")) != expected:
            raise ValueError("Frozen full source archive changed: " + name)
    prerequisite = contract["prerequisite"]
    prior, receipt = read(roles["stageB_result"]), read(roles["stageB_receipt"])
    failures = [row["name"] for row in prior["checks"] if not row["passed"]]
    if (
        prior["status"] != prerequisite["stageB_expected_status"]
        or prior["all_declared_controls_passed"] is not False
        or sorted(failures)
        != sorted(prerequisite["exactly_preserved_ordinary_FD_failures"])
        or prior["original_stationarity_passed"] is not False
        or prior["expanded_MCDHF_optimized"] is not False
        or receipt["status"] != "foreground_closed"
        or receipt["exit_code"] != 1
        or receipt["wall_stop"] is not False
        or receipt["source_and_inputs_unchanged"] is not True
        or receipt["result_sha256"]
        != contract["input_roles"]["stageB_result"]["sha256"]
    ):
        raise ValueError("Closed StageB baseline / unchanged failure boundary changed")
    models = read(roles["prepared_models"])
    if (
        models["baseline_configs"]
        != read(roles["models"])["models"]["complete5f"]["configs"]
        or models["expanded_configs"]
        != models["baseline_configs"]
        + read(roles["core_core_proposal"])["candidate_NR_configs"]
        or len(models["baseline_configs"]) != 91
        or len(models["expanded_configs"]) != 100
    ):
        raise ValueError("Complete 91+9 configuration proposal changed")
    if (HERE / "preparation_manifest.json").exists():
        manifest = read(HERE / "preparation_manifest.json")
        for name, expected in manifest["files_sha256"].items():
            if sha(HERE / name) != expected:
                raise ValueError("Frozen preparation changed: " + name)
    return contract, roles, source


def authorize(path):
    contract, roles, source = metadata()
    binding = read(path)
    expected = dict(
        execution_authorized_by_root=True,
        pending_exact_root_execution_contract=False,
        once_only=True,
        preserve_all_configs_and_signed_amplitudes=True,
        literal_interpreter="/data/code/atomx/.venv/bin/python",
        driver_sha256=sha(HERE / "run.py"),
        contract_sha256=sha(HERE / "contract.json"),
        preparation_manifest_sha256=sha(HERE / "preparation_manifest.json"),
        source_map_sha256=contract["source_map_sha256"],
        input_roles=contract["input_roles"],
        resources=contract["resources"],
        acceptance_sha256=digest_json(contract["acceptance"]),
        accepted_iteration=13,
        stationarity_remains_failed=True,
    )
    for name, value in expected.items():
        if binding.get(name) != value:
            raise ValueError("Missing/mismatched NEW root seal: " + name)
    if os.path.abspath(sys.executable) != expected["literal_interpreter"]:
        raise ValueError("Use the literal audited project interpreter")
    return binding, contract, roles, source


def execute(binding_path):
    binding, contract, roles, source = authorize(binding_path)
    output = HERE / "actual"
    output.mkdir()  # Exclusive once-only claim, even if admission then fails.
    started = time.monotonic()
    caps, gates, fixed = (
        contract["resources"],
        contract["acceptance"],
        contract["fixed"],
    )
    attempted, completed, checks, archives, work = Counter(), Counter(), [], [], {}
    for forbidden in ("optimizer", "gradient", "Dirac_spectrum", "GPU"):
        attempted[forbidden] = completed[forbidden] = 0
    restored, points = [], {}
    work.update(
        connected_pairs_total=0,
        coefficient_table_nnz_total=0,
        coefficient_tables=[],
        projection_shapes=[],
        hS_kappas=[],
        radial_keys=0,
    )
    done = threading.Event()

    def progress(stage):
        write(
            output / "progress.json",
            dict(
                stage=stage,
                attempted=attempted,
                completed=completed,
                elapsed_uncontrolled_s=time.monotonic() - started,
                peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            ),
        )

    def check(name, error, limit=0.0):
        error = float(error)
        row = dict(
            name=name,
            error=error if math.isfinite(error) else None,
            limit=limit,
            passed=math.isfinite(error) and error <= limit,
        )
        checks.append(row)
        if not row["passed"]:
            progress("first_failed_gate/" + name)
            raise ValueError("Unchanged gate failed: " + name)

    def save(name, **arrays):
        path = output / name
        if path.exists():
            raise FileExistsError("Preserve completed arrays: " + name)
        temp = path.with_name(path.name + ".part")
        with temp.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
        archives.append(dict(path=name, sha256=sha(path)))
        progress("saved/" + name)

    def observe(module, name, count, maximum, after=None):
        original = getattr(module, name)

        def call(*args, **kwargs):
            attempted[count] += 1
            if attempted[count] > maximum:
                raise RuntimeError("Prospective call stop: " + count)
            result = original(*args, **kwargs)
            completed[count] += 1
            if after is not None:
                after(result, args, kwargs)
            return result

        restored.append((module, name, original))
        setattr(module, name, call)
        return original

    def alarm(*_):
        raise TimeoutError("Inner 900-second whole-run wall stop")

    def heartbeat():
        while not done.wait(30):
            print(
                json.dumps(
                    dict(
                        heartbeat_elapsed_s=time.monotonic() - started,
                        attempted=dict(attempted),
                        completed=dict(completed),
                        peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        / 1024,
                    )
                ),
                flush=True,
            )

    try:
        # Caps precede numerical file access and heavy imports. Root owns the
        # independent 915-second process-group wall guard for blocked C calls.
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMBA_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            os.environ[name] = "1"
        affinity = sorted(os.sched_getaffinity(0))
        mem = {
            line.split(":", 1)[0]: int(line.split()[1])
            for line in Path("/proc/meminfo").read_text().splitlines()
            if len(line.split()) >= 2 and line.split()[1].isdigit()
        }
        admission = dict(
            original_affinity=affinity,
            CPU=caps["CPU"],
            threads=1,
            host_available_GiB=mem["MemAvailable"] / 1024**2,
            load_average=os.getloadavg(),
            admitted=caps["CPU"] in affinity
            and mem["MemAvailable"] / 1024**2 >= caps["minimum_host_available_GiB"],
            uncontrolled_load=True,
            timing_claim_valid=False,
        )
        write(output / "resource_admission.json", admission)
        if not admission["admitted"]:
            raise RuntimeError("CPU27 / host available-memory admission failed")
        os.sched_setaffinity(0, {caps["CPU"]})
        byte_limit = int(caps["AS_GiB"] * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (byte_limit, byte_limit))
        signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, caps["inner_seconds"])
        threading.Thread(target=heartbeat, daemon=True).start()
        cache_dir = output / "numba_cache"
        cache_dir.mkdir()
        os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)
        (output / "execution_source.py").write_bytes(Path(__file__).read_bytes())
        write(output / "exact_execution_binding.json", binding)
        for role, row in contract["input_roles"].items():
            if row["deferred_numerical"]:
                plan = contract["array_access"][role]
                if roles[role].stat().st_size > plan["compressed_bytes_max"]:
                    raise RuntimeError(
                        "Before numerical archive hash compressed-byte cap: " + role
                    )
                if sha(roles[role]) != row["sha256"]:
                    raise ValueError("Bound numerical archive changed: " + role)
        sys.path.insert(0, str(source))
        import numpy as np
        from scipy import linalg

        import atomx.angular.csf_hamiltonian as angular
        import atomx.angular.mscheme as mscheme
        import atomx.physics.csf.matrix as matrix
        import atomx.physics.csf.overlap as overlap
        import atomx.physics.csf.tracking as tracking
        from atomx.config.csf import enumerate_jj_csfs, get_shared_orbital_basis
        from atomx.constants import HARTREE_CM1
        from atomx.methods import ci as ci_module
        from atomx.physics.csf.effective_one_body import CIReferenceBinding
        from atomx.physics.csf.levels import CILevel, ci_level_blocks
        from atomx.physics.csf.state_average import CIStateAverage
        from atomx.potentials.nuclear import make_nuclear_potential

        observe(linalg, "eigh", "complete_CI_eigh", caps["max_complete_CI_eigh"])
        observe(
            mscheme,
            "null_space",
            "angular_null_space",
            caps["max_angular_null_space_calls"],
        )
        observe(np.linalg, "svd", "tracking_SVD", caps["max_tracking_SVD_calls"])

        def projection_after(value, args, kwargs):
            work["projection_shapes"].append(
                dict(
                    raw=int(value[1].shape[0]),
                    CSFs=int(value[1].shape[1]),
                    symmetry=plain(args[2]),
                )
            )

        observe(
            overlap,
            "_projection",
            "public_projection",
            caps["max_projection_calls"],
            projection_after,
        )
        observe(tracking, "ci_subspace_overlaps", "tracker_signed_overlap", 3)
        observe(overlap, "ci_subspace_overlaps", "samepoint_signed_overlap", 1)
        observe(tracking.CIBasisTracking, "resolve", "tracker_resolve", 2)
        decoded_bytes, archive_loads = 0, 0

        def archive(role):
            nonlocal decoded_bytes, archive_loads
            archive_loads += 1
            plan = contract["array_access"][role]
            if archive_loads > caps["max_input_archive_loads"]:
                raise RuntimeError("Archive-load resource stop")
            # Bound zip metadata and selected uncompressed sizes before decode.
            with zipfile.ZipFile(roles[role]) as z:
                infos = z.infolist()
                if len(infos) > 256 or len({v.filename for v in infos}) != len(infos):
                    raise ValueError(
                        "Duplicate/oversized numerical archive field census"
                    )
                names = list(plan["fields"])
                if plan.get("basis_prefix"):
                    names += [
                        v.filename[:-4]
                        for v in infos
                        if v.filename.startswith("basis_")
                        and v.filename.endswith(".npy")
                    ]
                if len(names) > plan["fields_max"] or len(set(names)) != len(names):
                    raise RuntimeError("Selected complete archive field-count stop")
                selected_size = sum(
                    z.getinfo(name + ".npy").file_size for name in names
                )
                if selected_size > caps[
                    "max_input_decoded_bytes"
                ] - decoded_bytes + 16384 * len(names):
                    raise RuntimeError("Before decode selected uncompressed-byte stop")
            result = {}
            with np.load(roles[role], allow_pickle=False) as a:
                for name in names:
                    value = a[name]
                    spec = plan["fields"].get(name)
                    if spec and (
                        value.shape != tuple(spec["shape"])
                        or value.dtype.str != spec["dtype"]
                    ):
                        raise ValueError(
                            "Native input shape/dtype: " + role + "/" + name
                        )
                    if value.dtype.hasobject or not np.isfinite(value).all():
                        raise ValueError(
                            "Nonfinite/nonnumeric archive field: " + role + "/" + name
                        )
                    decoded_bytes += value.nbytes
                    if decoded_bytes > caps["max_input_decoded_bytes"]:
                        raise RuntimeError("Complete decoded input-byte stop")
                    result[name] = value.copy()
            write(
                output / ("input_" + role + "_decoded.json"),
                dict(
                    fields=[
                        dict(name=k, shape=v.shape, dtype=v.dtype.str, bytes=v.nbytes)
                        for k, v in result.items()
                    ],
                    cumulative_decoded_bytes=decoded_bytes,
                ),
            )
            return result

        baseline_arrays, point = archive("baseline_whole_CI"), archive("baseline_point")
        geometry = archive("baseline_geometry")
        radial, reference_arrays, metric_arrays = (
            archive("geometry_workspace"),
            archive("tracking_ci"),
            archive("screened_geometry"),
        )
        data = {
            k[6:]: v.item() if v.ndim == 0 else v.copy()
            for k, v in radial.items()
            if k.startswith("basis_")
        }
        check("workspace_ns123", data["ns"] != 123)
        keys = tuple(tuple(map(int, k)) for k in baseline_arrays["keys"])
        metric = {k: metric_arrays[f"s_{k}"] for k in fixed["kappas"]}
        waves = {
            k: (p.copy(), q.copy())
            for k, p, q in zip(
                keys, baseline_arrays["P"], baseline_arrays["Q"], strict=True
            )
        }
        reference_keys = tuple(tuple(map(int, k)) for k in radial["orbital_keys"])
        reference_waves = {
            k: (p.copy(), q.copy())
            for k, p, q in zip(reference_keys, radial["P"], radial["Q"], strict=True)
        }
        check(
            "baseline_point_keys_bytes",
            baseline_arrays["keys"].tobytes() != point["keys"].tobytes(),
        )
        for name in ("P", "Q"):
            check(
                "baseline_point_" + name + "_bytes",
                baseline_arrays[name].tobytes() != point[name].tobytes(),
            )
        check(
            "baseline_point_original_x_bytes",
            point["x"].tobytes() != geometry["x"].tobytes(),
        )
        for i, row in enumerate(read(roles["new_layout"])):
            p = geometry[f"projector_{i}"]
            kap = row["keys"][0][1]
            s = metric[kap]
            check(
                "projector_Gram/" + str(kap),
                np.max(abs(p.T @ s @ p - np.eye(p.shape[1]))),
                gates["saved_P_metric_Gram"],
            )
            c = np.column_stack([np.concatenate(waves[tuple(k)]) for k in row["keys"]])
            coord = point["x"][slice(*row["slice"])].reshape(row["shape"])
            reconstructed = p @ coord
            check(
                "literal31_Px_bytes/" + str(kap), reconstructed.tobytes() != c.tobytes()
            )
            check(
                "radial_Gram/" + str(kap),
                np.max(abs(c.T @ s @ c - np.eye(c.shape[1]))),
                gates["full_same_kappa_radial_Gram"],
            )
            leak = c - p @ (p.T @ s @ c)
            check(
                "original_P_leak/" + str(kap),
                np.max(np.einsum("ij,ij->j", leak, s @ leak)),
                gates["original_P_leak_S_squared"],
            )
        nucleus = make_nuclear_potential(70, "fermi", A=175)
        owner = read(roles["baseline_owner"])
        binding_radial = asdict(
            CIReferenceBinding.capture(
                [(*waves[k], k[1]) for k in sorted(keys)], {}, data, 70, 123, nucleus
            )
        )
        check(
            "baseline_owner_same_actual_radial_binding",
            plain(binding_radial) != owner["radial_binding"],
        )
        check(
            "baseline_owner_fixed_header",
            any(
                (
                    owner["schema_version"] != 1,
                    owner["representation"] != "dense-real-full-CSF",
                    owner["units"] != "Hartree",
                    owner["hamiltonian_terms"] != ["dirac-coulomb"],
                    owner["max_determinants"] != 2000,
                    owner["orbital_basis_keys"] != plain(keys),
                    owner["orbital_keys"] != plain(sorted(keys)),
                )
            ),
        )
        state = read(roles["baseline_state"])
        for label, name in (
            ("metric_sha256", "metric_sha256"),
            ("geometry_sha256", "geometry_sha256"),
            ("objective_identity", "objective_identity"),
        ):
            # StageB state identity field names are fixed in its bound JSON.
            actual_key = {
                "metric_sha256": "metric_sha256",
                "geometry_sha256": "geometry_sha256",
                "objective_identity": "objective_identity",
            }[name]
            check("baseline_" + label, state.get(actual_key) != fixed[name])

        specs = read(roles["prepared_models"])
        sectors = {tuple(v) for v in specs["selected_sectors"]}
        full, selected, indices, identities, config_maps = {}, {}, {}, {}, {}

        def signature(csf):
            s = angular.csf_signature(csf)
            return json.dumps(
                plain((s.subshells, s.coupling, csf.total_J2, csf.parity)),
                separators=(",", ":"),
            )

        for name in ("baseline", "expanded"):
            full[name], config_maps[name] = [], []
            for number, row in enumerate(specs[name + "_configs"]):
                start = len(full[name])
                added = enumerate_jj_csfs(row["config"])
                full[name].extend(added)
                config_maps[name].append(
                    dict(
                        NR_index=number,
                        family=row["family"],
                        configuration=row["config"],
                        parent_start=start,
                        parent_stop=len(full[name]),
                    )
                )
            indices[name] = [
                i for i, c in enumerate(full[name]) if (c.total_J2, c.parity) in sectors
            ]
            selected[name] = [full[name][i] for i in indices[name]]
            identities[name] = [signature(c) for c in selected[name]]
            expected_count = 1595 if name == "baseline" else 2574
            check(
                name + "/complete_parent",
                len(full[name]) != (3789 if name == "baseline" else 5741),
            )
            check(name + "/complete_selected", len(selected[name]) != expected_count)
            check(
                name + "/unique_full_signed_CSFs",
                len({signature(c) for c in full[name]}) != len(full[name]),
            )
            check(
                name + "/electron_number69", {c.n_electrons for c in full[name]} != {69}
            )
            check(
                name + "/same_all31_keys",
                tuple(sorted(get_shared_orbital_basis(selected[name]))) != keys,
            )
            for row in specs["counts"]["blocks"]:
                sector = row["twoJ"], row["parity"]
                census = row["baseline" if name == "baseline" else "candidate"]
                check(
                    name + "/complete_block/" + str(sector),
                    sum((c.total_J2, c.parity) == sector for c in selected[name])
                    != census["csf"],
                )
        check(
            "baseline_bound_canonical_signed_order",
            identities["baseline"] != read(roles["baseline_basis"])["signatures"],
        )
        check(
            "baseline_owner_complete_signed_CSF_identities",
            plain(
                [
                    (c.n_electrons, c.parity, angular.csf_signature(c).key)
                    for c in selected["baseline"]
                ]
            )
            != owner["CSFs"],
        )
        positions = {s: i for i, s in enumerate(identities["expanded"])}
        mapping = [positions[s] for s in identities["baseline"]]
        check("nested_exact_signed_basis_embedding", len(set(mapping)) != 1595)
        maps = {
            name: dict(
                configurations=config_maps[name],
                full_parent_signatures=[signature(c) for c in full[name]],
                selected_parent_indices=indices[name],
                selected_signatures=identities[name],
                selected_occupations=[
                    plain(angular.csf_signature(c).pattern) for c in selected[name]
                ],
            )
            for name in full
        }
        write(output / "complete_basis_and_configuration_maps.json", maps)
        write(
            output / "nested_embedding.json",
            dict(
                old_to_expanded=mapping,
                baseline_count=1595,
                expanded_count=2574,
                all_signed_basis_and_configuration_rows_preserved=True,
            ),
        )
        reference_csfs = [
            c
            for row in read(roles["reference_models"])["models"]["baseline"]["configs"]
            for c in enumerate_jj_csfs(row["config"])
        ]
        check("immutable_original_tracker531", len(reference_csfs) != 531)

        # Source representation controls before any H: exact pattern union
        # ordering and full common-core terms, never the signed-support shortcut.
        representation = []
        for row in specs["counts"]["blocks"]:
            sector = row["twoJ"], row["parity"]
            info = {}
            for name in selected:
                patterns = tuple(
                    sorted(
                        {
                            angular.csf_signature(c).pattern
                            for c in selected[name]
                            if (c.total_J2, c.parity) == sector
                        }
                    )
                )
                raw = sum(
                    dict(angular.pattern_m_counts(p)).get(sector[0], 0)
                    for p in patterns
                )
                union = tuple(k for k, _ in angular._union_offsets(patterns))
                closed = set.intersection(
                    *[{(n, k, q) for n, k, q in p if q == 2 * abs(k)} for p in patterns]
                )
                info[name] = dict(
                    raw=raw,
                    patterns=len(patterns),
                    union_keys=union,
                    common_closed_subshells=sorted(closed),
                    common_closed_electrons=sum(q for _, _, q in closed),
                )
                census = row["baseline" if name == "baseline" else "candidate"]
                check(name + "/exact_builder_raw/" + str(sector), raw != census["raw"])
                check(
                    name + "/exact_pattern_count/" + str(sector),
                    len(patterns) != census["patterns"],
                )
                check(
                    name + "/explicit_rawcap/" + str(sector),
                    raw,
                    caps["raw_determinants_per_block"],
                )
            check(
                "same_union_order/" + str(sector),
                info["baseline"]["union_keys"] != info["expanded"]["union_keys"],
            )
            check(
                "same_closed_core/" + str(sector),
                info["baseline"]["common_closed_subshells"]
                != info["expanded"]["common_closed_subshells"],
            )
            check(
                "deep_closed_core46/" + str(sector),
                info["baseline"]["common_closed_electrons"] != 46,
            )
            representation.append(dict(sector=sector, **info))
        write(output / "source_representation_controls.json", representation)

        def raw_groups(csfs, orbital_keys):
            labels = [
                (n, k, m)
                for n, k in orbital_keys
                for m in range(-2 * abs(k) + 1, 2 * abs(k), 2)
            ]
            result = {}
            for sector in sorted(sectors):
                masks = set()
                for c in csfs:
                    if (c.total_J2, c.parity) == sector:
                        masks.update(
                            mask
                            for mask, _ in angular.csf_magnetic_vector(
                                angular.csf_signature(c), sector[0], orbital_keys
                            )
                        )
                result[sector] = Counter(
                    tuple(
                        sorted(
                            (k, m)
                            for i, (_, k, m) in enumerate(labels)
                            if mask & (1 << i)
                        )
                    )
                    for mask in masks
                )
            return result

        groups = {name: raw_groups(selected[name], keys) for name in selected}
        groups["reference"] = raw_groups(reference_csfs, reference_keys)
        overlap_admission = []
        for bra, ket in (
            ("reference", "reference"),
            ("reference", "baseline"),
            ("reference", "expanded"),
            ("baseline", "expanded"),
        ):
            paircounts = {
                s: sum(
                    count * groups[ket][s][key] for key, count in groups[bra][s].items()
                )
                for s in sorted(sectors)
            }
            check(
                "full_signed_overlap_paircap/" + bra + "/" + ket,
                sum(paircounts.values()),
                caps["max_signed_overlap_pairs"],
            )
            if ket != "reference":
                check(
                    "full_signed_overlap_kinematic_bound/" + bra + "/" + ket,
                    sum(paircounts.values()),
                    caps["max_signed_compatible_pairs_sum"],
                )
            overlap_admission.append(
                dict(
                    bra=bra,
                    ket=ket,
                    compatible_pairs=sum(paircounts.values()),
                    blocks=[dict(sector=s, pairs=v) for s, v in paircounts.items()],
                )
            )
        write(output / "full_signed_overlap_admission.json", overlap_admission)

        def validate_ci(name, csfs, ci, h):
            n = len(csfs)
            check(
                name + "/whole_shape",
                h.shape != (n, n)
                or ci.state_energies.shape != (n,)
                or ci.mixing_coefficients.shape != (n, n),
            )
            check(
                name + "/Hermiticity",
                np.max(abs(h - h.T)),
                gates["matrix_hermiticity_Ha"],
            )
            c = ci.mixing_coefficients
            e = ci.state_energies
            check(
                name + "/all_root_residual",
                np.max(np.linalg.norm(h @ c - c * e, axis=0)),
                gates["all_complete_CI_residual_Ha"],
            )
            check(
                name + "/all_root_Gram",
                np.max(abs(c.T @ c - np.eye(n))),
                gates["all_complete_CI_Gram"],
            )
            _, blocks = ci_level_blocks(csfs, ci, require_full=True)
            for sector in sorted(sectors):
                rows = [
                    i for i, a in enumerate(csfs) if (a.total_J2, a.parity) == sector
                ]
                other = [
                    i for i, a in enumerate(csfs) if (a.total_J2, a.parity) != sector
                ]
                check(
                    name + "/cross_symmetry/" + str(sector),
                    np.max(abs(h[np.ix_(rows, other)])),
                    gates["cross_symmetry_zero_Ha"],
                )
            return blocks

        baseline_ci = ci_module.CIResult(
            baseline_arrays["energies"], baseline_arrays["mixing"]
        )
        baseline_h = baseline_arrays["H"]
        baseline_blocks = validate_ci(
            "baseline", selected["baseline"], baseline_ci, baseline_h
        )
        targets = read(roles["tracking_binding"])["targets"]
        expected_UIDs = {tuple(uid): label for label, uid in fixed["original_UIDs"]}
        check(
            "same_original_five_UID_selectors",
            {(t["twoJ"], t["parity"], t["models"]["DC"]["local_root"]) for t in targets}
            != set(expected_UIDs),
        )
        average = CIStateAverage(
            tuple(
                (t["twoJ"], t["parity"], t["models"]["DC"]["local_root"], 1.0)
                for t in targets
            )
        )
        tracker = tracking.CIBasisTracking(
            average,
            reference_csfs,
            ci_module.CIResult(
                reference_arrays["energies_Ha"], reference_arrays["mixing"]
            ),
            reference_waves,
            metric,
            max_determinant_pairs=caps["max_signed_overlap_pairs"],
        )

        def report(name, csfs, ci, blocks):
            weights, tracked = tracker.resolve(csfs, ci, waves, metric)
            write(
                output / (name + "_tracking.json"),
                dict(
                    weights=weights,
                    tracked=tracked,
                    original531_reference_not_rebased=True,
                ),
            )
            check(
                name + "/five_equal_weights",
                not (
                    np.count_nonzero(weights) == 5
                    and np.array_equal(weights[weights > 0], np.repeat(0.2, 5))
                ),
            )
            matches = {}
            for row in tracked:
                if (
                    len(row["reference_levels"]) != 1
                    or len(row["selected_levels"]) != 1
                ):
                    raise ValueError(
                        "Original rank-one UID cluster boundary changed; preserve complete native cluster rejection"
                    )
                uid = tuple(row["reference_levels"][0])
                chosen = tuple(row["selected_levels"][0])
                if uid in matches or uid not in expected_UIDs or uid[:2] != chosen[:2]:
                    raise ValueError("Changed/duplicate physical UID")
                matches[uid] = row
            check(name + "/complete_UIDs", set(matches) != set(expected_UIDs))
            rows = []
            for label, uid_list in fixed["original_UIDs"]:
                uid = tuple(uid_list)
                choice = matches[uid]["selected_levels"][0]
                column = int(blocks[choice[0], choice[1]][choice[2]])
                rows.append(
                    dict(
                        label=label,
                        UID=uid,
                        selected_level=choice,
                        column=column,
                        weight=float(weights[column]),
                        energy_Ha=float(ci.state_energies[column]),
                        tracker=matches[uid],
                    )
                )
            lower = rows[0]["energy_Ha"]
            for row in rows:
                nominal = fixed["nominal_NIST_cm1"][row["label"]]
                row.update(
                    excitation_cm1=(row["energy_Ha"] - lower) * HARTREE_CM1,
                    NIST_nominal_cm1=nominal,
                    NIST_uncertainty_cm1=None,
                )
                row["nominal_gap_cm1"] = row["excitation_cm1"] - nominal
                if name == "expanded":
                    previous = {v["label"]: v for v in points["baseline"]["rows"]}
                    du = row["energy_Ha"] - previous[row["label"]]["energy_Ha"]
                    dl = lower - previous["6s_1/2"]["energy_Ha"]
                    row.update(
                        delta_upper_Ha=du,
                        delta_lower_Ha=dl,
                        delta_upper_minus_lower_transition_cm1=(du - dl) * HARTREE_CM1,
                    )
            by_label = {r["label"]: r for r in rows}
            multiplets = {}
            for term, a, b in (("5d", "5d_3/2", "5d_5/2"), ("6p", "6p_1/2", "6p_3/2")):
                lo, hi = by_label[a], by_label[b]
                ga, gb = lo["UID"][0] + 1, hi["UID"][0] + 1
                multiplets[term] = dict(
                    centroid_cm1=(ga * lo["excitation_cm1"] + gb * hi["excitation_cm1"])
                    / (ga + gb),
                    splitting_cm1=hi["excitation_cm1"] - lo["excitation_cm1"],
                )
                if name == "expanded":
                    for v in ("centroid_cm1", "splitting_cm1"):
                        multiplets[term]["delta_" + v] = (
                            multiplets[term][v]
                            - points["baseline"]["multiplets"][term][v]
                        )
            result = dict(
                rows=rows,
                multiplets=multiplets,
                EOL_reported_Ha=float(weights @ ci.state_energies),
                nominal_four_gap_RMS_cm1=math.sqrt(
                    sum(r["nominal_gap_cm1"] ** 2 for r in rows[1:]) / 4
                ),
                original_stationarity_passed=False,
                expanded_MCDHF_optimized=False,
            )
            if name == "baseline":
                expected = fixed["baseline"]["rows"]
                oldtrack = read(roles["baseline_tracking"])
                check(
                    "baseline_exact_StageB_UID_weights",
                    plain(weights) != oldtrack["weights"]
                    or plain([r["selected_level"] for r in rows])
                    != [r["selected_level"] for r in expected],
                )
                check(
                    "baseline_exact_StageB_target_energy_bytes",
                    any(
                        float(r["energy_Ha"]).hex() != float(e["energy_Ha"]).hex()
                        for r, e in zip(rows, expected, strict=True)
                    ),
                )
                check(
                    "baseline_exact_StageB_reported_EOL",
                    result["EOL_reported_Ha"].hex()
                    != float(fixed["baseline"]["EOL_reported_Ha"]).hex(),
                )
                check(
                    "baseline_exact_StageB_principal_tracking",
                    plain(tracked) != oldtrack["tracked"],
                )
            points[name] = result
            write(output / (name + "_conditional_point.json"), result)
            return rows

        baseline_rows = report(
            "baseline", selected["baseline"], baseline_ci, baseline_blocks
        )
        progress("baseline_complete_bound_reference_verified_before_one_new_H")

        def connected_after(value, args, kwargs):
            count = len(value)
            check(
                "connected_pairs_per_completed_block/"
                + str(completed["connected_pairs"]),
                count,
                caps["max_connected_pairs_per_block"],
            )
            work["connected_pairs_total"] += count
            check(
                "connected_pairs_cumulative",
                work["connected_pairs_total"],
                caps["max_connected_pairs_total"],
            )

        observe(angular, "_connected_pairs", "connected_pairs", 5, connected_after)

        def table_after(value, args, kwargs):
            termkeys, table, pairs = value
            work["coefficient_table_nnz_total"] += int(table.nnz)
            work["coefficient_tables"].append(
                dict(
                    raw_determinants=len(args[0]),
                    keys=len(termkeys),
                    shape=table.shape,
                    nnz=int(table.nnz),
                    pairs=len(pairs),
                    CSR_bytes=int(
                        table.data.nbytes + table.indices.nbytes + table.indptr.nbytes
                    ),
                    pair_bytes=int(pairs.nbytes),
                )
            )
            write(output / "completed_angular_tables.json", work["coefficient_tables"])
            check(
                "coefficient_table_single_nnz/" + str(completed["coefficient_table"]),
                table.nnz,
                caps["max_coefficient_table_nnz_per_block"],
            )
            check(
                "coefficient_table_cumulative_nnz",
                work["coefficient_table_nnz_total"],
                caps["max_coefficient_table_nnz_total"],
            )

        observe(
            angular,
            "_coefficient_table",
            "coefficient_table",
            caps["max_coefficient_tables"],
            table_after,
        )

        def hS_after(value, args, kwargs):
            kappa = int(args[1])
            work["hS_kappas"].append(kappa)
            h, s = value
            save(
                "native_raw_hS_" + str(kappa) + ".npz",
                raw_h=h,
                S=s,
                kappa=np.array(kappa),
            )
            check(
                "native_hS_same_original_metric/" + str(kappa),
                s.tobytes() != metric[kappa].tobytes(),
            )

        observe(
            matrix,
            "build_dirac_matrices",
            "native_hS",
            caps["max_native_hS_builds"],
            hS_after,
        )
        observe(matrix, "_dirac_blocks", "hS_getter", caps["max_hS_requests"])
        original_evaluator = matrix._radial_evaluator
        radial_evaluators = []

        def counted_evaluator(*args, **kwargs):
            evaluator = original_evaluator(*args, **kwargs)

            def evaluate(key):
                attempted["radial_key_evaluation"] += 1
                if (
                    attempted["radial_key_evaluation"]
                    > caps["max_radial_key_evaluations"]
                ):
                    raise RuntimeError("Complete radial-key work stop; no term pruning")
                result = evaluator(key)
                completed["radial_key_evaluation"] += 1
                work["radial_keys"] = completed["radial_key_evaluation"]
                return result

            evaluate.cache_info = evaluator.cache_info
            evaluate.pair_cache_info = evaluator.pair_cache_info
            radial_evaluators.append(evaluator)
            return evaluate

        restored.append((matrix, "_radial_evaluator", original_evaluator))
        matrix._radial_evaluator = counted_evaluator

        def H_after(h, args, kwargs):
            save(
                "expanded_complete_H.npz",
                H=h,
                keys=np.asarray(keys),
                P=baseline_arrays["P"],
                Q=baseline_arrays["Q"],
            )
            embedded = h[np.ix_(mapping, mapping)]
            write(
                output / "nested_H_control.json",
                dict(
                    exact_bytes_equal=embedded.tobytes() == baseline_h.tobytes(),
                    maximum_absolute_difference_Ha=float(
                        np.max(abs(embedded - baseline_h))
                    ),
                    baseline_H_rebuilt=False,
                    all1595_rows_and_columns=True,
                    expanded_CI_not_yet_diagonalized=True,
                ),
            )
            check(
                "expanded_exact_nested1595_H_bytes",
                embedded.tobytes() != baseline_h.tobytes(),
            )

        observe(matrix, "build_ci_matrix", "H", caps["max_complete_H_builds"], H_after)
        observe(ci_module, "ci_solve", "CI", caps["max_complete_CI_calls"])
        new_ci = ci_module.ci_solve(
            selected["expanded"],
            waves,
            keys,
            data,
            70,
            123,
            solver="dense",
            n_states=None,
            nuclear_potential=nucleus,
            include_breit=False,
            max_determinants=2659,
            retain_hamiltonian=True,
        )
        new_h = new_ci.hamiltonian_for(
            selected["expanded"],
            waves,
            keys,
            data,
            70,
            123,
            nuclear_potential=nucleus,
            max_determinants=2659,
        )
        save(
            "expanded_whole_CI.npz",
            H=new_h,
            energies=new_ci.state_energies,
            mixing=new_ci.mixing_coefficients,
            keys=np.asarray(keys),
            P=baseline_arrays["P"],
            Q=baseline_arrays["Q"],
            old_to_expanded=np.asarray(mapping),
            selected_parent_indices=np.asarray(indices["expanded"]),
        )
        new_owner = new_ci.hamiltonian_snapshot.identity_payload()
        write(output / "expanded_owner.json", new_owner)
        check(
            "same_baseline_expanded_radial_owner",
            plain(new_owner["radial_binding"]) != owner["radial_binding"],
        )
        check(
            "all7_native_hS_producers",
            set(work["hS_kappas"]) != set(fixed["kappas"])
            or len(work["hS_kappas"]) != 7,
        )
        new_blocks = validate_ci("expanded", selected["expanded"], new_ci, new_h)
        for sector in sorted(sectors):
            oldroots = baseline_ci.state_energies[baseline_blocks[sector]]
            roots = new_ci.state_energies[new_blocks[sector]][: len(oldroots)]
            check(
                "expanded_all_old_ordered_Ritz/" + str(sector),
                max(0.0, float(np.max(roots - oldroots))),
                gates["inherited_ordered_Ritz_increase_Ha"],
            )
        new_rows = report("expanded", selected["expanded"], new_ci, new_blocks)
        # Full signed public contraction. Do not choose roots by this diagnostic,
        # energy or NIST. Original531 tracker already made all UID selections.
        bra_levels = tuple(CILevel(*r["selected_level"]) for r in baseline_rows)
        ket_levels = tuple(
            CILevel(j, p, i)
            for (j, p), columns in new_blocks.items()
            for i in range(len(columns))
        )
        signed = overlap.ci_subspace_overlaps(
            selected["baseline"],
            baseline_ci,
            waves,
            bra_levels,
            selected["expanded"],
            new_ci,
            waves,
            ket_levels,
            metric,
            max_determinant_pairs=caps["max_signed_overlap_pairs"],
        )
        save(
            "complete_signed_samepoint_overlap.npz",
            overlap=signed,
            baseline_levels=np.asarray([r["selected_level"] for r in baseline_rows]),
            expanded_levels=np.asarray([(s.J2, s.parity, s.root) for s in ket_levels]),
        )
        ket_positions = {(s.J2, s.parity, s.root): i for i, s in enumerate(ket_levels)}
        write(
            output / "samepoint_target_overlap_diagnostics.json",
            dict(
                no_new_selection=True,
                complete_candidate_roots=True,
                cluster_policy="Unchanged public overlap requires complete exact clusters; original531 tracker remains authoritative.",
                states=[
                    dict(
                        UID=a["UID"],
                        baseline_selected=a["selected_level"],
                        expanded_selected=b["selected_level"],
                        signed_overlap=float(
                            signed[i, ket_positions[tuple(b["selected_level"])]]
                        ),
                        squared_overlap=float(
                            signed[i, ket_positions[tuple(b["selected_level"])]] ** 2
                        ),
                    )
                    for i, (a, b) in enumerate(
                        zip(baseline_rows, new_rows, strict=True)
                    )
                ],
            ),
        )
        for name, count in (
            ("H", 1),
            ("CI", 1),
            ("complete_CI_eigh", 5),
            ("coefficient_table", 5),
            ("tracker_resolve", 2),
        ):
            check("exact_completed_producer/" + name, completed[name] != count)
        work["native_radial_caches"] = [
            dict(
                evaluation=v.cache_info()._asdict(),
                pair_potential=v.pair_cache_info()._asdict(),
            )
            for v in radial_evaluators
        ]
        work["native_caches"] = dict(
            block_structure=angular._block_structure.cache_info()._asdict(),
            CSF_sparse_projection=angular._block_csf_sparse_matrix.cache_info()._asdict(),
            span_verification=angular._verify_cached.cache_info()._asdict(),
        )
        for name, module in list(sys.modules.items()):
            if name == "atomx" or name.startswith("atomx."):
                filename = getattr(module, "__file__", None)
                if filename and not Path(filename).resolve().is_relative_to(
                    source.resolve()
                ):
                    raise ValueError(
                        "Runtime Atomx module escaped frozen source: " + name
                    )
        metadata()  # SOURCE/JSON-only invariant recheck; no numerical rehash.
        write(
            output / "result.json",
            dict(
                status="conditional_current31_5p_double_RCI_completed",
                all_declared_controls_passed=True,
                checks=checks,
                attempted=attempted,
                completed=completed,
                work=work,
                archives=archives,
                points=points,
                source_map_sha256=contract["source_map_sha256"],
                contract_sha256=sha(HERE / "contract.json"),
                driver_sha256=sha(__file__),
                execution_binding_sha256=sha(binding_path),
                baseline_complete_arrays_bound_at=contract["input_roles"][
                    "baseline_whole_CI"
                ],
                baseline_H_rebuilt=False,
                new_complete_H_builds=1,
                new_complete_CI_calls=1,
                original_stationarity_passed=False,
                expanded_stationarity_evaluated=False,
                expanded_MCDHF_optimized=False,
                full_spectrum_certified=False,
                complete_correlation_certified=False,
                experimental_gate_passed=False,
                nominal_improvement_required=False,
                conditional_fixed_orbital_only=True,
                input_decoded_bytes=decoded_bytes,
                input_archive_loads=archive_loads,
                elapsed_uncontrolled_s=time.monotonic() - started,
                peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                timing_claim_valid=False,
                actual_affinity=sorted(os.sched_getaffinity(0)),
            ),
        )
        progress("complete")
        print(
            json.dumps(
                dict(
                    status="conditional_current31_5p_double_RCI_completed",
                    actual=str(output),
                    new_H=completed["H"],
                    new_CI=completed["CI"],
                )
            ),
            flush=True,
        )
    except BaseException as error:
        write(
            output / "failure.json",
            dict(
                error_type=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
                checks=checks,
                attempted=attempted,
                completed=completed,
                work=work,
                archives=archives,
                points=points,
                source_map_sha256=contract["source_map_sha256"],
                contract_sha256=sha(HERE / "contract.json"),
                driver_sha256=sha(__file__),
                execution_binding_sha256=sha(binding_path),
                original_stationarity_passed=False,
                expanded_MCDHF_optimized=False,
                scientific_success_claim=False,
                no_retry=True,
                elapsed_uncontrolled_s=time.monotonic() - started,
                peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            ),
        )
        raise
    finally:
        done.set()
        signal.setitimer(signal.ITIMER_REAL, 0)
        for module, name, original in reversed(restored):
            setattr(module, name, original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-binding", type=Path)
    args = parser.parse_args()
    if args.execution_binding:
        execute(args.execution_binding)
    else:
        c, _, _ = metadata()
        print(
            json.dumps(
                dict(
                    status=c["status"],
                    source_only=True,
                    science_imports=0,
                    NPZ_reads=0,
                    NPZ_hashes=0,
                    physical_calls=0,
                    exact_root_seal_required=True,
                    contract_sha256=sha(HERE / "contract.json"),
                    driver_sha256=sha(__file__),
                )
            )
        )


if __name__ == "__main__":
    main()
