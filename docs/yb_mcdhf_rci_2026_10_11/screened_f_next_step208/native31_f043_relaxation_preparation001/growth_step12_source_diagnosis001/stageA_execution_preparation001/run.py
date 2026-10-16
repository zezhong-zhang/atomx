"""Offline chronological f043 secant replay; no physical producer or optimizer."""
import argparse
import ast
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
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_sections(source):
    """Extract exact frozen numerical AST, without importing its module."""
    tree = ast.parse(source)
    top = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    optimizer = top["optimize_orbitals"]
    nested = {n.name: n for n in optimizer.body if isinstance(n, ast.FunctionDef)}
    loop = next(n for n in optimizer.body if isinstance(n, ast.For)
                and ast.unparse(n.iter) == "range(max_iter + 1)")
    start = next(i for i, n in enumerate(loop.body)
                 if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "q")
    end = next(i for i, n in enumerate(loop.body[start:], start)
               if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "step")
    transport = next(i for i, n in enumerate(loop.body)
                     if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "transported")
    return dict(functions=[top["_residual_metric_sha256"], top["_polar_retraction"],
                          nested["tangent"], nested["spectral_precondition"], nested["retract"]],
                direction=loop.body[start:end], update=loop.body[transport:])


def run(binding_path):
    contract = load(HERE / "contract.json")
    binding = load(binding_path)
    expected = dict(execution_authorized_by_root=True, driver_sha256=sha(__file__),
                    contract_sha256=sha(HERE / "contract.json"),
                    preparation_sha256=sha(HERE / "preparation.json"),
                    CPU=31, threads=1, AS_GiB=1, inner_seconds=30, outer_seconds=45,
                    max_small_Gram_eigh=7, max_NPZ_loads=14,
                    max_H=0, max_CI=0, max_gradient=0, max_Dirac=0, max_optimizer=0)
    if any(binding.get(k) != v for k, v in expected.items()):
        raise ValueError("Separate exact root execution binding required")
    prep = load(HERE / "preparation.json")
    if (prep["all_source_only_checks_passed"] is not True
            or prep["driver_sha256"] != expected["driver_sha256"]
            or prep["contract_sha256"] != expected["contract_sha256"]):
        raise ValueError("Changed prepared consumer")
    if any(n == "numpy" or n.startswith("numpy.") or n == "atomx"
           or n.startswith("atomx.") for n in sys.modules):
        raise ValueError("Fresh process required")
    if any(os.environ.get(k) != "1" for k in contract["resources"]["thread_environment"]):
        raise ValueError("Single-thread environment required")
    available = sorted(os.sched_getaffinity(0))
    if 31 not in available or os.getloadavg()[0] > 32:
        raise ValueError("Host admission rejected")
    for row in contract["metadata_inputs"].values():
        if sha(REPO / row["path"]) != row["sha256"]:
            raise ValueError("Changed frozen source/metadata: " + row["path"])
    output = HERE / "actual"
    output.mkdir(exist_ok=False)
    os.sched_setaffinity(0, {31})
    maximum = 2**30
    resource.setrlimit(resource.RLIMIT_AS, (maximum, maximum))

    def timeout(*_):
        raise TimeoutError("30s offline cap")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(30)
    started = time.monotonic()
    checks, records, history_events, decoded = [], [], [], []
    counters = dict(NPZ_loads=0, fields_decoded=0, decoded_bytes=0,
                    archive_bytes_hashed=0, small_Gram_eigh_attempted=0,
                    small_Gram_eigh_completed=0, direction_replays=0,
                    transported_pairs=0, admitted_new_pairs=0, two_loop_pair_visits=0,
                    H=0, CI=0, new_gradient=0, Dirac=0, optimizer=0)
    status, error, summary, namespace = "not_started", None, None, {}
    try:
        import numpy as np

        def check(name, value, limit=0.):
            row = dict(name=name, error=float(value), limit=limit,
                       passed=bool(np.isfinite(value) and value <= limit))
            checks.append(row)
            if not row["passed"]:
                raise ValueError("Offline replay gate: " + name)

        def fields(index):
            row = contract["points"][index]
            path = REPO / row["array_path"]
            size = path.stat().st_size
            if size > 64 * 2**20 or counters["archive_bytes_hashed"] + size > 896 * 2**20:
                raise ValueError("Before archive byte cap; no model/data pruning")
            if sha(path) != row["array_sha256"]:
                raise ValueError("Changed archived point")
            counters["archive_bytes_hashed"] += size
            if counters["NPZ_loads"] >= 14:
                raise ValueError("Before NPZ open cap")
            counters["NPZ_loads"] += 1
            result = {}
            with np.load(path, allow_pickle=False) as archive:
                for name, spec in row["fields"].items():
                    if counters["fields_decoded"] >= 94:
                        raise ValueError("Before decoded field cap")
                    value = archive[name]
                    if value.shape != tuple(spec["shape"]) or value.dtype != np.dtype(spec["dtype"]):
                        raise ValueError("Native dtype/shape changed: " + name)
                    if not np.isfinite(value).all():
                        raise ValueError("Nonfinite archived geometry/gradient")
                    if counters["decoded_bytes"] + value.nbytes > 8 * 2**20:
                        raise ValueError("Before decoded byte cap")
                    result[name] = value.copy()
                    counters["fields_decoded"] += 1
                    counters["decoded_bytes"] += value.nbytes
                    decoded.append(dict(point=index, field=name, shape=list(value.shape),
                                        dtype=value.dtype.str, bytes=value.nbytes))
            return result

        source = (REPO / contract["metadata_inputs"]["optimizer_source"]["path"]).read_text()
        sections = source_sections(source)
        namespace = dict(np=np, hashlib=hashlib)

        def compiled(nodes, name):
            return compile(ast.Module(body=nodes, type_ignores=[]), name, "exec")

        exec(compiled(sections["functions"], "exact_f043_numerical_functions"), namespace)
        direction_code = compiled(sections["direction"], "exact_f043_two_loop_direction")
        update_code = compiled(sections["update"], "exact_f043_secant_transport_admission")
        eigh = np.linalg.eigh

        def small_eigh(value, *args, **kwargs):
            if counters["small_Gram_eigh_attempted"] >= 7 or not (value.ndim == 2
                    and value.shape[0] == value.shape[1] and value.shape[0] <= 7):
                raise ValueError("Before small polar Gram eigen cap")
            counters["small_Gram_eigh_attempted"] += 1
            result = eigh(value, *args, **kwargs)
            counters["small_Gram_eigh_completed"] += 1
            return result

        np.linalg.eigh = small_eigh
        baseline = fields(0)
        rows = load(REPO / contract["metadata_inputs"]["layout"]["path"])
        layout = tuple((tuple(map(tuple, r["keys"])), baseline[f"projector_{i}"],
                        tuple(r["shape"]), slice(*r["slice"])) for i, r in enumerate(rows))
        blocks = tuple(tuple(b) for b in (baseline[f"inverse_blocks_{i}"] for i in range(7)))
        namespace.update(layout=layout, inverse_diagonal=baseline["inverse_diagonal"],
                         inverse_blocks=blocks, history=[], history_size=10)
        namespace["precondition"] = namespace["spectral_precondition"]
        metric = namespace["_residual_metric_sha256"](layout, baseline["inverse_diagonal"], blocks)
        check("original_metric_byte_fingerprint", metric != contract["fixed"]["metric_sha256"])
        keys = tuple(map(tuple, baseline["keys"].tolist()))
        check("31_keys_and_complete_group_order", len(keys) != 31
              or set(keys) != {k for group, *_ in layout for k in group})
        tangent, precondition = namespace["tangent"], namespace["spectral_precondition"]
        previous = None
        for index in range(13):
            point = baseline if index == 0 else fields(index)
            check("same_ordered31keys_" + str(index), not np.array_equal(point["keys"], baseline["keys"]))
            x = point["x"]
            radial = dict(zip(keys, point["retained_radial"], strict=True))
            raw = np.zeros_like(x)
            for group, projector, shape, where in layout:
                raw[where] = (projector.T @ np.column_stack([radial[k] for k in group])).ravel()
                check("coordinate_Gram_" + str(index) + str(group[0][1]),
                      np.max(abs(x[where].reshape(shape).T @ x[where].reshape(shape) - np.eye(shape[1]))), 1e-10)
            g = tangent(x, raw)
            pg = tangent(x, precondition(g))
            check("source_raw_coordinate_" + str(index), np.max(abs(raw - point["raw_coordinate_gradient"])), 1e-10)
            check("source_tangent_coordinate_" + str(index), np.max(abs(g - point["tangent_gradient"])), 1e-10)
            check("source_pg_" + str(index), np.max(abs(pg - point["preconditioned_gradient"])), 1e-10)
            manifest = load(REPO / contract["points"][index]["manifest_path"])
            check("source_residual_scalar_" + str(index), abs(float(np.linalg.norm(pg))
                  - manifest["diagnostics"]["orbital_residual"]), 1e-10)
            check("source_indicator_scalar_" + str(index), abs(.5 * abs(float(g @ pg))
                  - manifest["diagnostics"]["estimated_energy_error"]), 1e-10)
            namespace.update(x=x, g=g, preconditioned_gradient=pg)
            if previous is not None:
                namespace.update(old_x=previous[0], old_g=previous[1])
                old_history = list(namespace["history"])
                event = dict(point=index, before_pair_count=len(old_history), transported=[])
                for s, y, rho in old_history:
                    st, yt = tangent(x, s), tangent(x, y)
                    sy = float(st @ yt)
                    event["transported"].append(dict(previous_rho=float(rho), sTy=sy,
                        norm_s=float(np.linalg.norm(st)), norm_y=float(np.linalg.norm(yt)),
                        retained=bool(sy > 1e-12 * np.linalg.norm(st) * np.linalg.norm(yt))))
                counters["transported_pairs"] += len(old_history)
                exec(update_code, namespace)
                curvature, s, y = namespace["curvature"], namespace["s"], namespace["y"]
                admitted = curvature > 1e-12 * np.linalg.norm(s) * np.linalg.norm(y)
                counters["admitted_new_pairs"] += int(admitted)
                event.update(new_curvature=float(curvature), new_norm_s=float(np.linalg.norm(s)),
                             new_norm_y=float(np.linalg.norm(y)), new_pair_admitted=bool(admitted),
                             after_pair_count=len(namespace["history"]))
                history_events.append(event)
            pair_count = len(namespace["history"])
            counters["two_loop_pair_visits"] += 2 * pair_count
            exec(direction_code, namespace)
            counters["direction_replays"] += 1
            next_manifest = load(REPO / contract["points"][index + 1]["manifest_path"])
            ls = next_manifest["diagnostics"]["accepted_line_search"]
            check("original_primary_attempt_" + str(index + 1), ls["attempt"] != 0
                  or ls["stable_callback_used"] is not False)
            check("actual_original_slope_" + str(index + 1),
                  abs(float(namespace["derivative"]) - ls["directional_derivative"]), 1e-10)
            d, sd = namespace["direction"], -pg
            correction = d - sd
            records.append(dict(point=index, pair_count_before_direction=pair_count,
                pair_count_after_direction=len(namespace["history"]), slope=float(namespace["derivative"]),
                direction_norm=float(np.linalg.norm(d)), steepest_norm=float(np.linalg.norm(sd)),
                original_metric_steepest_slope=float(g @ sd), correction_norm=float(np.linalg.norm(correction)),
                signed_SD_correction_cross=float(sd @ correction),
                direction_norm_squared_identity=float(sd @ sd + correction @ correction + 2 * sd @ correction),
                angle_cosine=float(d @ sd / (np.linalg.norm(d) * np.linalg.norm(sd)))))
            previous = x, g
        check("chronological_pair_work_caps", counters["transported_pairs"] > 65
              or counters["two_loop_pair_visits"] > 150 or counters["admitted_new_pairs"] > 12)
        direction = namespace["direction"]
        ls = load(REPO / contract["points"][13]["manifest_path"])["diagnostics"]["accepted_line_search"]
        predicted = namespace["retract"](namespace["x"] + ls["step"] * direction)
        endpoint = fields(13)["x"]
        check("original_step13_polar_endpoint", np.max(abs(predicted - endpoint)), 1e-12)
        final_history = namespace["history"]
        cross = [[float(s @ y) for _, y, _ in final_history] for s, _, _ in final_history]
        parts = []
        g = namespace["g"]
        for group, _, shape, where in layout:
            dm, gm = direction[where].reshape(shape), g[where].reshape(shape)
            parts.extend(dict(key=list(key), direction_norm_squared=float(dm[:, j] @ dm[:, j]),
                              signed_g_dot_direction=float(gm[:, j] @ dm[:, j])) for j, key in enumerate(group))
        check("all31_signed_slope_partition", abs(sum(p["signed_g_dot_direction"] for p in parts)
              - namespace["derivative"]), 1e-10)
        transported = tangent(endpoint, direction)
        norm = np.linalg.norm(transported)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Transported direction is zero; no replacement")
        file = output / "reconstructed_direction.npz"
        with file.open("xb") as stream:
            np.savez_compressed(stream, x12=namespace["x"], g12=g, direction12=direction,
                pg12=namespace["preconditioned_gradient"], predicted_x13=predicted,
                literal_x13=endpoint, transported_direction13=transported,
                normalized_transported_direction13=transported / norm)
        summary = dict(direction_array_sha256=sha(file), final_pair_count=len(final_history),
                       full_signed_sTy_cross_matrix=cross, all31_slope_partitions=parts,
                       transported_direction13_norm=float(norm), future_stageB_qualified=False)
        status = "offline_original_history_slope_and_endpoint_reproduced"
    except BaseException as failure:
        status = "first_offline_failure_preserved_no_retry"
        error = dict(type=type(failure).__name__, message=str(failure), traceback=traceback.format_exc())
        if namespace.get("np") is not None:
            forensic = {k: namespace[k] for k in ("x", "g", "direction", "preconditioned_gradient")
                        if k in namespace}
            if forensic:
                file = output / "failed_replay_coordinates.npz"
                with file.open("xb") as stream:
                    namespace["np"].savez_compressed(stream, **forensic)
                error["derived_failure_array_sha256"] = sha(file)
    finally:
        signal.alarm(0)
        write(output / "result.json", dict(status=status, error=error, checks=checks,
            all_declared_replay_controls_passed=error is None and all(r["passed"] for r in checks),
            counters=counters, decoded_fields=decoded, history_events=history_events,
            direction_records=records, summary=summary, source_sha256=contract["source_sha256"],
            metric_sha256=contract["fixed"]["metric_sha256"], driver_sha256=sha(__file__),
            contract_sha256=sha(HERE / "contract.json"), binding_sha256=sha(binding_path),
            CPU=31, threads=1, AS_GiB=1, elapsed_uncontrolled_s=time.monotonic() - started,
            peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            physical_derivative_or_stationarity_certified=False, new_optimizer_run=False,
            no_automatic_stageB_or_continuation=True))
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-binding", type=Path, required=True)
    run(parser.parse_args().execution_binding)
