"""Closed-array retained-H contraction discriminator; SOURCE-only default."""

import argparse
import hashlib
import json
import os
import resource
import signal
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "AGENTS.md").is_file())


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def metadata(*, array_hashes=False):
    c = read(HERE / "contract.json")
    paths = {k: ROOT / v["path"] for k, v in c["input_roles"].items()}
    assert (
        sum(
            p.stat().st_size
            for k, p in paths.items()
            if c["input_roles"][k]["kind"] == "metadata"
        )
        <= c["resources"]["metadata_bytes_max"]
    )
    for k, p in paths.items():
        row = c["input_roles"][k]
        if row["kind"] == "metadata" or array_hashes:
            assert sha(p) == row["sha256"], k
        else:
            with zipfile.ZipFile(p) as z:
                observed = dict(
                    file_bytes=p.stat().st_size,
                    entries=[
                        dict(
                            filename=i.filename,
                            uncompressed_bytes=i.file_size,
                            compressed_bytes=i.compress_size,
                        )
                        for i in z.infolist()
                    ],
                )
            assert observed == c["zip_directory_census"][k], k
    receipt, result, failure = [
        read(paths[k]) for k in ("parent_receipt", "parent_result", "parent_failure")
    ]
    assert (
        receipt["status"] == "foreground_closed"
        and receipt["exit_code"] == 1
        and receipt["source_and_inputs_unchanged"]
    )
    assert (
        receipt["result_sha256"] == c["input_roles"]["parent_result"]["sha256"]
        and receipt["failure_sha256"] == c["input_roles"]["parent_failure"]["sha256"]
    )
    assert (
        len(result["checks"]) == 745
        and sum(r["passed"] for r in result["checks"]) == 743
    )
    assert not result["all_declared_controls_passed"]
    assert [r["name"] for r in failure["checks"] if not r["passed"]] == c[
        "original_gates"
    ]["ordinary_FD_failures"]
    assert result["source_map_sha256"] == c["source_map_sha256"]
    assert {k: result[k] for k in c["baseline_identities"]} == c["baseline_identities"]
    manifests = {r["path"]: r["sha256"] for r in result["archives"]}
    for k in ("baseline_arrays", "trial_arrays"):
        row = c["input_roles"][k]
        assert manifests[row["parent_manifest_path"]] == row["sha256"]
    source = ROOT / c["source_archive"]
    identity = read(paths["source_identity"])
    assert (
        identity["source_map_sha256"] == c["source_map_sha256"]
        and len(identity["all_atomx_python_sha256"]) == 356
    )
    assert all(
        sha(source / k.removeprefix("src/")) == v
        for k, v in identity["all_atomx_python_sha256"].items()
    )
    return c, paths


def execute(binding_path):
    c, paths = metadata()
    b = read(binding_path)
    assert b["execution_authorized_by_root"] is True
    assert b["driver_sha256"] == sha(__file__) and b["contract_sha256"] == sha(
        HERE / "contract.json"
    )
    assert b["preparation_sha256"] == sha(HERE / "preparation.json")
    assert b["input_roles"] == c["input_roles"] and b["resources"] == c["resources"]
    prep = read(HERE / "preparation.json")
    assert (
        prep["all_SOURCE_checks_passed"]
        and prep["driver_sha256"] == sha(__file__)
        and prep["contract_sha256"] == sha(HERE / "contract.json")
    )
    caps = c["resources"]
    assert sys.executable == caps["interpreter_literal"]
    assert all(
        os.environ.get(k) == "1"
        for k in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMBA_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
        )
    )
    assert caps["CPU"] in os.sched_getaffinity(0)
    os.sched_setaffinity(0, {caps["CPU"]})
    maximum = int(caps["AS_GiB"] * 2**30)
    resource.setrlimit(resource.RLIMIT_AS, (maximum, maximum))
    signal.signal(
        signal.SIGALRM,
        lambda *_: (_ for _ in ()).throw(TimeoutError("declared120s cap")),
    )
    signal.alarm(caps["inner_seconds"])
    out = HERE / "actual"
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    checks = []
    counts = dict(
        NPZ_load=0,
        whole_matrix_products=0,
        full_matrix_vector_products=0,
        signed_DeltaH_contractions=0,
        H=0,
        CI=0,
        gradient=0,
        Dirac=0,
        optimizer=0,
        radial_key_evaluations=0,
    )
    targets = []
    wide_targets = []
    try:
        metadata(array_hashes=True)
        assert not any(k == "atomx" or k.startswith("atomx.") for k in sys.modules)
        import numpy as np

        wide = np.longdouble

        def check(name, passed, observed=None):
            checks.append(dict(name=name, passed=bool(passed), observed=observed))
            if not passed:
                raise ValueError(name)

        def product(left, right):
            counts["whole_matrix_products"] += 1
            check(
                "whole_matrix_product_cap",
                counts["whole_matrix_products"] <= caps["whole_matrix_products_max"],
            )
            return left @ right

        def mv(matrix, vector):
            counts["full_matrix_vector_products"] += 1
            check(
                "full_MV_cap",
                counts["full_matrix_vector_products"]
                <= caps["full_matrix_vector_products_max"],
            )
            return matrix @ vector

        def f(x):
            y = float(x)
            if not np.isfinite(y):
                raise FloatingPointError("nonfinite exported scalar")
            return y

        check(
            "longdouble_mantissa_at_least63",
            np.finfo(wide).nmant >= 63,
            int(np.finfo(wide).nmant),
        )
        archives = []
        for role in ("baseline_arrays", "trial_arrays"):
            counts["NPZ_load"] += 1
            check("two_archive_cap", counts["NPZ_load"] <= caps["NPZ_loads_exact"])
            with np.load(paths[role], allow_pickle=False) as z:
                check(
                    role + "/exact_complete_fields",
                    set(z.files) == set(c["fixed"]["keys_fields"]),
                )
                arrays = {k: z[k].copy() for k in z.files}
            check(
                role + "/all_finite_real",
                all(
                    np.isfinite(a).all() and not np.iscomplexobj(a)
                    for a in arrays.values()
                ),
            )
            for k in ("H", "energies", "mixing", "P", "Q"):
                check(
                    role + "/" + k + "_binary64", arrays[k].dtype == np.dtype("float64")
                )
            check(
                role + "/complete_shapes",
                arrays["H"].shape == (1595, 1595)
                and arrays["mixing"].shape == (1595, 1595)
                and arrays["energies"].shape == (1595,)
                and arrays["keys"].shape == (31, 2)
                and arrays["P"].shape == (31, 123)
                and arrays["Q"].shape == (31, 123),
            )
            check(
                role + "/whole_H_symmetry",
                arrays["H"].tobytes() == arrays["H"].T.tobytes(),
            )
            for a in arrays.values():
                a.setflags(write=False)
            archives.append(arrays)
        check(
            "decoded_array_byte_cap",
            sum(a.nbytes for x in archives for a in x.values())
            <= caps["decoded_array_bytes_max"],
        )
        old, new = archives
        check("same_all31_ordered_keys", old["keys"].tobytes() == new["keys"].tobytes())
        owners = [read(paths[k]) for k in ("baseline_owner", "trial_owner")]
        mixed = read(paths["mixed"])
        states = read(paths["states"])
        check(
            "same_complete_CSF_basis",
            owners[0]["CSFs"] == owners[1]["CSFs"] and len(owners[0]["CSFs"]) == 1595,
        )
        for i, (role, a, owner) in enumerate(
            zip(("baseline", "h0_plus"), archives, owners, strict=True)
        ):
            check(
                role + "/schema1_DC_real_Ha",
                owner["schema_version"] == 1
                and owner["representation"] == "dense-real-full-CSF"
                and owner["units"] == "Hartree"
                and owner["hamiltonian_terms"] == ["dirac-coulomb"],
            )
            check(
                role + "/same_all31_owner_keys",
                owner["orbital_keys"] == a["keys"].tolist(),
            )
            payload = json.dumps(owner, sort_keys=True, separators=(",", ":")).encode()
            check(
                role + "/full_owner_input_identity",
                hashlib.sha256(payload).hexdigest()
                == mixed["endpoint_input_sha256"][i],
            )
            frame = repr((a["H"].shape, a["H"].dtype.str)).encode() + a["H"].tobytes()
            check(
                role + "/full_retained_H_identity",
                hashlib.sha256(frame).hexdigest() == mixed["endpoint_H_sha256"][i],
            )
            residual = product(a["H"], a["mixing"]) - a["mixing"] * a["energies"]
            maximum = f(np.max(np.linalg.norm(residual, axis=0)))
            del residual
            check(
                role + "/all1595_residual",
                maximum <= c["original_gates"]["full_CI_residual_Ha"],
                maximum,
            )
            gram = product(a["mixing"].T, a["mixing"]) - np.eye(1595)
            maximum = f(np.max(abs(gram)))
            del gram
            check(
                role + "/all1595_Gram",
                maximum <= c["original_gates"]["full_CI_Gram"],
                maximum,
            )
        origin = wide(old["H"][0, 0])
        H0 = old["H"].astype(wide)
        H1 = new["H"].astype(wide)
        delta = H1 - H0
        H0.flat[::1596] -= origin
        H1.flat[::1596] -= origin
        selected0 = {tuple(r["UID"]): r for r in states["baseline"]["rows"]}
        selected1 = {tuple(r["UID"]): r for r in states["h0_plus"]["rows"]}
        ledger = {tuple(r["reference_level"]): r for r in mixed["targets"]}
        uids = list(map(tuple, c["fixed"]["five_UIDs"]))
        check(
            "all_five_original_UIDs",
            set(selected0) == set(selected1) == set(ledger) == set(uids),
        )
        for uid in uids:
            row0, row1, saved = selected0[uid], selected1[uid], ledger[uid]
            check(
                "target_binding/" + str(uid),
                tuple(saved["selected_reference"])
                == uid
                == tuple(saved["selected_trial"])
                and tuple(row0["selected_level"])
                == uid
                == tuple(row1["selected_level"]),
            )
            weight = wide(saved["weight"])
            check(
                "target_weight/" + str(uid),
                saved["weight"] == row0["weight"] == row1["weight"] == 0.2,
            )
            j0, j1 = row0["column"], row1["column"]
            check(
                "target_saved_roots/" + str(uid),
                float(old["energies"][j0]).hex() == float(row0["energy_Ha"]).hex()
                and float(new["energies"][j1]).hex() == float(row1["energy_Ha"]).hex(),
            )
            a = old["mixing"][:, j0].astype(wide)
            v = new["mixing"][:, j1].astype(wide)
            phase = saved["phase"]
            check(
                "original_whole_column_phase/" + str(uid),
                phase in (-1, 1)
                and phase
                == (
                    1 if np.dot(old["mixing"][:, j0], new["mixing"][:, j1]) >= 0 else -1
                ),
            )
            n0 = a @ a
            n1 = v @ v
            check(
                "positive_actual_norms/" + str(uid),
                np.isfinite(n0 + n1) and n0 > 0 and n1 > 0,
            )
            check(
                "exported_actual_norms/" + str(uid),
                f(n0) == saved["norm0"] and f(n1) == saved["norm1"],
            )
            d = v - phase * a
            ka = mv(H0, a)
            kv = mv(H0, v)
            kv1 = mv(H1, v)
            dv = mv(delta, v)
            counts["signed_DeltaH_contractions"] += 1
            R = v @ dv / n1
            q0 = a @ ka / n0
            q1 = v @ kv1 / n1
            direct_q = q1 - q0
            l0 = wide(old["energies"][j0])
            l1 = wide(new["energies"][j1])
            root_delta = l1 - l0
            offset0 = q0 - (l0 - origin)
            offset1 = q1 - (l1 - origin)
            direct_q_from_roots = root_delta + offset1 - offset0
            k_d = kv - phase * ka
            rq0 = phase * (ka - q0 * a)
            ray_terms = (
                d @ (k_d - q0 * d) / n1,
                2 * d @ rq0 / n1,
                (phase * a) @ rq0 / n1,
            )
            ray_response = sum(ray_terms, wide(0))
            r0 = phase * (ka - (l0 - origin) * a)
            r1 = kv1 - (l1 - origin) * v
            reported_terms = (
                d @ (k_d - (l0 - origin) * d) / n1,
                2 * d @ r0 / n1,
                (phase * a) @ r0 / n1,
                -v @ r1 / n1,
            )
            reported_response = sum(reported_terms, wide(0))
            diagonal = np.sum(np.diag(delta) * v * v, dtype=wide) / n1
            wide_targets.append(
                dict(
                    weight=weight,
                    R_DeltaH_Ha=R,
                    direct_common_origin_quotient_delta_Ha=direct_q,
                    direct_lambda_delta_Ha=root_delta,
                    rayleigh_response_Ha=ray_response,
                    reported_response_Ha=reported_response,
                )
            )
            targets.append(
                dict(
                    UID=list(uid),
                    label=row0["label"],
                    weight=f(weight),
                    phase=int(phase),
                    columns=[j0, j1],
                    actual_norms=[f(n0), f(n1)],
                    actual_C_norm_semantics="longdouble retained complete column",
                    R_DeltaH_Ha=f(R),
                    R_diagonal_Ha=f(diagonal),
                    R_signed_offdiagonal_Ha=f(R - diagonal),
                    direct_common_origin_quotient_delta_Ha=f(direct_q),
                    direct_root_plus_replayed_offsets_Ha=f(direct_q_from_roots),
                    direct_lambda_delta_Ha=f(root_delta),
                    replayed_q_minus_lambda_Ha=[f(offset0), f(offset1)],
                    saved_q_minus_lambda_Ha=saved["quotient_minus_reported"],
                    q_minus_lambda_replay_minus_saved_Ha=[
                        f(offset0 - wide(saved["quotient_minus_reported"][0])),
                        f(offset1 - wide(saved["quotient_minus_reported"][1])),
                    ],
                    rayleigh_response_terms_Ha=[f(x) for x in ray_terms],
                    reported_response_terms_Ha=[f(x) for x in reported_terms],
                    rayleigh_response_Ha=f(ray_response),
                    reported_response_Ha=f(reported_response),
                    saved_rayleigh_response_terms_Ha=saved["rayleigh_terms"],
                    saved_reported_response_terms_Ha=saved["reported_terms"],
                    matrix_R_plus_ray_response_minus_direct_q_Ha=f(
                        R + ray_response - direct_q
                    ),
                    matrix_R_plus_reported_response_minus_lambda_Ha=f(
                        R + reported_response - root_delta
                    ),
                    common_origin_vs_root_plus_offsets_Ha=f(
                        direct_q - direct_q_from_roots
                    ),
                )
            )

        # Output is scalar only; actual full-array sums above retain interference.
        def weighted(field):
            return sum((t["weight"] * t[field] for t in wide_targets), wide(0))

        aggregate_wide = {
            key: weighted(field)
            for key, field in [
                ("stored_matrix_radial", "R_DeltaH_Ha"),
                ("direct_stored_quotient", "direct_common_origin_quotient_delta_Ha"),
                ("direct_reported_roots", "direct_lambda_delta_Ha"),
                ("replayed_rayleigh_response", "rayleigh_response_Ha"),
                ("replayed_reported_response", "reported_response_Ha"),
            ]
        }
        aggregate = {key: f(value) for key, value in aggregate_wide.items()}
        polynomial = wide(mixed["explicit_radial"]) + wide(
            mixed["reconstruction_radial"]
        )
        comparisons = dict(
            saved_radial_polynomial_Ha=f(polynomial),
            radial_polynomial_minus_complete_stored_matrix_radial_Ha=f(
                polynomial - aggregate_wide["stored_matrix_radial"]
            ),
            saved_exported_quotient_delta_Ha=mixed[
                "direct_retained_quotient_difference"
            ],
            direct_stored_quotient_minus_saved_exported_Ha=f(
                aggregate_wide["direct_stored_quotient"]
                - wide(mixed["direct_retained_quotient_difference"])
            ),
            saved_rayleigh_response_Ha=mixed["rayleigh_response"],
            replayed_rayleigh_response_minus_saved_Ha=f(
                aggregate_wide["replayed_rayleigh_response"]
                - wide(mixed["rayleigh_response"])
            ),
            saved_radial_plus_replayed_response_minus_direct_stored_quotient_Ha=f(
                polynomial
                + aggregate_wide["replayed_rayleigh_response"]
                - aggregate_wide["direct_stored_quotient"]
            ),
            complete_matrix_radial_plus_replayed_response_minus_direct_stored_quotient_Ha=f(
                aggregate_wide["stored_matrix_radial"]
                + aggregate_wide["replayed_rayleigh_response"]
                - aggregate_wide["direct_stored_quotient"]
            ),
            saved_quotient_closure_Ha=mixed["quotient_closure_error"],
        )
        check("exact_two_archives", counts["NPZ_load"] == 2)
        check("four_complete_matrix_products", counts["whole_matrix_products"] == 4)
        check("twenty_complete_MVs", counts["full_matrix_vector_products"] == 20)
        check(
            "five_full_signed_DeltaH_contractions",
            counts["signed_DeltaH_contractions"] == 5,
        )
        metadata(array_hashes=True)
        check(
            "no_Atomx_import_or_new_physics",
            not any(k == "atomx" or k.startswith("atomx.") for k in sys.modules),
        )
        write(
            out / "result.json",
            dict(
                status="closed_saved_complete_contraction_discriminator",
                all_original_structure_residual_Gram_controls_passed=True,
                checks=checks,
                counts=counts,
                targets=targets,
                aggregate_Ha=aggregate,
                comparisons_Ha=comparisons,
                arithmetic=dict(
                    dtype=wide.__name__,
                    mantissa_fraction_bits=int(np.finfo(wide).nmant),
                    common_origin_Ha=f(origin),
                ),
                source_map_sha256=c["source_map_sha256"],
                contract_sha256=sha(HERE / "contract.json"),
                driver_sha256=sha(__file__),
                binding_sha256=sha(binding_path),
                inputs_sha256={k: v["sha256"] for k, v in c["input_roles"].items()},
                zero_hypothesis="unresolved_no_certified_arithmetic_error_bound",
                ordinary_FD_passed=False,
                original_two_failures_preserved=True,
                physics_cause_resolved=False,
                objective_or_callback_changed=False,
                controlled_timing=False,
                elapsed_uncontrolled_s=time.monotonic() - start,
                peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            ),
        )
    except Exception as error:
        write(
            out / "failure.json",
            dict(
                status="failed",
                error=repr(error),
                checks=checks,
                counts=counts,
                completed_targets=targets,
                original_two_failures_preserved=True,
                no_retry=True,
                elapsed_uncontrolled_s=time.monotonic() - start,
            ),
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", type=Path)
    args = parser.parse_args()
    if args.execute:
        execute(args.execute)
    else:
        metadata()
        print(
            "SOURCE/ZIP-directory only: 0 numerical imports, array-body reads/hashes or scientific calls."
        )
