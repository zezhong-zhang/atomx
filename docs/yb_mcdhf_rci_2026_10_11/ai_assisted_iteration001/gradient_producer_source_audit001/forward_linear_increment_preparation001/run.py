"""Complete forward linear-increment comparison; SOURCE-only by default."""

import argparse
import hashlib
import json
import os
import resource
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "AGENTS.md").is_file())


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def key_tuple(value):
    return tuple(key_tuple(x) if isinstance(x, list) else x for x in value)


def metadata(*, arrays=False):
    c = read(HERE / "contract.json")
    paths = {key: ROOT / row["path"] for key, row in c["input_roles"].items()}
    total = 0
    for key, path in paths.items():
        row = c["input_roles"][key]
        if row["kind"] == "metadata":
            total += path.stat().st_size
            assert sha(path) == row["sha256"], key
        elif arrays:
            assert sha(path) == row["sha256"], key
    assert total <= c["resources"]["metadata_bytes_max"]
    identity = read(paths["source_identity"])
    assert identity["source_map_sha256"] == c["source_map_sha256"]
    assert len(identity["all_atomx_python_sha256"]) == 356
    source = ROOT / c["source_archive"]
    assert all(
        sha(source / p.removeprefix("src/")) == value
        for p, value in identity["all_atomx_python_sha256"].items()
    )
    receipt = read(paths["replay_receipt"])
    replay = read(paths["replay_result"])
    assert receipt["status"] == "foreground_closed" and receipt["exit_code"] == 0
    assert receipt["source_and_inputs_unchanged"] and not receipt["outer_wall_stop"]
    assert receipt["result_sha256"] == c["input_roles"]["replay_result"]["sha256"]
    assert len(replay["checks"]) == 96 and all(x["passed"] for x in replay["checks"])
    assert replay["source_map_sha256"] == c["source_map_sha256"]
    assert (
        replay["inputs_sha256"]["baseline_arrays"]
        == c["input_roles"]["baseline_arrays"]["sha256"]
    )
    assert (
        replay["inputs_sha256"]["trial_arrays"]
        == c["input_roles"]["trial_arrays"]["sha256"]
    )
    parent = read(paths["parent_result"])
    assert (
        len(parent["checks"]) == 745
        and sum(x["passed"] for x in parent["checks"]) == 743
    )
    assert not parent["all_declared_controls_passed"]
    manifest = {x["path"]: x["sha256"] for x in parent["archives"]}
    for key, row in c["input_roles"].items():
        if "parent_manifest_path" in row:
            assert manifest[row["parent_manifest_path"]] == row["sha256"], key
    native_order = [row["reference_level"] for row in read(paths["mixed"])["targets"]]
    assert native_order == c["preflight_native_order"] == c["fixed"]["five_UIDs"]
    assert set(map(tuple, native_order)) == set(map(tuple, c["fixed"]["display_UIDs"]))
    assert [row["weight"] for row in read(paths["mixed"])["targets"]] == [0.2] * 5
    previous_receipt, previous_failure = (
        read(paths["predecessor_receipt"]),
        read(paths["predecessor_failure"]),
    )
    assert (
        previous_receipt["status"] == "foreground_closed"
        and previous_receipt["exit_code"] == 1
    )
    assert (
        previous_receipt["source_and_inputs_unchanged"]
        and not previous_receipt["outer_wall_stop"]
    )
    assert (
        previous_receipt["failure_sha256"]
        == c["input_roles"]["predecessor_failure"]["sha256"]
    )
    assert (
        len(previous_failure["checks"]) == 160
        and sum(row["passed"] for row in previous_failure["checks"]) == 159
    )
    assert [row["name"] for row in previous_failure["checks"] if not row["passed"]] == [
        "original_native_target_accumulation_order"
    ]
    assert (
        previous_failure["counts"]["ordinary_radial_evaluation"]
        == previous_failure["counts"]["signed_target_density"]
        == previous_failure["counts"]["H"]
        == previous_failure["counts"]["CI_eigh"]
        == 0
    )
    closed, closure = read(paths["closed_result"]), read(paths["closed_receipt"])
    assert closure["status"] == "foreground_closed" and closure["exit_code"] == 0
    assert closure["source_and_inputs_unchanged"] and not closure["outer_wall_stop"]
    assert closure["result_sha256"] == c["input_roles"]["closed_result"]["sha256"]
    assert len(closed["checks"]) == 191 and all(
        row["passed"] for row in closed["checks"]
    )
    assert closed["source_map_sha256"] == c["source_map_sha256"]
    assert [row["UID"] for row in closed["targets"]] == native_order
    for role in (
        "baseline_arrays",
        "trial_arrays",
        "baseline_owner",
        "trial_owner",
        "states",
        "mixed",
        "models",
        "canonical_basis",
    ):
        assert closed["inputs_sha256"][role] == c["input_roles"][role]["sha256"], role
    ledger = read(paths["closed_keys"])
    assert (
        ledger["all_structural_keys"] and ledger["no_threshold_or_permutation_shortcut"]
    )
    assert len(ledger["keys"]) == 19589
    assert len({key_tuple(row["key"]) for row in ledger["keys"]}) == 19589
    states, mixed = read(paths["states"]), read(paths["mixed"])
    rows0 = {tuple(row["UID"]): row for row in states["baseline"]["rows"]}
    rows1 = {tuple(row["UID"]): row for row in states["h0_plus"]["rows"]}
    targets = {tuple(row["UID"]): row for row in closed["targets"]}
    responses = {tuple(row["reference_level"]): row for row in mixed["targets"]}
    assert (
        set(rows0)
        == set(rows1)
        == set(targets)
        == set(responses)
        == set(map(tuple, native_order))
    )
    replay_targets = {tuple(row["UID"]): row for row in replay["targets"]}
    for uid in map(tuple, native_order):
        old, new, target, response = (
            rows0[uid],
            rows1[uid],
            targets[uid],
            responses[uid],
        )
        assert (
            tuple(old["selected_level"])
            == tuple(new["selected_level"])
            == tuple(response["selected_reference"])
            == tuple(response["selected_trial"])
            == uid
        )
        assert (
            old["weight"]
            == new["weight"]
            == target["weight"]
            == response["weight"]
            == 0.2
        )
        assert (
            0 <= old["column"] < 1595
            and 0 <= new["column"] < 1595
            and target["column"] == new["column"]
        )
        assert (
            response["norm0"] > 0
            and response["norm1"] == target["actual_norm1"] > 0
            and response["phase"] in (-1, 1)
        )
        assert replay_targets[uid]["columns"] == [old["column"], new["column"]]
        assert replay_targets[uid]["actual_norms"] == [
            response["norm0"],
            response["norm1"],
        ]
        assert replay_targets[uid]["phase"] == response["phase"]
    owners = [read(paths[role]) for role in ("baseline_owner", "trial_owner")]
    assert owners[0]["CSFs"] == owners[1]["CSFs"] and len(owners[0]["CSFs"]) == 1595
    assert (
        owners[0]["orbital_keys"] == owners[1]["orbital_keys"]
        and len(owners[0]["orbital_keys"]) == 31
    )
    for index, owner in enumerate(owners):
        assert (
            owner["representation"] == "dense-real-full-CSF"
            and owner["units"] == "Hartree"
            and owner["hamiltonian_terms"] == ["dirac-coulomb"]
        )
        payload = json.dumps(owner, sort_keys=True, separators=(",", ":")).encode()
        assert (
            hashlib.sha256(payload).hexdigest() == mixed["endpoint_input_sha256"][index]
        )
    assert all(
        set(row)
        == {
            "key",
            "endpoint0",
            "endpoint1",
            "explicit",
            "reconstruction",
            "endpoint_delta_minus_polynomial",
        }
        for row in ledger["keys"]
    )
    assert (
        sum(row["key"][0] == "h" for row in ledger["keys"]) == 57
        and sum(row["key"][0] == "r" for row in ledger["keys"]) == 19532
    )
    return c, paths


def execute(binding_path):
    c, paths = metadata()
    binding = read(binding_path)
    assert binding["execution_authorized_by_root"] is True
    assert binding["driver_sha256"] == sha(__file__) and binding[
        "contract_sha256"
    ] == sha(HERE / "contract.json")
    assert binding["preparation_sha256"] == sha(HERE / "preparation.json")
    assert (
        binding["input_roles"] == c["input_roles"]
        and binding["resources"] == c["resources"]
    )
    assert read(HERE / "preparation.json")["all_SOURCE_checks_passed"]
    cap = c["resources"]
    assert sys.executable == cap["interpreter_literal"]
    assert all(os.environ.get(name) == "1" for name in c["thread_environment"])
    assert cap["CPU"] in os.sched_getaffinity(0)
    os.sched_setaffinity(0, {cap["CPU"]})
    limit = int(cap["AS_GiB"] * 2**30)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    signal.signal(
        signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("90s cap"))
    )
    signal.alarm(cap["inner_seconds"])
    out = HERE / "actual"
    out.mkdir(exist_ok=False)
    os.environ["NUMBA_CACHE_DIR"] = str(out / "numba_cache")
    start = time.monotonic()
    checks, targets = [], []
    counts = {name: 0 for name in cap["calls_max"]}
    phase = "resource admission"

    def check(name, passed, observed=None):
        checks.append(dict(name=name, passed=bool(passed), observed=observed))
        if not passed:
            raise ValueError(name)

    def tick(name):
        counts[name] += 1
        if counts[name] > cap["calls_max"][name]:
            raise ValueError("work cap " + name)

    def progress(name):
        nonlocal phase
        phase = name
        write(
            out / "progress.json",
            dict(
                phase=phase,
                counts=counts,
                elapsed_uncontrolled_s=time.monotonic() - start,
            ),
        )

    def guard(obj, name, counter):
        def forbidden(*_args, **_kwargs):
            tick(counter)
            raise ValueError("forbidden " + name)

        setattr(obj, name, forbidden)

    def wrap(obj, name, counter):
        original = getattr(obj, name)

        def counted(*args, **kwargs):
            tick(counter)
            return original(*args, **kwargs)

        setattr(obj, name, counted)

    try:
        load1 = os.getloadavg()[0]
        available = next(
            int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        )
        write(
            out / "resource_admission.json",
            dict(
                load1=load1,
                available_GiB=available / 2**30,
                CPU=cap["CPU"],
                AS_GiB=cap["AS_GiB"],
                controlled_timing=False,
            ),
        )
        check(
            "before_arrays_imports_resource_admission",
            load1 <= cap["maximum_system_load1"]
            and available >= cap["minimum_host_available_GiB"] * 2**30,
        )
        metadata(arrays=True)
        check(
            "no_prior_Atomx_import",
            not any(
                name == "atomx" or name.startswith("atomx.") for name in sys.modules
            ),
        )
        import numpy as np
        import scipy.linalg as la

        wide = np.longdouble
        check(
            "wide_fraction_bits63",
            np.finfo(wide).nmant >= 63,
            int(np.finfo(wide).nmant),
        )
        source = ROOT / c["source_archive"]
        sys.path.insert(0, str(source))
        from atomx.angular import csf_hamiltonian as angular
        from atomx.angular import mscheme
        from atomx.config.csf import enumerate_jj_csfs
        from atomx.methods import ci
        from atomx.methods.csf import variational
        from atomx.numerics.bspline import convol, dirac_assembly, orbital_optimization
        from atomx.physics.csf import matrix, radial_density, radial_increment

        for obj, name, counter in (
            (matrix, "build_ci_matrix", "H"),
            (angular, "exact_block_matrix", "H"),
            (angular, "evaluate_block", "H"),
            (ci, "ci_solve", "CI"),
            (ci, "_solve_blocked", "CI"),
            (dirac_assembly, "build_dirac_matrices", "hS"),
            (radial_increment, "build_dirac_matrices", "hS"),
            (matrix, "_radial_evaluator", "radial"),
            (matrix, "_bspline_pair_potential", "radial"),
            (convol.ConvolCache, "get_rkb_t", "radial"),
            (radial_increment, "bspline_dc_radial_increment", "radial"),
            (variational.CSFOrbitalFunctional, "__init__", "functional"),
            (variational.CSFOrbitalFunctional, "gradient_from_weights", "gradient"),
            (orbital_optimization, "optimize_orbitals", "optimizer"),
            (la, "eigh", "eigh"),
            (np.linalg, "eigh", "eigh"),
            (np.linalg, "eigvalsh", "eigh"),
        ):
            guard(obj, name, counter)
        wrap(mscheme, "null_space", "angular_nullspace")
        wrap(angular, "block_terms", "angular_block_terms")
        wrap(angular, "csf_block_matrix", "angular_projection")
        wrap(
            radial_density.CIRadialDensity, "contract_bilinear", "signed_target_density"
        )
        endpoints, decoded = [], 0
        for role in ("baseline_arrays", "trial_arrays"):
            tick("NPZ_load")
            with np.load(paths[role], allow_pickle=False) as archive:
                check(
                    role + "/exact_complete_fields",
                    set(archive.files) == set(c["archive_fields"]),
                )
                data = {}
                for name in archive.files:
                    value = archive[name]
                    decoded += value.nbytes
                    check("decode_bound", decoded <= cap["decoded_array_bytes_max"])
                    check(
                        role + "/finite_real/" + name,
                        np.isfinite(value).all() and not np.iscomplexobj(value),
                    )
                    data[name] = value.copy()
            check(
                role + "/complete_shapes",
                data["H"].shape == data["mixing"].shape == (1595, 1595)
                and data["energies"].shape == (1595,)
                and data["keys"].shape == (31, 2)
                and data["P"].shape == data["Q"].shape == (31, 123),
            )
            check(
                role + "/binary64",
                all(
                    data[k].dtype == np.dtype("float64")
                    for k in ("H", "mixing", "energies", "P", "Q")
                ),
            )
            check(role + "/symmetric_H", data["H"].tobytes() == data["H"].T.tobytes())
            owner = read(
                paths["baseline_owner" if role == "baseline_arrays" else "trial_owner"]
            )
            mixed = read(paths["mixed"])
            index = 0 if role == "baseline_arrays" else 1
            payload = json.dumps(owner, sort_keys=True, separators=(",", ":")).encode()
            frame = (
                repr((data["H"].shape, data["H"].dtype.str)).encode()
                + data["H"].tobytes()
            )
            check(
                role + "/owner_input_frame",
                hashlib.sha256(payload).hexdigest()
                == mixed["endpoint_input_sha256"][index],
            )
            check(
                role + "/owner_H_frame",
                hashlib.sha256(frame).hexdigest() == mixed["endpoint_H_sha256"][index],
            )
            check(
                role + "/owner_DC_real_Ha",
                owner["representation"] == "dense-real-full-CSF"
                and owner["units"] == "Hartree"
                and owner["hamiltonian_terms"] == ["dirac-coulomb"]
                and owner["orbital_keys"] == data["keys"].tolist(),
            )
            endpoints.append(data)
        old, new = endpoints
        check("same31keys", old["keys"].tobytes() == new["keys"].tobytes())
        check(
            "same1595owner_basis",
            read(paths["baseline_owner"])["CSFs"] == read(paths["trial_owner"])["CSFs"],
        )
        progress("bound arrays; native full angular construction")
        models = read(paths["models"])
        sectors = set(map(tuple, models["selected_sectors"]))
        full = [
            csf
            for row in models["models"]["complete5f"]["configs"]
            for csf in enumerate_jj_csfs(row["config"])
        ]
        csfs = [csf for csf in full if (csf.total_J2, csf.parity) in sectors]
        check(
            "3789parent_1595selected69electrons",
            len(full) == 3789
            and len(csfs) == 1595
            and {x.n_electrons for x in full} == {69},
        )
        basis = []
        for csf in csfs:
            sig = angular.csf_signature(csf)
            basis.append(
                json.dumps(
                    (sig.subshells, sig.coupling, csf.total_J2, csf.parity),
                    separators=(",", ":"),
                )
            )
        check(
            "exact_signed_basis",
            basis
            == read(paths["canonical_basis"])["complete5f"]["retained_signatures"],
        )
        density = radial_density.CIRadialDensity(csfs, max_determinants=2000)
        check("five_angular_blocks", len(density.blocks) == 5)
        nnz = sum(s.table.nnz for s, _, _ in density.blocks)
        pairs = sum(len(s.pairs) for s, _, _ in density.blocks)
        check(
            "angular_nnz_pairs_bounds",
            nnz <= cap["angular_nnz_max"] and pairs <= cap["angular_pairs_max"],
            [nnz, pairs],
        )
        ledger = read(paths["closed_keys"])["keys"]
        ordered_keys = tuple(key_tuple(row["key"]) for row in ledger)
        structural_keys = tuple(
            dict.fromkeys(key for s, _, _ in density.blocks for key in s.keys)
        )
        check("exact_all19589_orderedkeys", ordered_keys == structural_keys)
        by_key = {key_tuple(row["key"]): row for row in ledger}
        matrices, precision = [], []
        for s, projection, columns in density.blocks:
            groups = {}
            for column in columns:
                sig = density.signatures[column]
                groups.setdefault(sig.pattern, []).append(sig)
            grouped = tuple(tuple(groups[p]) for p in s.patterns)
            sparse_projection = angular.csf_block_matrix(
                s.patterns, s.j2, grouped, sparse=True
            )
            check(
                "same_sparse_dense_projection/" + str(s.j2),
                sparse_projection.toarray().tobytes() == projection.tobytes(),
            )
            R0, R1, explicit, extra = [
                np.array([by_key[key][field] for key in s.keys])
                for field in ("endpoint0", "endpoint1", "explicit", "reconstruction")
            ]
            delta64, poly64 = R1 - R0, explicit + extra
            delta_wide, poly_wide = (
                R1.astype(wide) - R0.astype(wide),
                explicit.astype(wide) + extra.astype(wide),
            )
            table_wide = s.table.astype(wide)
            check(
                "all_wide_table_entries_preserved/" + str(s.j2),
                table_wide.nnz == s.table.nnz
                and table_wide.indptr.tobytes() == s.table.indptr.tobytes()
                and table_wide.indices.tobytes() == s.table.indices.tobytes()
                and np.array_equal(table_wide.data, s.table.data.astype(wide)),
            )
            forwarded = {}
            for label, table, values in (
                ("native_delta", s.table, delta64),
                ("native_poly", s.table, poly64),
                ("wide_delta", table_wide, delta_wide),
                ("wide_poly", table_wide, poly_wide),
            ):
                tick("forward_table_product")
                forwarded[label] = table @ values
                check(
                    "finite_complete_row_action/" + label,
                    forwarded[label].shape == (len(s.pairs),)
                    and np.isfinite(forwarded[label]).all(),
                )
            matrices.append((s, projection, sparse_projection, columns, forwarded))
            precision.append(
                dict(
                    twoJ=s.j2,
                    all_keys=len(s.keys),
                    native_delta_input_rounding_max=float(
                        np.max(abs(delta_wide - delta64.astype(wide)), initial=0.0)
                    ),
                    native_poly_input_rounding_max=float(
                        np.max(abs(poly_wide - poly64.astype(wide)), initial=0.0)
                    ),
                )
            )
        progress("all20 full forward row actions; five signed target contractions")
        states = read(paths["states"])
        rows = {tuple(row["UID"]): row for row in states["h0_plus"]["rows"]}
        refs = {tuple(row["UID"]): row for row in states["baseline"]["rows"]}
        saved = {
            tuple(row["UID"]): row for row in read(paths["closed_result"])["targets"]
        }
        mixed_targets = {tuple(row["reference_level"]): row for row in mixed["targets"]}
        coefficients = []
        for uid in map(tuple, c["fixed"]["five_UIDs"]):
            row, ref, previous = rows[uid], refs[uid], saved[uid]
            check(
                "unchanged_UID_roots_weights/" + str(uid),
                tuple(row["selected_level"]) == tuple(ref["selected_level"]) == uid
                and row["weight"] == ref["weight"] == previous["weight"] == 0.2
                and new["energies"][row["column"]] == row["energy_Ha"]
                and old["energies"][ref["column"]] == ref["energy_Ha"],
            )
            v, a = new["mixing"][:, row["column"]], old["mixing"][:, ref["column"]]
            norm, norm0 = (
                v.astype(wide) @ v.astype(wide),
                a.astype(wide) @ a.astype(wide),
            )
            check(
                "original_actual_norm_phase/" + str(uid),
                norm > 0
                and norm0 > 0
                and float(norm)
                == previous["actual_norm1"]
                == mixed_targets[uid]["norm1"]
                and float(norm0) == mixed_targets[uid]["norm0"]
                and mixed_targets[uid]["phase"] == (1 if a @ v >= 0 else -1),
            )
            tick("signed_target_vectors")
            accum = {
                label: wide(0)
                for label in ("native_delta", "native_poly", "wide_delta", "wide_poly")
            }
            absolute = dict(accum)
            sparse_wide_delta = wide(0)
            dense_sparse_max = 0.0
            for s, projection, sparse_projection, columns, forwarded in matrices:
                tick("dense_target_projection")
                dense = projection @ v[columns]
                tick("sparse_target_projection")
                amps = sparse_projection @ v[columns]
                dense_sparse_max = max(
                    dense_sparse_max, float(np.max(abs(dense - amps), initial=0.0))
                )
                i, j = s.pairs.T
                products = 0.5 * (dense[i] * dense[j] + dense[i] * dense[j])
                for label, values in forwarded.items():
                    tick("signed_row_contraction")
                    terms = products.astype(wide) * values.astype(wide)
                    accum[label] += np.sum(terms, dtype=wide) / norm
                    absolute[label] += np.sum(abs(terms), dtype=wide) / norm
                sparse_products = 0.5 * (amps[i] * amps[j] + amps[i] * amps[j])
                tick("signed_row_contraction")
                sparse_wide_delta += (
                    np.sum(
                        sparse_products.astype(wide) * forwarded["wide_delta"],
                        dtype=wide,
                    )
                    / norm
                )
            table = density.contract_bilinear(v, v)
            coef = [table.get(key, 0.0) for key in ordered_keys]
            transpose_delta_terms = [
                wide(value)
                * (wide(by_key[key]["endpoint1"]) - wide(by_key[key]["endpoint0"]))
                for key, value in zip(ordered_keys, coef, strict=True)
            ]
            transpose_poly_terms = [
                wide(value)
                * (wide(by_key[key]["explicit"]) + wide(by_key[key]["reconstruction"]))
                for key, value in zip(ordered_keys, coef, strict=True)
            ]
            transpose_delta, transpose_poly = (
                sum(transpose_delta_terms, wide(0)) / norm,
                sum(transpose_poly_terms, wide(0)) / norm,
            )
            coefficients.append(
                dict(
                    UID=list(uid),
                    actual_norm1=float(norm),
                    all_key_coefficients=coef,
                    complete_signed_key_delta_terms_Ha=list(
                        map(float, (term / norm for term in transpose_delta_terms))
                    ),
                    complete_signed_key_poly_terms_Ha=list(
                        map(float, (term / norm for term in transpose_poly_terms))
                    ),
                )
            )
            targets.append(
                dict(
                    UID=list(uid),
                    label=row["label"],
                    weight=0.2,
                    column=row["column"],
                    actual_norm1=float(norm),
                    all_signed_forward_Ha={
                        label: float(value) for label, value in accum.items()
                    },
                    all_signed_forward_absolute_term_sums_Ha={
                        label: float(value) for label, value in absolute.items()
                    },
                    saved_endpoint_forward_subtraction_Ha=previous[
                        "forward_dense_delta_Ha"
                    ],
                    saved_forward_signed_pair_delta_Ha=previous[
                        "forward_dense_signed_pair_increment_Ha"
                    ],
                    saved_stored_matrix_radial_Ha=previous["stored_matrix_radial_Ha"],
                    saved_transpose_delta_Ha=previous["ordinary_endpoint_radial_Ha"],
                    saved_transpose_poly_Ha=previous["radial_polynomial_Ha"],
                    replay_transpose_delta_Ha=float(transpose_delta),
                    replay_transpose_poly_Ha=float(transpose_poly),
                    dense_sparse_projection_max=dense_sparse_max,
                    primary_pair_products="nativeFP64 dense sameastranspose",
                    sparse_wide_delta_control_Ha=float(sparse_wide_delta),
                    diagnostics_Ha={
                        "wide_delta_minus_transpose": float(
                            accum["wide_delta"] - transpose_delta
                        ),
                        "wide_poly_minus_transpose": float(
                            accum["wide_poly"] - transpose_poly
                        ),
                        "sparse_minus_dense_wide_delta": float(
                            sparse_wide_delta - accum["wide_delta"]
                        ),
                        "native_delta_minus_transpose": float(
                            accum["native_delta"] - transpose_delta
                        ),
                        "native_poly_minus_transpose": float(
                            accum["native_poly"] - transpose_poly
                        ),
                        "wide_minus_native_delta": float(
                            accum["wide_delta"] - accum["native_delta"]
                        ),
                        "wide_minus_native_poly": float(
                            accum["wide_poly"] - accum["native_poly"]
                        ),
                        "native_delta_minus_saved_forward_signed_pair_delta": float(
                            accum["native_delta"]
                            - wide(previous["forward_dense_signed_pair_increment_Ha"])
                        ),
                        "native_delta_minus_saved_endpoint_subtraction": float(
                            accum["native_delta"]
                            - wide(previous["forward_dense_delta_Ha"])
                        ),
                        "wide_delta_minus_saved_stored": float(
                            accum["wide_delta"]
                            - wide(previous["stored_matrix_radial_Ha"])
                        ),
                    },
                )
            )
        write(
            out / "complete_trial_signed_key_contributions.json",
            dict(
                ordered_keys=ordered_keys,
                targets=coefficients,
                all19589_keys_including_exact_zeros=True,
            ),
        )
        for name, exact in cap["calls_exact"].items():
            check("exact_work/" + name, counts[name] == exact, counts[name])
        metadata(arrays=True)
        origins = {
            name: getattr(module, "__file__", None)
            for name, module in sys.modules.items()
            if name == "atomx" or name.startswith("atomx.")
        }
        check(
            "all_frozen_Atomx_origins",
            all(
                value and Path(value).resolve().is_relative_to(source.resolve())
                for value in origins.values()
            ),
        )
        write(out / "imported_module_origins.json", origins)
        write(
            out / "result.json",
            dict(
                status="closed_complete_forward_linear_increment_diagnostic",
                checks=checks,
                counts=counts,
                targets=targets,
                input_precision=precision,
                angular=dict(
                    blocks=5, nnz=nnz, pairs=pairs, ordered_keys=len(ordered_keys)
                ),
                decoded_bytes=decoded,
                source_map_sha256=c["source_map_sha256"],
                driver_sha256=sha(__file__),
                contract_sha256=sha(HERE / "contract.json"),
                binding_sha256=sha(binding_path),
                inputs_sha256={
                    role: row["sha256"] for role, row in c["input_roles"].items()
                },
                original_failures_preserved=True,
                certified_error_bound=False,
                ordinary_FD_passed=False,
                gradient_repaired=False,
                optimizer_qualified=False,
                no_owner_constructed=True,
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
                phase=phase,
                error=repr(error),
                checks=checks,
                counts=counts,
                completed_targets=targets,
                original_failures_preserved=True,
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
        print("SOURCE only: 0 NPZ access, numerical imports or scientific calls.")
