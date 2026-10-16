"""Complete endpoint radial producer comparison; SOURCE-only by default."""

import argparse
import hashlib
import json
import math
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
    return c, paths


def execute(binding_path):
    c, paths = metadata()
    binding = read(binding_path)
    assert binding["execution_authorized_by_root"] is True
    assert binding["driver_sha256"] == sha(__file__)
    assert binding["contract_sha256"] == sha(HERE / "contract.json")
    assert binding["preparation_sha256"] == sha(HERE / "preparation.json")
    assert binding["input_roles"] == c["input_roles"]
    assert binding["resources"] == c["resources"]
    prep = read(HERE / "preparation.json")
    assert prep["all_SOURCE_checks_passed"] and prep["driver_sha256"] == sha(__file__)
    caps = c["resources"]
    assert sys.executable == caps["interpreter_literal"]
    for name in c["thread_environment"]:
        assert os.environ.get(name) == "1", name
    assert caps["CPU"] in os.sched_getaffinity(0)
    os.sched_setaffinity(0, {caps["CPU"]})
    limit = int(caps["AS_GiB"] * 2**30)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    signal.signal(
        signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("300s cap"))
    )
    signal.alarm(caps["inner_seconds"])
    out = HERE / "actual"
    out.mkdir(exist_ok=False)
    os.environ["NUMBA_CACHE_DIR"] = str(out / "numba_cache")
    start = time.monotonic()
    checks, targets, primitives = [], [], []
    counts = {name: 0 for name in caps["calls_max"]}
    phase = "bound inputs"

    def tick(name, number=1):
        counts[name] += number
        if counts[name] > caps["calls_max"][name]:
            raise ValueError("work cap " + name)

    def check(name, passed, observed=None):
        checks.append(dict(name=name, passed=bool(passed), observed=observed))
        if not passed:
            raise ValueError(name)

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

    def wrap(obj, name, counter):
        original = getattr(obj, name)

        def counted(*args, **kwargs):
            tick(counter)
            return original(*args, **kwargs)

        setattr(obj, name, counted)
        return original

    def block(obj, name, counter):
        def forbidden(*_args, **_kwargs):
            tick(counter)
            raise ValueError("forbidden producer " + name)

        setattr(obj, name, forbidden)

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
                CPU=caps["CPU"],
                AS_GiB=caps["AS_GiB"],
                timing_controlled=False,
            ),
        )
        check(
            "before_numerical_host_resource_admission",
            load1 <= caps["maximum_system_load1"]
            and available >= caps["minimum_host_available_GiB"] * 2**30,
        )
        metadata(arrays=True)
        check(
            "no_prior_Atomx_import",
            not any(k == "atomx" or k.startswith("atomx.") for k in sys.modules),
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
        from atomx.numerics.bspline import convol, dirac_assembly
        from atomx.physics.csf import matrix, radial_density, radial_increment

        # Imports may eagerly load method modules. Calls, not presence, are guarded.
        block(matrix, "build_ci_matrix", "H")
        block(angular, "exact_block_matrix", "H")
        block(angular, "evaluate_block", "H")
        block(dirac_assembly, "build_dirac_matrices", "native_hS_assembly")
        block(la, "eigh", "CI_eigh")
        block(np.linalg, "eigh", "CI_eigh")
        block(np.linalg, "eigvalsh", "CI_eigh")
        wrap(mscheme, "null_space", "angular_nullspace")
        wrap(angular, "block_terms", "angular_block_terms")
        wrap(angular, "csf_block_matrix", "angular_projection")
        wrap(
            radial_density.CIRadialDensity, "contract_bilinear", "signed_target_density"
        )
        wrap(matrix, "_density_sym", "density_kernel")
        wrap(radial_increment, "_density_sym", "density_kernel")
        wrap(matrix, "_sum_amb_banded", "banded_contraction")
        wrap(radial_increment, "_sum_amb_banded", "banded_contraction")
        wrap(matrix, "_bspline_pair_potential", "pair_potential")
        wrap(radial_increment, "_bspline_pair_potential", "pair_potential")
        original_bank = convol.ConvolCache.get_rkb_t

        def bank(self, rank, itype="pppp"):
            tick("bank_getter")
            if (rank, itype, "T") not in self._cache:
                tick("bank_transpose_miss")
            return original_bank(self, rank, itype)

        convol.ConvolCache.get_rkb_t = bank
        wrap(radial_increment._DCRadialIncrement, "components", "polynomial_cache_miss")
        decoded = 0

        def archive(role):
            nonlocal decoded
            tick("NPZ_load")
            with np.load(paths[role], allow_pickle=False) as z:
                expected = c["archive_fields"][role]
                if role == "geometry_workspace":
                    check(
                        role + "/required_fields_and_complete_basis_prefix",
                        set(expected) <= set(z.files)
                        and len(z.files) <= caps["geometry_archive_fields_max"]
                        and {key for key in z.files if key.startswith("basis_")}
                        == {key for key in expected if key.startswith("basis_")},
                    )
                else:
                    check(
                        role + "/complete_declared_fields",
                        set(z.files) == set(expected),
                    )
                data = {}
                for key in expected:
                    a = z[key]
                    decoded += a.nbytes
                    check(
                        "decoded_bytes_cap", decoded <= caps["decoded_array_bytes_max"]
                    )
                    check(
                        role + "/finite_real/" + key,
                        np.isfinite(a).all() and not np.iscomplexobj(a),
                    )
                    data[key] = a.copy()
            return data

        endpoints = [archive(role) for role in ("baseline_arrays", "trial_arrays")]
        for role, data in zip(("baseline", "h0_plus"), endpoints, strict=True):
            check(
                role + "/complete1595_31_shapes",
                data["H"].shape == (1595, 1595)
                and data["mixing"].shape == (1595, 1595)
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
            check(
                role + "/bitwise_symmetric_H",
                data["H"].tobytes() == data["H"].T.tobytes(),
            )
            owner = read(
                paths["baseline_owner" if role == "baseline" else "trial_owner"]
            )
            payload = json.dumps(owner, sort_keys=True, separators=(",", ":")).encode()
            mixed = read(paths["mixed"])
            index = 0 if role == "baseline" else 1
            check(
                role + "/original_owner_input_binding",
                hashlib.sha256(payload).hexdigest()
                == mixed["endpoint_input_sha256"][index],
            )
            frame = (
                repr((data["H"].shape, data["H"].dtype.str)).encode()
                + data["H"].tobytes()
            )
            check(
                role + "/original_owner_H_binding",
                hashlib.sha256(frame).hexdigest() == mixed["endpoint_H_sha256"][index],
            )
            check(
                role + "/DC_real_Ha",
                owner["hamiltonian_terms"] == ["dirac-coulomb"]
                and owner["units"] == "Hartree"
                and owner["representation"] == "dense-real-full-CSF",
            )
        old, new = endpoints
        check("same31_ordered_keys", old["keys"].tobytes() == new["keys"].tobytes())
        keys31 = tuple(map(tuple, old["keys"].tolist()))
        waves = [
            {key: (a["P"][i], a["Q"][i]) for i, key in enumerate(keys31)}
            for a in endpoints
        ]
        increments = archive("increments")
        check(
            "same31_increment_keys_shapes",
            increments["keys"].tobytes() == old["keys"].tobytes()
            and increments["P"].shape == increments["Q"].shape == (31, 123),
        )
        ds = {
            key: (increments["P"][i], increments["Q"][i])
            for i, key in enumerate(keys31)
        }
        geometry = archive("geometry_workspace")
        workspace = {
            key[6:]: value.item() if value.ndim == 0 else value.copy()
            for key, value in geometry.items()
            if key.startswith("basis_")
        }
        check("original_ns123_geometry", workspace["ns"] == 123)
        raw = {}
        for kappa in c["fixed"]["kappas"]:
            a = archive("raw_" + str(kappa))
            check(
                "raw_operator_shape/" + str(kappa),
                a["kappa"].item() == kappa
                and a["raw_h"].shape == a["S"].shape == (246, 246)
                and a["raw_h"].dtype == a["S"].dtype == np.dtype("float64"),
            )
            for value in (a["raw_h"], a["S"]):
                value.setflags(write=False)
            raw[kappa] = (a["raw_h"], a["S"])
        progress("complete bound arrays loaded; rebuild angular density only")
        models = read(paths["models"])
        sectors = set(map(tuple, models["selected_sectors"]))
        full = [
            csf
            for row in models["models"]["complete5f"]["configs"]
            for csf in enumerate_jj_csfs(row["config"])
        ]
        csfs = [csf for csf in full if (csf.total_J2, csf.parity) in sectors]
        check(
            "complete_parent3789_selected1595", len(full) == 3789 and len(csfs) == 1595
        )
        check("all69electrons", {csf.n_electrons for csf in full} == {69})
        basis = read(paths["canonical_basis"])["complete5f"]["retained_signatures"]
        actual_basis = []
        for csf in csfs:
            sig = angular.csf_signature(csf)
            actual_basis.append(
                json.dumps(
                    (sig.subshells, sig.coupling, csf.total_J2, csf.parity),
                    separators=(",", ":"),
                )
            )
        check("complete_original_signed_basis", actual_basis == basis)
        owner0, owner1 = read(paths["baseline_owner"]), read(paths["trial_owner"])
        check(
            "same_complete_owner_CSF_basis",
            owner0["CSFs"] == owner1["CSFs"] and len(owner0["CSFs"]) == 1595,
        )
        density = radial_density.CIRadialDensity(csfs, max_determinants=2000)
        sparse_projections = []
        for structure, projection, columns in density.blocks:
            groups = {}
            for column in columns:
                sig = density.signatures[column]
                groups.setdefault(sig.pattern, []).append(sig)
            grouped = tuple(tuple(groups[p]) for p in structure.patterns)
            sparse = angular.csf_block_matrix(
                structure.patterns, structure.j2, grouped, sparse=True
            )
            check(
                "genuine_sparse_dense_same_represented_projection/" + str(structure.j2),
                sparse.toarray().tobytes() == projection.tobytes(),
            )
            sparse_projections.append(sparse)
        structural_keys = tuple(
            dict.fromkeys(
                key for structure, _, _ in density.blocks for key in structure.keys
            )
        )
        check("complete19589_angular_capacity", len(structural_keys) == 19589)
        check("five_complete_density_blocks", len(density.blocks) == 5)
        total_nnz = sum(structure.table.nnz for structure, _, _ in density.blocks)
        total_pairs = sum(len(structure.pairs) for structure, _, _ in density.blocks)
        check(
            "complete_angular_nnz_budget",
            total_nnz <= caps["angular_nnz_max"],
            total_nnz,
        )
        check(
            "complete_angular_pairs_budget",
            total_pairs <= caps["angular_pairs_max"],
            total_pairs,
        )
        states = read(paths["states"])
        rows0 = {tuple(row["UID"]): row for row in states["baseline"]["rows"]}
        rows1 = {tuple(row["UID"]): row for row in states["h0_plus"]["rows"]}
        mixed_rows = {tuple(row["reference_level"]): row for row in mixed["targets"]}
        check(
            "original_native_target_accumulation_order",
            [row["reference_level"] for row in mixed["targets"]]
            == c["fixed"]["five_UIDs"],
        )
        saved_density = read(paths["baseline_signed_tables"])
        saved_tables = {tuple(row["UID"]): row for row in saved_density["targets"]}
        check(
            "complete_five_UIDs",
            set(rows0)
            == set(rows1)
            == set(mixed_rows)
            == set(saved_tables)
            == set(map(tuple, c["fixed"]["five_UIDs"])),
        )
        table_rows = []
        aggregate_weights = {}
        for uid in map(tuple, c["fixed"]["five_UIDs"]):
            row0, row1, saved = rows0[uid], rows1[uid], mixed_rows[uid]
            check(
                "unchanged_UID_weight/" + str(uid),
                tuple(row0["selected_level"]) == tuple(row1["selected_level"]) == uid
                and tuple(saved["selected_reference"])
                == tuple(saved["selected_trial"])
                == uid
                and row0["weight"] == row1["weight"] == saved["weight"] == 0.2,
            )
            col0, col1 = row0["column"], row1["column"]
            check(
                "original_reported_roots/" + str(uid),
                old["energies"][col0] == row0["energy_Ha"]
                and new["energies"][col1] == row1["energy_Ha"],
            )
            a, v = old["mixing"][:, col0], new["mixing"][:, col1]
            n0, n1 = a.astype(wide) @ a.astype(wide), v.astype(wide) @ v.astype(wide)
            check(
                "original_actual_norm_phase/" + str(uid),
                float(n0) == saved["norm0"]
                and float(n1) == saved["norm1"]
                and n0 > 0
                and n1 > 0
                and saved["phase"] == (1 if a @ v >= 0 else -1),
            )
            table0 = density.contract_bilinear(a, a)
            reference_table = {
                key_tuple(k): value for k, value in saved_tables[uid]["unfiltered"]
            }
            # Same frozen producer and original arithmetic; differences are diagnosed, not gated to zero.
            baseline_difference = max(
                (
                    abs(table0.get(key, 0.0) - reference_table.get(key, 0.0))
                    for key in set(table0) | set(reference_table)
                ),
                default=0.0,
            )
            table = density.contract_bilinear(v, v)
            for key, value in table.items():
                aggregate_weights[key] = math.fsum(
                    (aggregate_weights.get(key, 0.0), 0.2 * value / saved["norm1"])
                )
            table_rows.append(
                dict(
                    UID=uid,
                    label=row1["label"],
                    column=col1,
                    norm1=n1,
                    vector=v,
                    coefficients=table,
                    baseline_table_difference=baseline_difference,
                )
            )
        check(
            "complete_native_mixed_signed_union",
            len(aggregate_weights) == mixed["signed_density_terms"] == 19589,
        )

        # Explicit saved-operator adapter. It is not a native h/S assembly or an owner.
        def raw_replay(ns, kappa, Z, data):
            tick("saved_raw_operator_requests")
            if ns != 123 or Z != 70 or data is not workspace or kappa not in raw:
                raise ValueError("saved raw operator replay identity")
            return raw[kappa]

        matrix._dirac_blocks = raw_replay
        radial_increment.build_dirac_matrices = lambda ns, kappa, Z, data, potential: (
            raw_replay(ns, kappa, Z, data)
        )
        evaluators = [
            matrix._radial_evaluator(w, workspace, 70, 123, nuclear_potential=None)
            for w in waves
        ]
        increment = radial_increment.bspline_dc_radial_increment(
            waves[0],
            ds,
            workspace,
            70,
            123,
            nuclear_potential=None,
            cache=matrix._context_cache(workspace),
            trial=waves[1],
        )
        check(
            "same_original_explicit_increment_hash",
            increment.increment_sha256 == mixed["radial"]["increment_sha256"],
        )
        representation_terms = [
            (wide(t) - wide(r) - wide(d) - wide(e))
            for key in keys31
            for side in range(2)
            for r, d, e, t in zip(
                waves[0][key][side],
                ds[key][side],
                increment.reconstruction[key][side],
                waves[1][key][side],
                strict=True,
            )
        ]
        representation = dict(
            max_abs=float(max(map(abs, representation_terms))),
            signed_sum=float(sum(representation_terms, wide(0))),
            absolute_sum=float(sum(map(abs, representation_terms), wide(0))),
            all_components=31 * 2 * 123,
            units="native represented coefficient",
        )
        progress("complete signed density built; ordinary endpoint and polynomial keys")
        values0, values1, polynomial, reconstruction, right_delta = {}, {}, {}, {}, {}
        for key in structural_keys:
            tick("ordinary_radial_evaluation", 2)
            values0[key], values1[key] = evaluators[0](key), evaluators[1](key)
            tick("polynomial_key_request")
            polynomial[key], reconstruction[key] = increment.components(key)
            if key[0] == "h":
                _, na, kappa, nc = key
                h, metric = raw[kappa]
                endpoint = []
                for w in waves:
                    bra, ket = (
                        np.concatenate(w[na, kappa]),
                        np.concatenate(w[nc, kappa]),
                    )
                    endpoint.append(float(bra @ (h @ ket)) / float(ket @ metric @ ket))
                    tick("right_associated_onebody_endpoint")
                right_delta[key] = wide(endpoint[1]) - wide(endpoint[0])
            primitives.append(
                dict(
                    key=key,
                    endpoint0=values0[key],
                    endpoint1=values1[key],
                    explicit=polynomial[key],
                    reconstruction=reconstruction[key],
                    endpoint_delta_minus_polynomial=float(
                        wide(values1[key])
                        - wide(values0[key])
                        - wide(polynomial[key])
                        - wide(reconstruction[key])
                    ),
                )
            )
        write(
            out / "complete_ordered_radial_keys.json",
            dict(
                keys=primitives,
                all_structural_keys=True,
                no_threshold_or_permutation_shortcut=True,
            ),
        )
        progress("all radial keys complete; complete signed contractions")
        delta_H = new["H"].astype(wide) - old["H"].astype(wide)
        forward_cache = {}
        for block_index, (structure, _, _) in enumerate(density.blocks):
            forward_cache[block_index] = []
            for vals in (values0, values1):
                tick("angular_forward_table")
                forward_cache[block_index].append(
                    structure.table @ np.array([vals[key] for key in structure.keys])
                )
        aggregate_values = {}
        for row in table_rows:
            v = row["vector"].astype(wide)
            tick("stored_H_MV")
            stored_R = v @ (delta_H @ v) / row["norm1"]
            routes = {}
            for kind in ("h", "r"):
                terms0, terms1, diffs, explicit, extra, associated = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                for key in structural_keys:
                    if key[0] != kind:
                        continue
                    coefficient = wide(row["coefficients"].get(key, 0.0)) / row["norm1"]
                    terms0.append(coefficient * wide(values0[key]))
                    terms1.append(coefficient * wide(values1[key]))
                    diffs.append(
                        coefficient * (wide(values1[key]) - wide(values0[key]))
                    )
                    explicit.append(coefficient * wide(polynomial[key]))
                    extra.append(coefficient * wide(reconstruction[key]))
                    if kind == "h":
                        associated.append(coefficient * right_delta[key])
                routes[kind] = dict(
                    endpoint0=float(sum(terms0, wide(0))),
                    endpoint1=float(sum(terms1, wide(0))),
                    endpoint_delta=float(sum(diffs, wide(0))),
                    explicit=float(sum(explicit, wide(0))),
                    reconstruction=float(sum(extra, wide(0))),
                    endpoint_absolute_term_sums=[
                        float(sum(map(abs, terms0), wide(0))),
                        float(sum(map(abs, terms1), wide(0))),
                    ],
                    delta_absolute_term_sum=float(sum(map(abs, diffs), wide(0))),
                    polynomial_absolute_term_sum=float(
                        sum(map(abs, explicit), wide(0)) + sum(map(abs, extra), wide(0))
                    ),
                    endpoint_delta_minus_polynomial=float(
                        sum(diffs, wide(0))
                        - sum(explicit, wide(0))
                        - sum(extra, wide(0))
                    ),
                    right_associated_delta=float(sum(associated, wide(0)))
                    if kind == "h"
                    else None,
                )
            sparse_forward = [wide(0), wide(0)]
            dense_forward = [wide(0), wide(0)]
            sparse_forward_increment = wide(0)
            dense_forward_increment = wide(0)
            projection_difference = 0.0
            for block_index, (structure, projection, columns) in enumerate(
                density.blocks
            ):
                tick("dense_projected_target")
                dense = projection @ row["vector"][columns]
                tick("sparse_projected_target")
                sparse = sparse_projections[block_index] @ row["vector"][columns]
                projection_difference = max(
                    projection_difference,
                    float(np.max(abs(dense - sparse), initial=0.0)),
                )
                i, j = structure.pairs.T
                dense_products = 0.5 * (dense[i] * dense[j] + dense[i] * dense[j])
                sparse_products = 0.5 * (sparse[i] * sparse[j] + sparse[i] * sparse[j])
                for index, forward in enumerate(forward_cache[block_index]):
                    dense_forward[index] += (
                        np.sum(
                            dense_products.astype(wide) * forward.astype(wide),
                            dtype=wide,
                        )
                        / row["norm1"]
                    )
                    sparse_forward[index] += (
                        np.sum(
                            sparse_products.astype(wide) * forward.astype(wide),
                            dtype=wide,
                        )
                        / row["norm1"]
                    )
                forward_delta = forward_cache[block_index][1].astype(
                    wide
                ) - forward_cache[block_index][0].astype(wide)
                dense_forward_increment += (
                    np.sum(
                        dense_products.astype(wide) * forward_delta,
                        dtype=wide,
                    )
                    / row["norm1"]
                )
                sparse_forward_increment += (
                    np.sum(
                        sparse_products.astype(wide) * forward_delta,
                        dtype=wide,
                    )
                    / row["norm1"]
                )
            ordinary_R = (
                sum(
                    (
                        wide(row["coefficients"].get(key, 0.0))
                        * (wide(values1[key]) - wide(values0[key]))
                        for key in structural_keys
                    ),
                    wide(0),
                )
                / row["norm1"]
            )
            poly_R = (
                sum(
                    (
                        wide(row["coefficients"].get(key, 0.0))
                        * (wide(polynomial[key]) + wide(reconstruction[key]))
                        for key in structural_keys
                    ),
                    wide(0),
                )
                / row["norm1"]
            )
            targets.append(
                dict(
                    UID=list(row["UID"]),
                    label=row["label"],
                    weight=0.2,
                    column=row["column"],
                    actual_norm1=float(row["norm1"]),
                    native_baseline_density_minus_saved_max=row[
                        "baseline_table_difference"
                    ],
                    complete_signed_coefficients=len(row["coefficients"]),
                    routes_Ha=routes,
                    stored_matrix_radial_Ha=float(stored_R),
                    ordinary_endpoint_radial_Ha=float(ordinary_R),
                    radial_polynomial_Ha=float(poly_R),
                    ordinary_minus_stored_Ha=float(ordinary_R - stored_R),
                    polynomial_minus_ordinary_Ha=float(poly_R - ordinary_R),
                    polynomial_minus_stored_Ha=float(poly_R - stored_R),
                    dense_sparse_projection_max=projection_difference,
                    forward_dense_endpoint_Ha=list(map(float, dense_forward)),
                    forward_sparse_endpoint_Ha=list(map(float, sparse_forward)),
                    forward_dense_delta_Ha=float(dense_forward[1] - dense_forward[0]),
                    forward_sparse_delta_Ha=float(
                        sparse_forward[1] - sparse_forward[0]
                    ),
                    forward_dense_signed_pair_increment_Ha=float(
                        dense_forward_increment
                    ),
                    forward_sparse_signed_pair_increment_Ha=float(
                        sparse_forward_increment
                    ),
                )
            )
            for key, value in (
                ("stored_R", stored_R),
                ("ordinary_R", ordinary_R),
                ("polynomial_R", poly_R),
            ):
                aggregate_values[key] = (
                    aggregate_values.get(key, wide(0)) + wide(0.2) * value
                )
        reproduced_explicit = math.fsum(
            value * polynomial[key] for key, value in aggregate_weights.items()
        )
        reproduced_extra = math.fsum(
            value * reconstruction[key] for key, value in aggregate_weights.items()
        )
        for key, expected in c["resources"]["calls_exact"].items():
            check("exact_work/" + key, counts[key] == expected, counts[key])
        check("bound_arrays_geometry_source_unchanged", bool(metadata(arrays=True)))
        modules = {
            name: getattr(module, "__file__", None)
            for name, module in sys.modules.items()
            if name == "atomx" or name.startswith("atomx.")
        }
        check(
            "all_Atomx_module_origins_frozen",
            all(
                value and Path(value).resolve().is_relative_to(source.resolve())
                for value in modules.values()
            ),
        )
        write(out / "imported_module_origins.json", modules)
        write(
            out / "result.json",
            dict(
                status="closed_complete_endpoint_radial_diagnostic",
                checks=checks,
                counts=counts,
                decoded_array_bytes=decoded,
                angular=dict(
                    blocks=5,
                    nnz=total_nnz,
                    pairs=total_pairs,
                    structural_keys=len(structural_keys),
                ),
                targets=targets,
                aggregate_Ha={
                    key: float(value) for key, value in aggregate_values.items()
                },
                native_mixed_aggregate_replay_Ha=dict(
                    explicit=reproduced_explicit,
                    reconstruction=reproduced_extra,
                    explicit_minus_saved=reproduced_explicit - mixed["explicit_radial"],
                    reconstruction_minus_saved=reproduced_extra
                    - mixed["reconstruction_radial"],
                ),
                reconstruction_representation_residual=representation,
                counts_semantics=c["counts_semantics"],
                source_map_sha256=c["source_map_sha256"],
                driver_sha256=sha(__file__),
                contract_sha256=sha(HERE / "contract.json"),
                binding_sha256=sha(binding_path),
                inputs_sha256={
                    key: row["sha256"] for key, row in c["input_roles"].items()
                },
                ordinary_FD_passed=False,
                original_two_failures_preserved=True,
                no_owner_constructed=True,
                gradient_repaired=False,
                optimizer_qualified=False,
                certified_arithmetic_bound=False,
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
                completed_radial_keys=len(primitives),
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
            "SOURCE only: 0 NPZ hashes/header/body access; 0 numerical imports/calls."
        )
