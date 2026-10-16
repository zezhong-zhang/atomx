"""Restore historical Yb68 frozen bank; qualify joint full-active complete VN g bands."""

import argparse
import hashlib
import json
import os
import resource
import signal
import sys
import time
import traceback
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("/data/code/atomx").resolve()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata(contract):
    for path, info in contract["inputs"].items():
        if sha(ROOT / path) != info["sha256"]:
            raise RuntimeError("historical input changed: " + path)
    for group in ("frozen_predecessor", "unused004", "block_diagnosis"):
        for name, binding in contract[group].items():
            if sha(HERE / binding["path"]) != binding["sha256"]:
                raise RuntimeError(group + " changed: " + name)
    for key in ("header_metadata", "predecessor_header_metadata"):
        binding = contract[key]
        if sha(HERE / binding["path"]) != binding["sha256"]:
            raise RuntimeError("header metadata changed: " + key)
    old = json.loads(
        (HERE / contract["frozen_predecessor"]["contract"]["path"]).read_text()
    )
    for key in (
        "historical_model",
        "inputs",
        "workspace_whole_field_plan",
        "gates",
        "resources",
        "g_partners",
        "energy_band_upper_Ha",
        "original_driver_archive_gap",
    ):
        if contract[key] != old[key]:
            raise RuntimeError("unreviewed model/gate difference: " + key)
    binding = contract["source_chain"]["historical_available_epoch"]
    epoch = json.loads((HERE / binding["path"]).read_text())
    for path, digest in contract["source_chain"]["current_numerical_overrides"].items():
        if sha(ROOT / path) != digest:
            raise RuntimeError("current numerical source changed: " + path)
        epoch[path] = digest
    ready = all(contract["admission"].values())
    if ready:
        kernel = contract["kernel"]
        if (
            not kernel["source_sha256"]
            or epoch[kernel["source_path"]] != kernel["source_sha256"]
        ):
            raise RuntimeError("unbound kernel source")
        for path, digest in kernel["test_bindings"].items():
            if sha(ROOT / path) != digest:
                raise RuntimeError("kernel test binding changed: " + path)
        if not kernel["test_bindings"]:
            raise RuntimeError("kernel tests not bound")
        for key in (
            "kernel_eigh",
            "kernel_solve",
            "kernel_cholesky",
            "kernel_solve_triangular",
            "kernel_qr",
            "kernel_norm",
        ):
            if (
                type(contract["entry_caps"][key]) is not int
                or contract["entry_caps"][key] < 0
            ):
                raise RuntimeError("unbound kernel work cap: " + key)
    return epoch, ready


def execute(contract, epoch):
    actual = HERE / "actual"
    if actual.exists():
        raise RuntimeError("closed evidence directory; no automatic retry")
    cap = contract["resources"]
    if os.sched_getaffinity(0) != {cap["CPU"]}:
        raise RuntimeError("declared one CPU required")
    if any(os.environ.get(k) != "1" for k in cap["thread_environment"]):
        raise RuntimeError("serial thread environment required")
    available = next(
        int(x.split()[1]) * 1024
        for x in Path("/proc/meminfo").read_text().splitlines()
        if x.startswith("MemAvailable:")
    )
    if (
        os.getloadavg()[0] > cap["maximum_load1"]
        or available < cap["minimum_available_bytes"]
    ):
        raise RuntimeError("host admission failed")
    resource.setrlimit(resource.RLIMIT_AS, (cap["AS_bytes"], cap["AS_bytes"]))

    def alarm(*_):
        raise TimeoutError("inner600-second cap")

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(cap["inner_seconds"])
    actual.mkdir()
    started = time.monotonic()
    report = dict(
        status="RUNNING",
        contract_sha256=sha(HERE / "contract.json"),
        driver_sha256=sha(HERE / "run.py"),
        host_load=os.getloadavg(),
        available_bytes=available,
        counts={},
        checks=[],
        occupied_diagnostics=json.loads(
            (HERE / contract["frozen_predecessor"]["failure"]["path"]).read_text()
        )["occupied_diagnostics"],
        occupied_diagnostics_scope="archived003 source-bound only; no new occupied Fock diagnostic",
        g_channels=[],
        block_channels=[],
        scope=contract["scope"],
        current_stationarity_qualified=False,
        high_l_spectral_response_calculated=False,
    )

    def serial(value):
        if isinstance(value, dict):
            return {str(k): serial(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [serial(v) for v in value]
        if type(value) in (str, int, float, bool, type(None)):
            return value
        if hasattr(value, "tolist"):
            return serial(value.tolist())
        if hasattr(value, "item"):
            return serial(value.item())
        raise TypeError("unsupported result export: " + type(value).__name__)

    def write(name):
        report["elapsed_seconds_uncontrolled"] = time.monotonic() - started
        report["peak_RSS_MiB"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        )
        (actual / name).write_text(
            json.dumps(serial(report), indent=2, allow_nan=False) + "\n"
        )

    def entry(name, maximum):
        report["counts"][name] = report["counts"].get(name, 0) + 1
        if report["counts"][name] > maximum:
            raise RuntimeError("entry cap: " + name)

    def completed(name):
        report["counts"][name + "_completed"] = (
            report["counts"].get(name + "_completed", 0) + 1
        )

    def gate(name, observed, limit=None):
        if hasattr(observed, "item"):
            observed = observed.item()
        ok = bool(observed) if limit is None else bool(observed <= limit)
        report["checks"].append(
            dict(name=name, observed=observed, limit=limit, PASS=ok)
        )
        if not ok:
            raise RuntimeError("gate failed: " + name)

    def loaded_guard():
        bound = {}
        for name, module in tuple(sys.modules.items()):
            p = getattr(module, "__file__", None)
            if name.startswith("atomx") and p and Path(p).suffix == ".py":
                path = Path(p).resolve()
                if not path.is_relative_to(ROOT / "src/atomx"):
                    raise RuntimeError("non-native module origin: " + name)
                rel = str(path.relative_to(ROOT))
                digest = sha(path)
                if epoch.get(rel) != digest:
                    raise RuntimeError("loaded source changed: " + rel)
                bound[rel] = digest
        return bound

    try:
        import numpy as np
        import scipy.linalg as la

        sys.path.insert(0, str(ROOT / "src"))
        import atomx.methods.dirac_fock as df
        import atomx.methods.mbpt as mbpt
        from atomx.calculation_spec import CalculationSpec
        from atomx.config.subshell import Subshell
        from atomx.constants import C_SQ
        from atomx.numerics.bspline.convol import ConvolCache
        from atomx.numerics.bspline.dirac_assembly import (
            _get_keep_grid,
            refine_generalized_eigenblock,
        )
        from atomx.numerics.bspline.evaluate import eval_P, eval_Q
        from atomx.potentials.nuclear import make_nuclear_potential, nuclear_profile

        report["loaded_source_before"] = loaded_guard()
        write("partial.json")

        def owned(a):
            v = np.asarray(a)
            return np.frombuffer(v.tobytes(), dtype=v.dtype).reshape(v.shape)

        prefix = ROOT / "docs/rare_earth_yb_spectrum_2026_10_08/results/v68_default"
        info = json.loads((prefix / "core.json").read_text())
        structure = json.loads((prefix / "structure.json").read_text())
        gate("legacy_DF_converged_marker", info["converged"])
        gate(
            "legacy_DF_residual",
            info["residual"],
            contract["gates"]["legacy_core_tolerance"],
        )
        gate(
            "exact_restored_nuclear_profile",
            nuclear_profile(70, "fermi") == info["nuclear_profile"],
        )
        archive = prefix / "orbitals_workspace.npz"
        plans = contract["workspace_whole_field_plan"]
        with zipfile.ZipFile(archive) as z:
            gate(
                "exact_whole20field_census",
                set(z.namelist()) == {k + ".npy" for k in plans},
            )
            gate(
                "bounded_decoded_workspace_bytes",
                sum(x.file_size for x in z.infolist()),
                contract["workspace_decoded_bytes_cap"],
            )
            for field, plan in plans.items():
                with z.open(field + ".npy") as f:
                    version = np.lib.format.read_magic(f)
                    reader = (
                        np.lib.format.read_array_header_1_0
                        if version == (1, 0)
                        else np.lib.format.read_array_header_2_0
                    )
                    shape, fortran, dtype = reader(f, max_header_size=8192)
                gate(
                    "field_header:" + field,
                    list(shape) == plan["shape"]
                    and dtype.str == plan["dtype"]
                    and fortran == plan["fortran_order"],
                )
        entry("input_NPZ", 2)
        with np.load(archive, allow_pickle=False) as z:
            arrays = {k: owned(z[k]) for k in plans if k not in ("energies", "mixing")}
        completed("input_NPZ")
        data = {
            k[6:]: v.item() if v.ndim == 0 else v
            for k, v in arrays.items()
            if k.startswith("basis_")
        }
        for k, plan in plans.items():
            if "value" in plan:
                gate("geometry_scalar:" + k, arrays[k].item() == plan["value"])
        gate(
            "exact_saved_knot_identity",
            hashlib.sha256(data["t"].tobytes()).hexdigest()
            == info["calculation_spec"]["numerics"]["basis_knots"]["sha256"],
        )
        shells = [
            Subshell(n=int(n), kappa=int(k), occupation=float(q))
            for n, k, q in info["calculation_spec"]["state"]["subshells"]
        ]
        keys = [tuple(map(int, k)) for k in arrays["orbital_keys"]]
        gate("whole24distinct_radial_keys", len(keys) == len(set(keys)) == 24)
        index = {k: i for i, k in enumerate(keys)}
        raw = {}
        preserved = []
        for sh in shells:
            i = index[sh.n, sh.kappa]
            p, q = owned(arrays["P"][i]), owned(arrays["Q"][i])
            raw.setdefault(sh.kappa, []).append((p, q, sh.occupation))
            preserved.append((p, q))
        core = df.SCFResult(
            converged=info["converged"],
            iterations=info["iterations"],
            residual=info["residual"],
            total_energy=info["total_energy_hartree"],
            orbital_energies=structure["orbital_energies_hartree"],
            subshells=shells,
            orbitals_raw=raw,
            Z=70,
            _bsp_data=data,
            solver_diagnostics=info["solver_diagnostics"],
            calculation_spec=CalculationSpec(**info["calculation_spec"]),
        )
        core._nuclear_potential = make_nuclear_potential(70, "fermi")
        before_radial = hashlib.sha256(
            b"".join(x.tobytes() for pair in preserved for x in pair)
        ).hexdigest()
        report["legacy_DF"] = dict(
            converged=core.converged,
            residual=core.residual,
            total_energy_Ha=core.total_energy,
            source_role="historical metadata marks19core orbitals asDHF; restored owned frozen bank, no current stationarity qualification",
            core_radial_sha256=before_radial,
            nuclear_profile=info["nuclear_profile"],
        )
        capsule_plan = json.loads(
            (HERE / contract["predecessor_header_metadata"]["path"]).read_text()
        )
        capsule_path = HERE / contract["frozen_predecessor"]["failed_arrays"]["path"]
        with zipfile.ZipFile(capsule_path) as z:
            gate(
                "exact003_whole28field_census",
                set(z.namelist())
                == {k + ".npy" for k in capsule_plan["whole_field_plan"]},
            )
            gate(
                "capsule_whole_bytes",
                sum(x.file_size for x in z.infolist()),
                contract["capsule_decoded_bytes_cap"],
            )
            for field, plan in capsule_plan["whole_field_plan"].items():
                with z.open(field + ".npy") as f:
                    version = np.lib.format.read_magic(f)
                    reader = (
                        np.lib.format.read_array_header_1_0
                        if version == (1, 0)
                        else np.lib.format.read_array_header_2_0
                    )
                    shape, fortran, dtype = reader(f, max_header_size=8192)
                gate(
                    "capsule_field:" + field,
                    list(shape) == plan["shape"]
                    and dtype.str == plan["dtype"]
                    and fortran == plan["fortran_order"],
                )
        entry("input_NPZ", 2)
        with np.load(capsule_path, allow_pickle=False) as z:
            capsule = {k: owned(z[k]) for k in capsule_plan["deferred_decode"]}
        completed("input_NPZ")
        original_keys = np.array([(sh.n, sh.kappa) for sh in shells])
        gate(
            "archived19keys_exact",
            np.array_equal(capsule["occupied_keys"], original_keys),
        )
        gate(
            "archived19P_exact",
            np.array_equal(capsule["occupied_P"], np.array([x[0] for x in preserved])),
        )
        gate(
            "archived19Q_exact",
            np.array_equal(capsule["occupied_Q"], np.array([x[1] for x in preserved])),
        )
        for name in ("SCF", "CI", "optimizer"):
            report["counts"][name] = 0

        def forbidden(*_, **__):
            raise RuntimeError(
                "new SCF/CI/optimizer forbidden in frozen-bank qualification"
            )

        df.DFSolver.solve = forbidden
        cache = ConvolCache(data)
        original_raw = mbpt.build_dirac_matrices

        def raw_hs(*args, **kwargs):
            entry("raw_hS", 1)
            result = original_raw(*args, **kwargs)
            completed("raw_hS")
            return result

        mbpt.build_dirac_matrices = raw_hs
        for name, maximum in [
            ("compute_direct_potential_cell", 19),
            ("compute_exchange_matrix_cell", 34),
        ]:
            native = getattr(mbpt, name)

            def wrap(*args, _native=native, _name=name, _maximum=maximum, **kwargs):
                entry(_name, _maximum)
                result = _native(*args, **kwargs)
                completed(_name)
                return result

            setattr(mbpt, name, wrap)
        native_linear = la.solve
        native_cholesky = la.cholesky
        kernel_active = [False]
        for name in ("eigh", "solve", "cholesky", "solve_triangular", "qr", "norm"):
            native = getattr(la, name)

            def instrument(*args, _native=native, _name=name, **kwargs):
                if kernel_active[0]:
                    count = "kernel_" + _name
                    entry(count, contract["entry_caps"][count])
                    if _name != "norm":
                        gate(
                            "kernel_operand_row_cap:" + _name,
                            args[0].shape[0],
                            contract["kernel"]["dimension_cap"] + 1,
                        )
                elif _name == "eigh":
                    count = "native_Dirac_LAPACK_attempts"
                    entry(count, 3)
                else:
                    raise RuntimeError("unbudgeted outside-kernel LAPACK: " + _name)
                result = _native(*args, **kwargs)
                completed(count)
                return result

            setattr(la, name, instrument)

        def metric_solve(s, d):
            entry("metric_linear_solves", 2)
            result = native_linear(s, d, assume_a="pos", check_finite=True)
            completed("metric_linear_solves")
            return result

        keep, grid = _get_keep_grid(123, 122, 123, 1, 2)
        gate("same239_boundary_layout", len(keep) == 239)
        output = dict(
            occupied_keys=original_keys,
            occupied_P=np.array([x[0] for x in preserved]),
            occupied_Q=np.array([x[1] for x in preserved]),
        )
        for k in contract["g_partners"]:
            if k == 4:
                h, s = capsule["Fock_raw_4"], capsule["S_4"]
                e0 = capsule["original_full_electron_E_4"]
                full0 = capsule["original_full_electron_V_4"]
            else:
                gate("no_occupied_g_partner", k == -5 and k not in raw)
                entry("Fock", 1)
                h, s = mbpt._build_closed_shell_fock(
                    core, k, cache=cache, potential_mode="VN"
                )
                completed("Fock")
                entry("Dirac_spectrum", 1)
                solved = mbpt.solve_dirac(
                    h, s, 123, k, nsp=122, nsq=123, orthogonal_to=None
                )
                completed("Dirac_spectrum")
                e0, full0 = owned(solved.eigenvalues), owned(solved.coefficients)
            output.update(
                {
                    "Fock_raw_" + str(k): h,
                    "S_" + str(k): s,
                    "original_full_electron_E_" + str(k): e0,
                    "original_full_electron_V_" + str(k): full0,
                }
            )
            gate("original_electron_count_cap:" + str(k), len(e0), 123)
            gate("original_electron_branch:" + str(k), bool(np.all(e0 > -C_SQ)))
            gate("original_root_order:" + str(k), bool(np.all(np.diff(e0) > 0)))
            v0 = full0[keep, :]
            hr = np.tril(h[grid]) + np.tril(h[grid], -1).T
            sr = s[grid]
            mask0 = e0 < 50.0
            indices = np.flatnonzero(mask0)
            leading = len(indices)
            gate("nonempty_complete_band:" + str(k), leading > 0)
            gate(
                "complete_leading_band:" + str(k),
                np.array_equal(indices, np.arange(leading)),
            )
            if k == 4:
                gate(
                    "exact_original67band_indices",
                    np.array_equal(indices, capsule["fixed_band_indices_4"]),
                )
            gate(
                "original_whole_SGram:" + str(k),
                float(np.max(abs(v0.T @ sr @ v0 - np.eye(len(e0))))),
                contract["gates"]["full_S_Gram"],
            )
            before = dict(report["counts"])
            entry("block_kernel_calls", 2)
            kernel_active[0] = True
            try:
                e1, v1, block_info = refine_generalized_eigenblock(
                    owned(hr),
                    owned(sr),
                    owned(e0),
                    owned(v0),
                    leading_count=leading,
                    dimension_cap=contract["kernel"]["dimension_cap"],
                    column_cap=contract["kernel"]["column_cap"],
                    memory_cap_bytes=contract["kernel"]["memory_cap_bytes"],
                )
            finally:
                kernel_active[0] = False
            completed("block_kernel_calls")
            per_channel_caps = dict(
                kernel_eigh=1,
                kernel_solve=2 * leading,
                kernel_cholesky=2,
                kernel_solve_triangular=3 * leading + 2,
                kernel_qr=0,
                kernel_norm=3 * leading,
            )
            for name, maximum in per_channel_caps.items():
                actual_count = report["counts"].get(name, 0) - before.get(name, 0)
                gate(
                    "kernel_per_channel_cap:" + str(k) + ":" + name,
                    actual_count,
                    maximum,
                )
            gate(
                "complete_returned_dimensions:" + str(k),
                e1.shape == e0.shape and v1.shape == v0.shape,
            )
            gate(
                "finite_real_block:" + str(k),
                bool(
                    np.isrealobj(e1)
                    and np.isrealobj(v1)
                    and np.all(np.isfinite(e1))
                    and np.all(np.isfinite(v1))
                ),
            )
            full1 = np.zeros_like(full0)
            full1[keep, :] = v1
            output.update(
                {
                    "full_electron_E_" + str(k): e1,
                    "full_electron_V_" + str(k): full1,
                    "fixed_band_indices_" + str(k): indices,
                }
            )
            gate("original_branch_retained:" + str(k), bool(np.all(e1 > -C_SQ)))
            gate("all_root_order_retained:" + str(k), bool(np.all(np.diff(e1) > 0)))
            gate("band_membership_retained:" + str(k), np.array_equal(e1 < 50.0, mask0))
            gaps = np.minimum(np.r_[np.inf, np.diff(e0)], np.r_[np.diff(e0), np.inf])
            overlap = np.sum(v0 * (sr @ v1), axis=0)
            change = abs(e1 - e0) / gaps
            gate(
                "original_all_column_overlap:" + str(k),
                bool(abs(overlap).min() >= contract["root_identity"]["overlap_min"]),
            )
            gate(
                "original_all_root_gap_fraction:" + str(k),
                float(change.max()),
                contract["root_identity"]["maximum_original_gap_fraction_energy_shift"],
            )
            gram_error = float(np.max(abs(v1.T @ sr @ v1 - np.eye(len(e1)))))
            gate(
                "full_electron_S_Gram:" + str(k),
                gram_error,
                contract["gates"]["full_S_Gram"],
            )
            lower = native_cholesky(sr, lower=True, check_finite=False)
            u0, u1 = lower.T @ v0, lower.T @ v1
            projector0, projector1 = u0 @ u0.T, u1 @ u1.T
            projector_error = float(np.max(abs(projector1 @ projector1 - projector1)))
            gate(
                "whole_electron_metric_projector:" + str(k),
                projector_error,
                contract["gates"]["electron_projector_idempotence_metric_frame"],
            )
            d0, d1 = hr @ v0 - (sr @ v0) * e0, hr @ v1 - (sr @ v1) * e1
            combined = np.column_stack([d0, d1])
            norms = np.sqrt(
                np.maximum(0.0, np.sum(combined * metric_solve(sr, combined), axis=0))
            )
            residual0, residual1 = np.split(norms, 2)
            gate(
                "finite_full_ordinary_residuals:" + str(k),
                bool(np.all(np.isfinite(norms))),
            )
            maximum = float(residual1[mask0].max())
            limit = contract["gates"]["band_dual_residual_Ha"]
            report["checks"].append(
                dict(
                    name="complete50Ha_band_dual_residual:" + str(k),
                    observed=maximum,
                    limit=limit,
                    PASS=bool(maximum <= limit),
                )
            )
            p, q = full1[:123, mask0].T, full1[123:246, mask0].T
            sampled = np.array(
                [
                    np.sum(
                        (eval_P(pp, data) ** 2 + eval_Q(qq, data) ** 2) * data["grw"]
                    )
                    for pp, qq in zip(p, q, strict=True)
                ]
            )
            gate(
                "complete_band_Gauss_norm:" + str(k),
                float(abs(sampled - 1).max()),
                contract["gates"]["sampled_norm"],
            )
            outside0 = v1 - v0 @ (v0.T @ (sr @ v1))
            outside_norm = np.sqrt(
                np.maximum(0.0, np.sum(outside0 * (sr @ outside0), axis=0))
            )
            output.update(
                {
                    "band_E_" + str(k): e1[mask0],
                    "band_P_" + str(k): p,
                    "band_Q_" + str(k): q,
                    "band_Gauss_norm_" + str(k): sampled,
                    "original_full_dual_residual_" + str(k): residual0,
                    "full_dual_residual_" + str(k): residual1,
                    "original_root_separation_" + str(k): gaps,
                    "original_column_overlap_" + str(k): overlap,
                    "refined_energy_shift_" + str(k): e1 - e0,
                    "metric_electron_projector_" + str(k): projector1,
                    "original_metric_electron_projector_" + str(k): projector0,
                    "outside_original_span_metric_norm_" + str(k): outside_norm,
                }
            )
            report["block_channels"].append(
                dict(
                    kappa=k,
                    info=block_info,
                    completed_kernel_work={
                        name: report["counts"].get(name, 0) - before.get(name, 0)
                        for name in per_channel_caps
                    },
                )
            )
            report["g_channels"].append(
                dict(
                    kappa=k,
                    electron_branch_roots=len(e1),
                    band_roots=leading,
                    bound_roots=int(np.count_nonzero(e1[mask0] < 0)),
                    minimum_band_energy_Ha=float(e1[mask0].min()),
                    maximum_band_energy_Ha=float(e1[mask0].max()),
                    original_band_max_dual_residual_Ha=float(residual0[mask0].max()),
                    band_max_dual_residual_Ha=maximum,
                    full_branch_max_dual_residual_Ha=float(residual1.max()),
                    full_SGram_error=gram_error,
                    metric_projector_error=projector_error,
                    original_projector_max_entry_change=float(
                        abs(projector1 - projector0).max()
                    ),
                    maximum_outside_original_span_Snorm=float(outside_norm.max()),
                    minimum_original_column_absolute_overlap=float(abs(overlap).min()),
                    maximum_root_shift_gap_ratio=float(change.max()),
                    kernel_vs_independent_full_residual_max_difference_Ha=float(
                        abs(
                            residual1 - np.asarray(block_info["full_dual_residuals_Ha"])
                        ).max()
                    ),
                    band_selection="every original electron root E<50Ha",
                    spectral_response=False,
                )
            )
            seen = set()

            def payload(x, _seen=seen):
                if id(x) in _seen:
                    return 0
                _seen.add(id(x))
                if isinstance(x, np.ndarray):
                    return x.nbytes
                if isinstance(x, dict):
                    return sum(payload(v) for v in x.values())
                if isinstance(x, (tuple, list)):
                    return sum(payload(v) for v in x)
                return 0

            report["cache_array_payload_bytes"] = payload(cache._cache)
            gate(
                "convol_cache_array_payload_cap",
                report["cache_array_payload_bytes"],
                contract["entry_caps"]["convol_cache_array_payload_bytes"],
            )
            report["loaded_source_current"] = loaded_guard()
            write("partial.json")
        after = hashlib.sha256(
            b"".join(x.tobytes() for pair in preserved for x in pair)
        ).hexdigest()
        gate("historical19core_radials_unchanged", after == before_radial)
        gate(
            "historical_markers_unchanged",
            core.converged == info["converged"] and core.residual == info["residual"],
        )
        metadata(contract)
        report["loaded_source_after"] = loaded_guard()
        if any(not item["PASS"] for item in report["checks"]):
            raise RuntimeError(
                "unchanged complete-band Sdual gate failed; both g channels retained"
            )
        entry("output_NPZ", 1)
        np.savez_compressed(actual / "complete_arrays.npz", **output)
        completed("output_NPZ")
        report["output_arrays_sha256"] = sha(actual / "complete_arrays.npz")
        report["output_arrays_fields"] = {
            k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in output.items()
        }
        report["status"] = "PASS"
        write("result.json")
    except BaseException as exc:
        report["status"] = "FAIL"
        report["exception"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        if "output" in locals() and not report["counts"].get("output_NPZ", 0):
            entry("output_NPZ", 1)
            np.savez_compressed(actual / "failed_arrays.npz", **output)
            completed("output_NPZ")
            report["failed_arrays_sha256"] = sha(actual / "failed_arrays.npz")
        write("failure.json")
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    contract = json.loads((HERE / "contract.json").read_text())
    epoch, ready = metadata(contract)
    if args.execute:
        if not ready:
            raise RuntimeError(
                "shared kernel/source/tests and parent admission not sealed"
            )
        execute(contract, epoch)
    else:
        print(
            json.dumps(
                dict(
                    SOURCE_metadata_PASS=True,
                    numerical_imports=0,
                    execution_ready=ready,
                    numerical_execution_performed=False,
                    inherited_source_epoch_entries=len(epoch),
                ),
                indent=2,
            )
        )
