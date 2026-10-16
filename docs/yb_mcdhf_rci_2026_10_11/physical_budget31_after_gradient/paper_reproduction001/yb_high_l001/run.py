"""Restore true historical Yb68 DF; qualify complete current VN g bands only."""

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
            raise RuntimeError("input changed: " + path)
    binding = contract["source_binding"]
    if sha(HERE / binding["path"]) != binding["sha256"]:
        raise RuntimeError("source epoch metadata changed")
    assert contract["g_partners"] == [4, -5]
    assert contract["energy_band_upper_Ha"] == 50.0
    assert contract["historical_model"]["electrons"] == 68
    return json.loads((HERE / binding["path"]).read_text())


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
        occupied_diagnostics=[],
        g_channels=[],
        scope=contract["scope"],
        current_stationarity_qualified=False,
        high_l_spectral_response_calculated=False,
    )

    def write(name):
        report["elapsed_seconds_uncontrolled"] = time.monotonic() - started
        report["peak_RSS_MiB"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        )
        (actual / name).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

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
        from atomx.numerics.bspline.dirac_assembly import _get_keep_grid
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
                    and not fortran,
                )
        entry("input_NPZ", 1)
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
            source_role="restored historical converged DF19, not a fresh stationary result",
            core_radial_sha256=before_radial,
            nuclear_profile=info["nuclear_profile"],
        )
        for name in ("SCF", "CI", "optimizer"):
            report["counts"][name] = 0

        def forbidden(*_, **__):
            raise RuntimeError(
                "new SCF/CI/optimizer forbidden in frozen-core qualification"
            )

        df.DFSolver.solve = forbidden
        cache = ConvolCache(data)
        operators = {}
        spectra = {}
        original_fock = mbpt._build_closed_shell_fock

        def fock(c, k, **kwargs):
            entry("Fock", 9)
            if c is not core or kwargs.get("potential_mode") != "VN":
                raise RuntimeError("full restored68e VN reference required")
            h, s = original_fock(c, k, **kwargs)
            operators[k] = h, s
            completed("Fock")
            return h, s

        mbpt._build_closed_shell_fock = fock
        original_raw = mbpt.build_dirac_matrices

        def raw_hs(*args, **kwargs):
            entry("raw_hS", 9)
            v = original_raw(*args, **kwargs)
            completed("raw_hS")
            return v

        mbpt.build_dirac_matrices = raw_hs
        for name, maximum in [
            ("compute_direct_potential_cell", 171),
            ("compute_exchange_matrix_cell", 264),
        ]:
            native = getattr(mbpt, name)

            def wrap(*args, _native=native, _name=name, _max=maximum, **kwargs):
                entry(_name, _max)
                value = _native(*args, **kwargs)
                completed(_name)
                return value

            setattr(mbpt, name, wrap)
        native_solve = mbpt.solve_dirac

        def dirac_solve(h, s, ns, k, **kwargs):
            entry("Dirac_spectrum", 2)
            result = native_solve(h, s, ns, k, **kwargs)
            spectra[k] = result
            completed("Dirac_spectrum")
            return result

        mbpt.solve_dirac = dirac_solve
        native_eigh = la.eigh

        def eigh(*args, **kwargs):
            entry("Dirac_LAPACK", 6)
            value = native_eigh(*args, **kwargs)
            completed("Dirac_LAPACK")
            return value

        la.eigh = eigh
        native_linear = la.solve

        def metric_solve(s, d):
            entry("metric_linear_solves", 11)
            value = native_linear(s, d, assume_a="pos", check_finite=True)
            completed("metric_linear_solves")
            return value

        keep, grid = _get_keep_grid(123, 122, 123, 1, 2)
        output = {
            "occupied_keys": np.array([(x.n, x.kappa) for x in shells]),
            "occupied_P": np.array([x[0] for x in preserved]),
            "occupied_Q": np.array([x[1] for x in preserved]),
            "nuclear_grid": np.array(
                [core._nuclear_potential(float(r)) for r in data["gr"].ravel()]
            ).reshape(data["gr"].shape),
        }

        def cache_payload():
            seen = set()

            def total(x):
                if id(x) in seen:
                    return 0
                seen.add(id(x))
                if isinstance(x, np.ndarray):
                    return x.nbytes
                if isinstance(x, dict):
                    return sum(total(v) for v in x.values())
                if isinstance(x, (tuple, list)):
                    return sum(total(v) for v in x)
                return 0

            value = total(cache._cache)
            gate(
                "bounded_convol_cache_array_payload",
                value,
                contract["entry_caps"]["convol_cache_array_payload_bytes"],
            )
            return value

        for k in sorted(raw):
            h, s = fock(core, k, cache=cache, potential_mode="VN")
            hr = np.tril(h[grid]) + np.tril(h[grid], -1).T
            sr = s[grid]
            selected = [x for x in shells if x.kappa == k]
            v = np.column_stack([np.r_[p, q][keep] for p, q, _ in raw[k]])
            gate(
                "occupied_S_Gram:" + str(k),
                float(np.max(abs(v.T @ sr @ v - np.eye(len(selected))))),
                contract["gates"]["full_S_Gram"],
            )
            oldeps = np.array([core.orbital_energies[x.label] for x in selected])
            rayleigh = np.sum(v * (hr @ v), axis=0)
            d = np.column_stack(
                [hr @ v - (sr @ v) * oldeps, hr @ v - (sr @ v) * rayleigh]
            )
            norms = np.sqrt(np.maximum(0.0, np.sum(d * metric_solve(sr, d), axis=0)))
            if not np.all(np.isfinite(norms)):
                raise FloatingPointError("nonfinite occupied diagnostic")
            for i, sh in enumerate(selected):
                report["occupied_diagnostics"].append(
                    dict(
                        key=[sh.n, k],
                        label=sh.label,
                        old_epsilon_Ha=float(oldeps[i]),
                        current_rayleigh_Ha=float(rayleigh[i]),
                        old_energy_dual_residual_Ha=float(norms[i]),
                        rayleigh_dual_residual_Ha=float(norms[i + len(selected)]),
                    )
                )
            output["Fock_raw_" + str(k)] = h
            output["S_" + str(k)] = s
            report["cache_array_payload_bytes"] = cache_payload()
            loaded_guard()
            write("partial.json")
        for k in contract["g_partners"]:
            occupied, band = mbpt.build_virtual_basis(
                core,
                k,
                cache=cache,
                e_max=50.0,
                potential_mode="VN",
                orthogonal_to_occupied=True,
            )
            gate("no_occupied_g:" + str(k), not occupied)
            result = spectra[k]
            e = result.eigenvalues
            v = result.coefficients[keep, :]
            h, s = operators[k]
            hr = np.tril(h[grid]) + np.tril(h[grid], -1).T
            sr = s[grid]
            mask = e < 50.0
            band_e = np.array([x[2] for x in band])
            gate("whole_electron_branch:" + str(k), bool(np.all(e > -C_SQ)))
            gate("complete_declared_band:" + str(k), np.array_equal(band_e, e[mask]))
            gate("nonempty_g_band:" + str(k), len(band) > 0)
            gate(
                "full_electron_S_Gram:" + str(k),
                float(np.max(abs(v.T @ sr @ v - np.eye(len(e))))),
                contract["gates"]["full_S_Gram"],
            )
            u = la.cholesky(sr, lower=True).T @ v
            projector = u @ u.T
            gate(
                "whole_electron_metric_projector:" + str(k),
                float(np.max(abs(projector @ projector - projector))),
                contract["gates"]["electron_projector_idempotence_metric_frame"],
            )
            d = hr @ v - (sr @ v) * e
            residual = np.sqrt(np.maximum(0.0, np.sum(d * metric_solve(sr, d), axis=0)))
            gate(
                "complete50Ha_band_dual_residual:" + str(k),
                float(np.max(residual[mask])),
                contract["gates"]["band_dual_residual_Ha"],
            )
            p = np.array([x[0] for x in band])
            q = np.array([x[1] for x in band])
            norms = np.array(
                [
                    np.sum(
                        (eval_P(pp, data) ** 2 + eval_Q(qq, data) ** 2) * data["grw"]
                    )
                    for pp, qq in zip(p, q, strict=True)
                ]
            )
            gate(
                "complete50Ha_band_Gauss_norm:" + str(k),
                float(np.max(abs(norms - 1))),
                contract["gates"]["sampled_norm"],
            )
            channel = dict(
                kappa=k,
                electron_branch_roots=len(e),
                band_roots=len(band),
                bound_roots=int(np.count_nonzero(band_e < 0)),
                minimum_band_energy_Ha=float(band_e.min()),
                maximum_band_energy_Ha=float(band_e.max()),
                full_branch_max_dual_residual_Ha=float(residual.max()),
                band_max_dual_residual_Ha=float(residual[mask].max()),
                band_selection="all native roots below50Ha",
                spectral_response=False,
            )
            report["g_channels"].append(channel)
            output.update(
                {
                    "Fock_raw_" + str(k): h,
                    "S_" + str(k): s,
                    "full_electron_E_" + str(k): e,
                    "full_electron_V_" + str(k): result.coefficients,
                    "band_E_" + str(k): band_e,
                    "band_P_" + str(k): p,
                    "band_Q_" + str(k): q,
                    "full_dual_residual_" + str(k): residual,
                    "band_Gauss_norm_" + str(k): norms,
                    "metric_electron_projector_" + str(k): projector,
                }
            )
            report["cache_array_payload_bytes"] = cache_payload()
            loaded_guard()
            write("partial.json")
        after = hashlib.sha256(
            b"".join(x.tobytes() for pair in preserved for x in pair)
        ).hexdigest()
        gate("historical19core_radials_unchanged", after == before_radial)
        gate(
            "historical_DF_markers_preserved",
            core.converged == info["converged"] and core.residual == info["residual"],
        )
        metadata(contract)
        report["loaded_source_after"] = loaded_guard()
        np.savez_compressed(actual / "complete_arrays.npz", **output)
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
        if "output" in locals():
            for k, (h, s) in operators.items():
                output["Fock_raw_" + str(k)] = h
                output["S_" + str(k)] = s
            for k, spec in spectra.items():
                output["full_electron_E_" + str(k)] = spec.eigenvalues
                output["full_electron_V_" + str(k)] = spec.coefficients
            np.savez_compressed(actual / "failed_arrays.npz", **output)
            report["failed_arrays_sha256"] = sha(actual / "failed_arrays.npz")
        write("failure.json")
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    contract = json.loads((HERE / "contract.json").read_text())
    epoch = metadata(contract)
    if args.execute:
        execute(contract, epoch)
    else:
        print(
            json.dumps(
                {
                    "SOURCE_metadata_PASS": True,
                    "inputs": len(contract["inputs"]),
                    "numerical_imports": 0,
                    "source_epoch_entries": len(epoch),
                }
            )
        )
