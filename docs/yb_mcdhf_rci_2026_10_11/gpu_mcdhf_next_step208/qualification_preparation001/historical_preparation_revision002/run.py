"""PRIVATE native31 CPU1/CPU4/CUDA correctness preparation; default is metadata.

No numerical imports or array bytes are accessed before a separate exact root
seal and the actual native31 directional prerequisite have been admitted.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import resource
import signal
import site
import sys
import time
import traceback
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).absolute().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").is_file())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    # np.longdouble.item()/tolist() can return the same NumPy scalar.
    if type(value).__module__.startswith("numpy") and getattr(value, "ndim", None) == 0:
        kind = value.dtype.kind
        return bool(value) if kind == "b" else int(value) if kind in "iu" else float(value)
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    return value


def write(path, value):
    temp = path.with_name(path.name + ".part")
    temp.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def metadata(*, arrays=False):
    contract = load(HERE / "contract.json")
    roles = {name: ROOT / row["path"] for name, row in contract["input_roles"].items()}
    deferred = []
    for name, row in contract["input_roles"].items():
        if roles[name].suffix == ".npz" and not arrays:
            deferred.append(name)
        elif sha(roles[name]) != row["sha256"]:
            raise ValueError("bound input changed: " + name)
    identity = load(roles["f043_identity"])
    source = ROOT / contract["source_archive"]
    if (identity["source_map_sha256"] != contract["fixed"]["source_map_sha256"]
            or len(identity["all_atomx_python_sha256"]) != 356):
        raise ValueError("full f043356 source map required")
    for name, value in identity["all_atomx_python_sha256"].items():
        if sha(source / name.removeprefix("src/")) != value:
            raise ValueError("frozen source changed: " + name)
    reference = load(roles["native31_result"])
    if (not reference["all_declared_controls_passed"] or len(reference["checks"]) != 438
            or any(not row["passed"] for row in reference["checks"])
            or reference["source_map_sha256"] != contract["fixed"]["source_map_sha256"]
            or reference["finite_CI_response_qualified"] or reference["expanded_MCDHF_optimized"]):
        raise ValueError("closed HF-partial reference scope changed")
    return contract, roles, source, deferred


def pointer(value, names):
    for name in names:
        value = value[name]
    return value


def admit_directional(bound, fixed):
    """Explicit root-filled real-result roles; no old29 or made-up certificate."""
    required = {"result", "domain_ledger", "root_receipt", "assertions", "recorded_domains", "recorded_controls_passed", "raw_kernel_diagnostic_only"}
    if not isinstance(bound, dict) or set(bound) != required:
        raise ValueError("actual native31 FD roles are mandatory and still deferred")
    documents = {}
    for role in ("result", "domain_ledger", "root_receipt"):
        row = bound[role]
        if (set(row) != {"path", "sha256"} or not row["path"].startswith(
                "docs/yb_mcdhf_rci_2026_10_11/screened_f_next_step208/")):
            raise ValueError("native31 actual FD role required, never old29")
        if sha(ROOT / row["path"]) != row["sha256"]:
            raise ValueError("actual31 FD role changed")
        documents[role] = load(ROOT / row["path"])
    receipt = documents["root_receipt"]
    result = documents["result"]
    if (receipt.get("status") != "foreground_closed" or receipt.get("exit_code") not in (0, 1)
            or bound["raw_kernel_diagnostic_only"] is not True
            or type(bound["recorded_controls_passed"]) is not bool
            or result.get("all_declared_controls_passed") != bound["recorded_controls_passed"]
            or not isinstance(bound["recorded_domains"], dict)
            or result.get("derivative_domains") != bound["recorded_domains"]):
        raise ValueError("closed actual31 domain statuses must be retained exactly; no production admission")
    domains = result.get("derivative_domains")
    if (result.get("status") != "native31_single_combined_direction_two_h_FD_completed"
            or set(result.get("points", {})) != {"baseline", "h0_minus", "h0_plus", "h1_minus", "h1_plus"}
            or result.get("completed", {}).get("H") != 5 or result.get("completed", {}).get("CI") != 5
            or set(domains) != {"1e-05", "5e-06"}
            or documents["domain_ledger"] != domains
            or result.get("single_direction_finite_measurement_only") is not True
            or result.get("full31_space_certificate") is not False
            or result.get("no_energy_callback") is not True
            or result.get("ordinary_reported_objective_unchanged") is not True):
        raise ValueError("complete actual31 five-point/two-scale FD domain ledger required; baseline-only failure is not a prerequisite")
    for name, domain in domains.items():
        ordinary = domain.get("ordinary_reported", {})
        centered = domain.get("independent_centered", [])
        if (type(ordinary.get("passed")) is not bool or ordinary.get("h") != float(name)
                or len(centered) != 2
                or [r.get("reference") for r in centered] != ["fixedLambda_ideal", "public_retained"]
                or any(type(r.get("passed")) is not bool for r in centered)
                or domain.get("production_callback") is not False
                or domain.get("no_uniform_error_bound") is not True):
            raise ValueError("ordinary and both centered statuses must be complete and distinct")
    expected = {"source_sha256": fixed["source_map_sha256"],
                "metric_sha256": fixed["new_metric_sha256"],
                "geometry_sha256": fixed["new_geometry_sha256"],
                "state_sha256": fixed["new_state_sha256"],
                "objective_sha256": fixed["objective_identity"]}
    assertions = bound["assertions"]
    if set(assertions) != set(expected):
        raise ValueError("complete actual31 directional assertion mapping required")
    for name, value in expected.items():
        row = assertions[name]
        if (set(row) != {"document", "pointer"} or row["document"] != "result"
                or not isinstance(row["pointer"], list) or not row["pointer"]
                or pointer(documents[row["document"]], row["pointer"]) != value):
            raise ValueError("actual31 directional proof does not bind " + name)
    return documents


def authorize(binding_path):
    contract, roles, source, _ = metadata(arrays=False)
    binding = load(binding_path)
    required = dict(execution_authorized_by_root=True, driver_sha256=sha(__file__),
                    preparation_contract_sha256=sha(HERE / "contract.json"),
                    preparation_sha256=sha(HERE / "preparation.json"),
                    scientific_inputs_sha256=contract["input_roles"],
                    resources=contract["resources"], source_map_sha256=contract["fixed"]["source_map_sha256"],
                    actual_native31_directional_prerequisite=contract["actual_native31_directional_prerequisite"])
    if any(binding.get(k) != v for k, v in required.items()):
        raise ValueError("separate exact root execution seal required")
    prep = load(HERE / "preparation.json")
    if not prep["all_passed"] or prep["driver_sha256"] != sha(__file__):
        raise ValueError("source preparation is stale")
    admit_directional(binding.get("actual_native31_directional_prerequisite"), contract["fixed"])
    metadata(arrays=True)
    return binding, contract, roles, source


def literal_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def execute(binding_path):
    binding, contract, roles, source = authorize(binding_path)
    output = HERE / "actual"
    output.mkdir(exist_ok=False)
    (output / "ONE_ATTEMPT").write_text(sha(binding_path) + "\n")
    start = time.monotonic()
    caps, gates, fixed = contract["resources"], contract["gates"], contract["fixed"]
    calls = {name: {"started": 0, "completed": 0} for name in caps["calls"]}
    request_ledgers = {name: {"requests": 0, "completed": 0, "hits": 0, "misses": 0}
                       for name in ("hS", "factors", "dense_banks", "component_moments")}
    controls, restored, checkpoints = [], [], []
    phase, previous_NumBa_mask, nb = "entry", None, None

    def progress():
        write(output / "progress.json", dict(phase=phase, calls=calls,
              request_ledgers=request_ledgers, controls=len(controls),
              elapsed_uncontrolled_s=time.monotonic() - start))

    def begin(name):
        calls[name]["started"] += 1
        if calls[name]["started"] > caps["calls"][name]:
            progress()
            raise RuntimeError("prospective call cap: " + name)

    def finish(name):
        calls[name]["completed"] += 1

    def check(name, error, limit=0.):
        row = dict(name=name, error=float(error), limit=limit,
                   passed=math.isfinite(float(error)) and float(error) <= limit)
        controls.append(row)
        if not row["passed"]:
            write(output / "first_failure.json", row)
            progress()
            raise AssertionError(name)

    def counted(module, name, counter):
        original = getattr(module, name)
        def forwarding(*args, **kwargs):
            begin(counter)
            value = original(*args, **kwargs)
            finish(counter)
            return value
        restored.append((module, name, original))
        setattr(module, name, forwarding)

    def save(name, **arrays):
        path = output / name
        np.savez_compressed(path, **arrays)
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        checkpoints.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
        progress()

    def watchdog(signum, frame):
        raise TimeoutError("prospective inner wall limit")

    previous_affinity = os.sched_getaffinity(0)
    try:
        if not set(caps["CPU4"]) <= previous_affinity:
            raise RuntimeError("actual permitted affinity does not admit four declared CPUs")
        os.sched_setaffinity(0, {caps["CPU1"]})
        check("actual_CPU1_affinity", os.sched_getaffinity(0) != {caps["CPU1"]})
        for variable in caps["single_thread_env"]:
            check("thread_environment_" + variable, os.environ.get(variable) != "1")
        check("configured_NumBa_max4", os.environ.get("NUMBA_NUM_THREADS") != "4")
        if max(os.getloadavg()) > caps["maximum_host_load"]:
            raise RuntimeError("host load admission failed; no numerical work")
        resource.setrlimit(resource.RLIMIT_AS, (caps["virtual_AS_bytes"], caps["virtual_AS_bytes"]))
        signal.signal(signal.SIGALRM, watchdog)
        signal.alarm(caps["inner_seconds"])
        runtime = ROOT / contract["runtime"]["isolated_runtime"]
        os.environ["CUDA_PATH"] = str(runtime / "nvidia/cu13")
        os.environ["NUMBA_CACHE_DIR"] = str(output / "numba_cache")
        site.addsitedir(str(runtime))
        if any(name == "atomx" or name.startswith("atomx.") for name in sys.modules):
            raise RuntimeError("fresh process with no preloaded Atomx required")
        sys.path.insert(0, str(source))
        import numba as nb
        import numpy as np
        from scipy import linalg

        import atomx.methods.csf.variational as variational
        import atomx.numerics.bspline.convol as convol
        import atomx.numerics.bspline.dirac_assembly as dirac
        import atomx.numerics.bspline.orbital_optimization as optimization
        import atomx.physics.csf.matrix as matrix
        import atomx.physics.csf.tracking as tracking_module
        from atomx.angular.csf_hamiltonian import csf_signature
        from atomx.config.csf import enumerate_jj_csfs, get_shared_orbital_basis
        from atomx.methods import ci as ci_module
        from atomx.physics.csf.levels import ci_level_blocks
        from atomx.physics.csf.radial_density import CIRadialDensity
        from atomx.physics.csf.state_average import CIStateAverage
        from atomx.potentials.nuclear import make_nuclear_potential

        adapter_module = literal_module(roles["adapter"], "private_native31_qualified_adapter")
        identity = load(roles["f043_identity"])
        for name, module in tuple(sys.modules.items()):
            if name == "atomx" or name.startswith("atomx."):
                file = getattr(module, "__file__", None)
                if file and str(file).endswith(".py"):
                    key = "src/" + str(Path(file).relative_to(source))
                    check("loaded_source_" + name, sha(file) != identity["all_atomx_python_sha256"][key])
        for name, version in contract["runtime"]["packages"].items():
            check("runtime_version_" + name, importlib.metadata.version(name) != version)
        check("NumPy_literal_module", str(Path(np.__file__).absolute()) != contract["runtime"]["NumPy_literal_file"])
        check("Numba_literal_module", str(Path(nb.__file__).absolute()) != contract["runtime"]["Numba_literal_file"])
        previous_NumBa_mask = nb.get_num_threads()
        nb.set_num_threads(1)
        write(output / "runtime_admission.json", dict(packages={k: importlib.metadata.version(k)
              for k in contract["runtime"]["packages"]}, NumPy=np.__file__, NumBa=nb.__file__,
              original_affinity=sorted(previous_affinity), affinity_CPU1=sorted(os.sched_getaffinity(0)),
              configured_NumBa_max=4, original_NumBa_mask=previous_NumBa_mask,
              current_NumBa_mask_before_public_setup=nb.get_num_threads(), source_map=fixed["source_map_sha256"]))
        counted(variational, "build_dirac_matrices", "initial_hS")
        counted(matrix, "build_dirac_matrices", "CI_hS_producer")
        counted(matrix, "build_ci_matrix", "H")
        counted(ci_module, "ci_solve", "CI")
        counted(dirac, "solve_dirac", "validation_Dirac")
        counted(np.linalg, "eigvalsh", "SPD_eigvalsh")
        counted(linalg, "eigh", "scipy_eigh")
        counted(np.linalg, "svd", "tracking_SVD")
        counted(tracking_module.CIBasisTracking, "resolve", "tracking_resolve")
        counted(tracking_module, "ci_subspace_overlaps", "tracking_overlap")
        counted(CIRadialDensity, "contract_bilinear", "target_RDM")
        counted(optimization, "optimize_orbitals", "optimizer")
        original_blocks = matrix._dirac_blocks
        def blocks(ns, kappa, Z, workspace):
            begin("hS_getter")
            ledger = request_ledgers["hS"]
            ledger["requests"] += 1
            hit = (("h0", ns, kappa, Z) in matrix._SLATER_MEMO and matrix._MEMO_CONTEXT is workspace)
            value = original_blocks(ns, kappa, Z, workspace)
            ledger["completed"] += 1
            ledger["hits" if hit else "misses"] += 1
            finish("hS_getter")
            return value
        restored.append((matrix, "_dirac_blocks", original_blocks))
        matrix._dirac_blocks = blocks
        original_factors = convol.ConvolCache.get_coulomb_factors
        def factors(cache, rank, component="pppp"):
            begin("factor_getter")
            ledger = request_ledgers["factors"]
            ledger["requests"] += 1
            hit = (rank, component, "semiseparable") in cache._cache
            value = original_factors(cache, rank, component)
            ledger["completed"] += 1
            ledger["hits" if hit else "misses"] += 1
            finish("factor_getter")
            return value
        restored.append((convol.ConvolCache, "get_coulomb_factors", original_factors))
        convol.ConvolCache.get_coulomb_factors = factors
        original_banks = convol.ConvolCache.get_rkb_t
        def banks(cache, rank, component="pppp"):
            begin("dense_bank_getter")
            ledger = request_ledgers["dense_banks"]
            ledger["requests"] += 1
            hit = (rank, component, "T") in cache._cache
            value = original_banks(cache, rank, component)
            ledger["completed"] += 1
            ledger["hits" if hit else "misses"] += 1
            finish("dense_bank_getter")
            return value
        restored.append((convol.ConvolCache, "get_rkb_t", original_banks))
        convol.ConvolCache.get_rkb_t = banks
        original_moments = convol.ConvolCache._component_pair_moments
        def moments(cache, rank, component):
            begin("component_moment_getter")
            ledger = request_ledgers["component_moments"]
            ledger["requests"] += 1
            hit = (rank, component, "pair_moments") in cache._cache
            value = original_moments(cache, rank, component)
            ledger["completed"] += 1
            ledger["hits" if hit else "misses"] += 1
            finish("component_moment_getter")
            return value
        restored.append((convol.ConvolCache, "_component_pair_moments", original_moments))
        convol.ConvolCache._component_pair_moments = moments
        # Request wrappers are observations, not producer-count assumptions.
        counted(convol, "_triang_sym", "same_component_inner_producer")
        counted(convol, "_triang_pqpq", "mixed_component_inner_producer")
        for name in ("_build_rkb_sym", "_build_rkb_pqpq", "_build_rkb_qpqp"):
            counted(convol, name, "dense_bank_producer")
        counted(convol, "_assemble_near_rkb", "near_factor_producer")
        counted(matrix, "_density_sym", "H_density")
        counted(variational, "_density_sym", "ordinary_density")
        counted(convol, "_density_sym", "private_density")

        def archive(role):
            begin("NPZ_load")
            with np.load(roles[role], allow_pickle=False) as arrays:
                value = {k: arrays[k].copy() for k in arrays.files}
            finish("NPZ_load")
            return value
        phase = "complete_public_setup"
        progress()
        point, radial, native_hS, reference_ci_arrays = (archive(role) for role in
                 ("recovered_point", "workspace", "original_hS", "tracking_CI"))
        saved_ci, saved_g = archive("recovered_CI"), archive("saved_gradient_metric")
        check("saved_public_radial_gradient31_contiguous246", saved_g["full_normalized_radial"].shape != (31, 246))
        for field in ("full_normalized_raw_coordinate", "full_normalized_tangent", "full_normalized_preconditioned_tangent"):
            check("saved_complete3689_" + field, saved_g[field].shape != (3689,))
        data = {k[6:]: v.item() if v.ndim == 0 else v.copy()
                for k, v in radial.items() if k.startswith("basis_")}
        keys = tuple(map(tuple, fixed["keys31"]))
        check("saved_complete31_order", tuple(map(tuple, point["keys"])) != keys)
        models = load(roles["models"])
        sectors = {tuple(v) for v in models["selected_sectors"]}
        parent = [c for row in models["models"]["complete5f"]["configs"]
                  for c in enumerate_jj_csfs(row["config"])]
        csfs = [c for c in parent if (c.total_J2, c.parity) in sectors]
        def signature(c):
            s = csf_signature(c)
            return json.dumps(plain((s.subshells, s.coupling, c.total_J2, c.parity)), separators=(",", ":"))
        check("complete3789_parent1595_targets", len(parent) != 3789 or len(csfs) != 1595)
        check("charge69", {c.n_electrons for c in parent} != {69})
        check("signed_CSF_order", [signature(c) for c in csfs] != load(roles["basis"])["signatures"])
        check("public31_orbitals", tuple(sorted(get_shared_orbital_basis(csfs))) != keys)
        reference_models = load(roles["reference_models"])["models"]
        reference_csfs = [c for row in reference_models["baseline"]["configs"]
                         for c in enumerate_jj_csfs(row["config"])]
        targets = load(roles["tracking_binding"])["targets"]
        average = CIStateAverage(tuple((r["twoJ"], r["parity"], r["models"]["DC"]["local_root"], 1.) for r in targets))
        reference_waves = {tuple(map(int, k)): (p.copy(), q.copy()) for k, p, q in
                           zip(radial["orbital_keys"], radial["P"], radial["Q"], strict=True)}
        tracker = tracking_module.CIBasisTracking(average, reference_csfs,
             ci_module.CIResult(reference_ci_arrays["energies_Ha"], reference_ci_arrays["mixing"]),
             reference_waves, {k: native_hS[f"s_{k}"] for k in fixed["kappas"]}, max_determinant_pairs=30000)
        nucleus = make_nuclear_potential(70, "fermi", A=175)
        functional = variational.CSFOrbitalFunctional(csfs, data, 70, nucleus,
             state_average=average, level_tracking=tracker, retain_hamiltonian=True, max_determinants=2000)
        check("public_functional_keys", functional.keys != keys)
        for kappa in fixed["kappas"]:
            check("same_L_completed_h_" + str(kappa), not np.array_equal(functional.one_body[kappa], native_hS[f"h_{kappa}"]))
            check("same_S_" + str(kappa), not np.array_equal(functional.metric[kappa], native_hS[f"s_{kappa}"]))
        layout = tuple((tuple(map(tuple, row["keys"])), point[f"projector_{i}"],
                        tuple(row["shape"]), slice(*row["slice"]))
                       for i, row in enumerate(load(roles["layout"])))
        inverses = tuple(tuple(blocks) for blocks in (point[f"inverse_blocks_{i}"] for i in range(7)))
        state = optimization.OrbitalOptimizationState.from_arrays(functional, layout=layout,
             coordinates=point["x"], inverse_diagonal=point["inverse_diagonal"], inverse_blocks=inverses,
             projector_source="nuclear", objective_identity=fixed["objective_identity"], occupation_floor=.05)
        state.validate(functional)
        check("native3689_metric_identity", state.residual_metric_sha256 != fixed["new_metric_sha256"])
        check("native3689_geometry_identity", state.geometry_sha256 != fixed["new_geometry_sha256"])
        check("native3689_state_identity", state.state_sha256 != fixed["new_state_sha256"])
        waves = {}
        for group, projector, shape, where in state.layout:
            c = projector @ state.coordinates[where].reshape(shape)
            check("actual_SGram_" + str(group[0][1]), np.max(abs(c.T @ functional.metric[group[0][1]] @ c - np.eye(len(group)))), gates["radial_Gram"])
            for j, key in enumerate(group):
                waves[key] = c[:123, j].copy(), c[123:, j].copy()
                stored = np.concatenate((point["P"][keys.index(key)], point["Q"][keys.index(key)]))
                check("actual_stored_point_" + str(key), not np.array_equal(c[:, j], stored))
                outside = c[:, j] - projector @ (projector.T @ functional.metric[key[1]] @ c[:, j])
                check("actual_originalP_leakage_" + str(key), abs(float(outside @ functional.metric[key[1]] @ outside)), gates["P_leak_S_squared"])

        phase = "fresh_full1595_H_CI"
        progress()
        ci = ci_module.ci_solve(csfs, waves, keys, data, 70, 123,
                nuclear_potential=nucleus, convol_cache=functional.cache, solver="dense",
                n_states=None, include_breit=False, max_determinants=2000, retain_hamiltonian=True)
        h = ci.hamiltonian_for(csfs, waves, keys, data, 70, 123, nuclear_potential=nucleus, max_determinants=2000)
        check("saved_actual_owner_header", ci.hamiltonian_snapshot.identity_payload() != load(roles["recovered_owner"]))
        check("saved_same_epoch_H_bitwise", not np.array_equal(h, saved_ci["H"]))
        check("saved_same_epoch_allroots_bitwise", not np.array_equal(ci.state_energies, saved_ci["energies"]))
        check("complete_H_symmetry", not np.array_equal(h, h.T))
        check("all1595_eigenpair_column2norm", np.max(np.linalg.norm(h @ ci.mixing_coefficients - ci.mixing_coefficients * ci.state_energies, axis=0)), gates["CI_residual_Ha"])
        check("all1595_CI_Gram", np.max(abs(ci.mixing_coefficients.T @ ci.mixing_coefficients - np.eye(1595))), gates["CI_Gram"])
        _, blocks_map = ci_level_blocks(csfs, ci, require_full=True)
        for sector, columns in blocks_map.items():
            rows = [i for i, c in enumerate(csfs) if (c.total_J2, c.parity) == sector]
            exterior = [i for i in range(len(csfs)) if i not in set(rows)]
            check("cross_sector_" + str(sector), np.max(abs(h[np.ix_(rows, exterior)])), gates["cross_symmetry_zero_Ha"])
            groups, group = [], []
            for column in columns:
                if group and ci.state_energies[column] - ci.state_energies[group[-1]] > gates["cluster_gap_Ha"]:
                    groups.append(group)
                    group = []
                group.append(int(column))
            groups.append(group)
            for group in groups:
                a = saved_ci["mixing"][np.ix_(rows, group)]
                b = ci.mixing_coefficients[np.ix_(rows, group)]
                check("full_signed_span_" + str(group), np.linalg.norm(a @ a.T - b @ b.T), gates["signed_span"])
        weights, tracked = tracker.resolve(csfs, ci, waves, functional.metric)
        check("original_equal_five_weights", np.count_nonzero(weights) != 5 or not np.array_equal(weights[weights > 0], np.repeat(.2, 5)))
        check("same_saved_tracking_UIDs_weights", plain(dict(weights=weights, tracked=tracked)) != load(roles["recovered_tracking"]))
        matched = {tuple(r["reference_levels"][0]): r for r in tracked}
        check("five_original_UIDs", set(matched) != {tuple(v) for _, v in fixed["target_UIDs"]})
        tables, contributions, target_rows = load(roles["signed_tables"]), {}, []
        for label, uid in fixed["target_UIDs"]:
            row = matched[tuple(uid)]
            check("single_same_sector_target_" + label, len(row["reference_levels"]) != 1 or len(row["selected_levels"]) != 1 or row["selected_levels"][0][:2] != tuple(uid[:2]))
            j, parity, root = row["selected_levels"][0]
            column = int(blocks_map[j, parity][root])
            c = ci.mixing_coefficients[:, column]
            norm = sum((Fraction.from_float(float(v))**2 for v in c), Fraction(0))
            if norm <= 0:
                raise ValueError("nonpositive complete target norm")
            table = functional.radial_density.contract_bilinear(c, c)
            for key, value in table.items():
                contributions.setdefault(key, []).append(Fraction.from_float(float(weights[column])) * Fraction.from_float(float(value)) / norm)
            target_rows.append(dict(label=label, UID=uid, column=column,
                    exact_CI_norm=(str(norm.numerator), str(norm.denominator)), norm_defect=float(norm - 1),
                    unfiltered=[[k, v] for k, v in table.items()]))
        check("all_fresh_unfiltered_target_tables_exact", plain(target_rows) != tables["targets"])
        def represent(value):
            result = float(value)
            if not math.isfinite(result) or (value != 0 and result == 0):
                raise ValueError("unrepresentable nonzero angular coefficient; no omission")
            return result
        exact = {k: sum(v, Fraction(0)) for k, v in contributions.items()}
        full_weights = {k: represent(v) for k, v in exact.items() if v != 0}
        check("fresh_full_normalized_RDM_exact", plain([[k, v] for k, v in full_weights.items()]) != tables["full_normalized"])
        signed_layout = adapter_module.collect_signed_layout(keys, full_weights.items())
        union = load(roles["adapter_union"])
        check("full706_union_order_identity", signed_layout.source_union_sha256 != union["source_union_sha256"] or len(signed_layout.jobs) != 706)
        check("full19589_signed_table_identity", signed_layout.weights_sha256 != union["weights_sha256"] or len(signed_layout.items) != 19589)
        save("complete_public_point.npz", H=h, energies=ci.state_energies, mixing=ci.mixing_coefficients,
             keys=np.array(keys), P=np.array([waves[k][0] for k in keys]), Q=np.array([waves[k][1] for k in keys]),
             x=state.coordinates, inverse_diagonal=state.inverse_diagonal,
             **{f"projector_{i}": r[1] for i, r in enumerate(state.layout)},
             **{f"inverse_blocks_{i}": np.stack(b) for i, b in enumerate(state.inverse_blocks)})
        write(output / "actual_owner.json", ci.hamiltonian_snapshot.identity_payload())
        owner_identity = ci.hamiltonian_snapshot.identity_payload()
        write(output / "actual_signed_tables.json", dict(targets=target_rows, full_normalized=[[k, v] for k, v in full_weights.items()]))

        def coordinate(radial_gradient):
            result = np.zeros(3689, dtype=radial_gradient.dtype)
            lookup = {k: radial_gradient[i] for i, k in enumerate(functional.keys)}
            for group, projector, _, where in state.layout:
                result[where] = (projector.astype(result.dtype).T @ np.column_stack([lookup[k] for k in group])).ravel()
            return result
        def tangent(vector):
            result = vector.copy()
            for _, _, shape, where in state.layout:
                y, g = state.coordinates[where].reshape(shape).astype(vector.dtype), vector[where].reshape(shape)
                multiplier = y.T @ g
                result[where] = (g - .5 * y @ (multiplier + multiplier.T)).ravel()
            return result
        def metric_action(vector):
            result = vector.copy()
            for (_, _, shape, where), inverse in zip(state.layout, state.inverse_blocks, strict=True):
                g = vector[where].reshape(shape)
                result[where] = np.column_stack([b.astype(vector.dtype) @ g[:, j] for j, b in enumerate(inverse)]).ravel()
            return tangent(result)

        def pack(gradient):
            radial_g = np.array([gradient[k] for k in keys])
            check("full_public_private_gradient31_contiguous246", radial_g.shape != (31, 246))
            raw = coordinate(radial_g)
            tg = tangent(raw)
            return dict(radial=radial_g, raw_coordinate=raw, tangent=tg, pg=metric_action(tg))
        def compare(name, values, reference):
            for key in values:
                tolerance = gates["pg"] if key == "pg" else gates["g"]
                check(name + "/complete_" + key, np.max(abs(values[key] - reference[key])), tolerance)
            current_h = ci.hamiltonian_for(csfs, waves, keys, data, 70, 123,
                    nuclear_potential=nucleus, max_determinants=2000)
            check(name + "/same_owned_complete_H", not np.array_equal(current_h, h))
            check(name + "/same_owner_input_matrix_identity", ci.hamiltonian_snapshot.identity_payload() != owner_identity)
            check(name + "/all_CI_residual_Ha", np.max(np.linalg.norm(current_h @ ci.mixing_coefficients - ci.mixing_coefficients * ci.state_energies, axis=0)), gates["CI_residual_Ha"])
            check(name + "/all_CI_Gram", np.max(abs(ci.mixing_coefficients.T @ ci.mixing_coefficients - np.eye(1595))), gates["CI_Gram"])
            save(name + "_complete_gradient.npz", **values)

        original_band = variational._convol_case1_semiseparable_t
        observed, source_cursor, component_cursor = [], [0], [0]
        original_class_potential = variational.CSFOrbitalFunctional._pair_potential
        def potential(instance, rank, a, b, **kwargs):
            begin("ordinary_source_potential")
            job = signed_layout.jobs[source_cursor[0]]
            check("ordinary_source_order_" + str(source_cursor[0]), instance is not functional or (rank, job[1], job[2]) != job or a is not waves[job[1]] or b is not waves[job[2]])
            value = original_class_potential(instance, rank, a, b, **kwargs)
            source_cursor[0] += 1
            finish("ordinary_source_potential")
            return value
        def band(*args, **kwargs):
            begin("ordinary_band")
            value = original_band(*args, **kwargs)
            check("ordinary_component_shape_" + str(component_cursor[0]), value.shape != (123, int(args[7])))
            orders = data.get("ksp", data["ks"]), data.get("ksq", data["ks"])
            left, right = divmod(component_cursor[0] % 4, 2)
            check("ordinary_PP_PQ_QP_QQ_order_" + str(component_cursor[0]),
                  (int(args[6]), int(args[7]), int(args[8])) != (123, orders[left], orders[right]))
            observed.append(value.copy())
            component_cursor[0] += 1
            finish("ordinary_band")
            return value
        restored.extend(((variational.CSFOrbitalFunctional, "_pair_potential", original_class_potential),
                         (variational, "_convol_case1_semiseparable_t", original_band)))
        variational.CSFOrbitalFunctional._pair_potential = potential
        variational._convol_case1_semiseparable_t = band
        phase = "ordinary_full_signed_gradient"
        progress()
        begin("public_full_gradient")
        ordinary = pack(functional.gradient_from_weights(waves, full_weights, record_diagnostics=False))
        finish("public_full_gradient")
        variational.CSFOrbitalFunctional._pair_potential = original_class_potential
        variational._convol_case1_semiseparable_t = original_band
        check("complete_observed706_four_bands", source_cursor[0] != 706 or component_cursor[0] != 2824)
        reference = dict(radial=saved_g["full_normalized_radial"], raw_coordinate=saved_g["full_normalized_raw_coordinate"],
                         tangent=saved_g["full_normalized_tangent"], pg=saved_g["full_normalized_preconditioned_tangent"])
        compare("ordinary_vs_saved438", ordinary, reference)
        saved_bands = np.zeros((706, 4, 123, max(functional.data.get("ksp", data["ks"]), functional.data.get("ksq", data["ks"]))))
        for index, value in enumerate(observed):
            saved_bands[index // 4, index % 4, :, :value.shape[1]] = value
        save("ordinary_all2824_bands.npz", bands=saved_bands)
        binding_fields = load(roles["adapter_binding"])
        native_binding = adapter_module.NativeBinding(**{k: tuple(tuple(v) for v in value) if k == "public_file_sha256" else value
                   for k, value in binding_fields.items() if k != "scope"})
        backend_results, CPU1_engine = {}, None
        for backend_name, backend, workers in (("NumBa_CPU1", "cpu", 1), ("NumBa_CPU4", "cpu", 4), ("CUDA", "cuda", 1)):
            phase = backend_name
            progress()
            os.sched_setaffinity(0, set(caps["CPU4"]) if workers == 4 else {caps["CPU1"]})
            check(backend_name + "/actual_affinity", os.sched_getaffinity(0) != (set(caps["CPU4"]) if workers == 4 else {caps["CPU1"]}))
            mask = nb.get_num_threads()
            cold_GPU = None
            if backend == "cuda":
                from numba import cuda
                begin("CUDA_context")
                context_start = time.monotonic()
                context = cuda.current_context()
                free, total = context.get_memory_info()
                finish("CUDA_context")
                cold_GPU = dict(uncontrolled_context_admission_s=time.monotonic() - context_start,
                    free_bytes=int(free), total_bytes=int(total), device_name=str(context.device.name),
                    compute_capability=context.device.compute_capability)
                check("CUDA_actual_free_payload", free < caps["device_payload_bytes"])
            before = time.monotonic()
            begin("private_owner_factory")
            candidate = adapter_module.NativeBatchGradient(functional, waves, full_weights, ci,
                   binding=native_binding, backend=backend, threads=workers, expected_weights_sha256=signed_layout.weights_sha256)
            finish("private_owner_factory")
            if workers == 4:
                check("CPU4_reuse_same_NumBa_CPU_factory", candidate.executor.nb is not CPU1_engine.nb or candidate.executor.backend != "cpu")
                check("CPU4_same_source_factory_plan", candidate.executor.plan.identity != CPU1_engine.plan.identity or candidate.executor.factor_value_sha256 != CPU1_engine.factor_value_sha256)
                candidate.executor.kernels = CPU1_engine.kernels
            elif backend == "cpu":
                CPU1_engine = candidate.executor
            construction_s = time.monotonic() - before
            chunks = candidate.executor.chunks
            checked_components, bitwise_components, maximum_scaled = [0], [0], [0.]
            def observe_packets(snapshot, chunks=chunks, backend_name=backend_name,
                                checked_components=checked_components,
                                bitwise_components=bitwise_components, maximum_scaled=maximum_scaled):
                cursor = 0
                for jobs, packets in chunks(snapshot):
                    expected = signed_layout.jobs[cursor:cursor + len(jobs)]
                    check(backend_name + "/packet_job_order_" + str(cursor), jobs != expected)
                    for i in range(len(jobs)):
                        for comp in range(4):
                            target = saved_bands[cursor + i, comp]
                            value = packets[i, comp]
                            error = float(np.max(abs(value - target)) / max(1., float(np.max(abs(target)))))
                            check(backend_name + "/band_" + str(checked_components[0]), error, gates["band_scaled"])
                            maximum_scaled[0] = max(maximum_scaled[0], error)
                            bitwise_components[0] += int(np.array_equal(value, target))
                            checked_components[0] += 1
                    cursor += len(jobs)
                    yield jobs, packets
                check(backend_name + "/complete706_jobs", cursor != 706)
            candidate.executor.chunks = observe_packets
            begin("private_full_gradient")
            started = time.monotonic()
            try:
                nb.set_num_threads(workers)
                actual_private_mask = nb.get_num_threads()
                check(backend_name + "/actual_private_NumBa_mask", actual_private_mask != workers)
                values = pack(candidate.gradient_from_weights(waves, full_weights, ci,
                       expected_weights_sha256=signed_layout.weights_sha256, record_diagnostics=False))
            finally:
                nb.set_num_threads(mask)
            elapsed = time.monotonic() - started
            finish("private_full_gradient")
            check(backend_name + "/thread_mask_restored", nb.get_num_threads() != mask)
            compare(backend_name + "_vs_ordinary", values, ordinary)
            compare(backend_name + "_vs_saved438", values, reference)
            check(backend_name + "/every2824_component", checked_components[0] != 2824)
            check(backend_name + "/all39064_signed_incidences", candidate.counters["signed_term_incidence"] != 39064)
            check(backend_name + "/density992_requests", candidate.counters["density_band_requests"] != 992)
            check(backend_name + "/four2824_expansions", candidate.counters["band_expansions"] != 2824)
            check(backend_name + "/sixscan_sixrow", candidate.executor.counters["scan_kernel"] != 6 or candidate.executor.counters["row_kernel"] != 6)
            ptx_audit = {}
            if backend == "cuda":
                for i, kernel in enumerate(candidate.executor.kernels):
                    ptx = kernel.inspect_asm()
                    text = "\n".join(ptx.values()) if isinstance(ptx, dict) else str(ptx)
                    (output / f"CUDA_kernel_{i}.ptx").write_text(text)
                    check("CUDA_no_atomic_or_FMA_" + str(i), "fma." in text or "atom." in text or "red." in text)
                    ptx_audit[str(i)] = dict(sha256=sha(output / f"CUDA_kernel_{i}.ptx"), signatures=len(kernel.signatures))
                check("CUDA_synchronized_all_transfers", candidate.executor.counters["stream_sync"] != 7)
            backend_results[backend_name] = dict(uncontrolled_construction_s=construction_s,
                uncontrolled_first_gradient_including_JIT_and_archive_observers_s=elapsed,
                bitwise_bands=bitwise_components[0], all_bands=checked_components[0], max_scaled_band_error=maximum_scaled[0],
                adapter_calls=candidate.counters, engine_calls=candidate.executor.counters,
                before_and_restored_NumBa_mask=mask, actual_NumBa_mask_during_gradient=actual_private_mask,
                factor_value_sha256=candidate.executor.factor_value_sha256,
                resident_identity=candidate.executor.resident_identity, preflight=candidate.executor.preflight,
                PTX=ptx_audit, cold_GPU=cold_GPU, timing_speed_certificate=False)
            write(output / "backend_completed.json", backend_results)
        signal.alarm(0)
        metadata(arrays=True)
        phase = "completed"
        for name, count in (("H", 1), ("CI", 1), ("public_full_gradient", 1), ("private_full_gradient", 3),
                            ("private_owner_factory", 3), ("NPZ_load", 6), ("CUDA_context", 1), ("optimizer", 0)):
            check("actual_complete_call_count_" + name, calls[name]["completed"] != count)
        write(output / "result.json", dict(status="native31_private_batch_correctness_completed",
              all_declared_controls_passed=True, checks=controls, calls=calls, request_ledgers=request_ledgers,
              backends=backend_results, archives=checkpoints, root_execution_binding_sha256=sha(binding_path),
              contract_sha256=sha(HERE / "contract.json"), source_map_sha256=fixed["source_map_sha256"],
              actual_native31_directional_prerequisite=binding["actual_native31_directional_prerequisite"],
              original_stationarity_passed=False, expanded_MCDHF_optimized=False,
              full_H_CI_on_GPU=False, controlled_speedup_qualified=False, elapsed_uncontrolled_s=time.monotonic() - start))
    except BaseException as exc:
        write(output / "result.json", dict(status="failed_partial", all_declared_controls_passed=False,
              failure_type=type(exc).__name__, failure=str(exc), traceback=traceback.format_exc(), phase=phase,
              checks=controls, calls=calls, request_ledgers=request_ledgers, archives=checkpoints,
              root_execution_binding_sha256=sha(binding_path), elapsed_uncontrolled_s=time.monotonic() - start,
              controlled_speedup_qualified=False, original_stationarity_passed=False))
        raise
    finally:
        signal.alarm(0)
        for module, name, function in reversed(restored):
            setattr(module, name, function)
        if nb is not None and previous_NumBa_mask is not None:
            nb.set_num_threads(previous_NumBa_mask)
        os.sched_setaffinity(0, previous_affinity)
        write(output / "thread_affinity_restore.json", dict(original_affinity=sorted(previous_affinity),
              restored_affinity=sorted(os.sched_getaffinity(0)), original_NumBa_mask=previous_NumBa_mask,
              restored_NumBa_mask=nb.get_num_threads() if nb is not None and previous_NumBa_mask is not None else None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", type=Path)
    args = parser.parse_args()
    if args.execute is None:
        contract, _, _, deferred = metadata(arrays=False)
        print(json.dumps(dict(mode="SOURCE_JSON_ONLY", deferred_array_roles=deferred,
              FD_prerequisite=contract["actual_native31_directional_prerequisite"], numerical_calls=0), indent=2))
    else:
        execute(args.execute)


if __name__ == "__main__":
    main()
