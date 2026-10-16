"""PRIVATE resident native31 CPU1/CPU4/CUDA timing; default is metadata.

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
import statistics
import subprocess
import sys
import time
import traceback
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
    admit_window(binding, contract)
    metadata(arrays=True)
    return binding, contract, roles, source


def literal_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def admit_window(binding, contract):
    """Root declares isolated window after both possible competing jobs close."""
    admission = binding.get("controlled_window_prerequisites", {})
    if (set(admission) != {"receipts", "no_other_numerical_gpu_browser_or_hourly_jobs"}
            or admission["no_other_numerical_gpu_browser_or_hourly_jobs"] is not True
            or set(admission["receipts"]) != set(contract["resources"]["window_receipts_deferred"])):
        raise ValueError("exact root-controlled nonoverlap window required")
    for name, row in admission["receipts"].items():
        if set(row) == {"not_scheduled", "reason"}:
            if row["not_scheduled"] is not True or not isinstance(row["reason"], str) or not row["reason"]:
                raise ValueError("explicit root-only unscheduled disposition required")
        elif set(row) == {"path", "sha256"}:
            if not row["path"].startswith("docs/yb_mcdhf_rci_2026_10_11/") or sha(ROOT / row["path"]) != row["sha256"]:
                raise ValueError("closed nonoverlap receipt drift: " + name)
            receipt = load(ROOT / row["path"])
            if receipt.get("status") != "foreground_closed" or type(receipt.get("exit_code")) is not int:
                raise ValueError("competing process has no actual closed receipt: " + name)
        else:
            raise ValueError("closed receipt or explicit unscheduled root declaration required")


def cpu_ticks():
    rows = {}
    for line in Path("/proc/stat").read_text().splitlines():
        label, *values = line.split()
        if label.startswith("cpu") and label[3:].isdigit():
            rows[int(label[3:])] = tuple(map(int, values[:8]))
    return rows


def task_affinities():
    rows = {}
    for path in Path('/proc/self/task').iterdir():
        try:
            rows[int(path.name)] = os.sched_getaffinity(int(path.name))
        except ProcessLookupError:
            continue
    return rows


def idle_guard(caps, *, cuda_owner_pid=None):
    """Runtime-only idle sampling; failure invalidates timing, not physics gates."""
    before = cpu_ticks()
    time.sleep(caps["host_guards"]["idle_seconds"])
    after = cpu_ticks()
    cpu_rows = {}
    for cpu in caps["CPU4"]:
        changes = [b - a for a, b in zip(before[cpu], after[cpu], strict=True)]
        total = sum(changes)
        if total <= 0:
            raise RuntimeError("idle CPU sample has no elapsed ticks")
        busy = (total - changes[3] - changes[4]) / total
        system = sum(changes[i] for i in (4, 5, 6, 7)) / total
        cpu_rows[str(cpu)] = dict(busy_fraction=busy, system_fraction=system,
            passed=busy <= caps["host_guards"]["maximum_idle_busy_fraction_each_used_CPU"]
            and system <= caps["host_guards"]["maximum_system_iowait_irq_softirq_steal_fraction_each_CPU"])
    query = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=5)
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=5)
    gpu_rows = [row.strip().split(", ") for row in query.stdout.splitlines() if row.strip()]
    process_rows = [row.strip().split(", ") for row in processes.stdout.splitlines() if row.strip()]
    if len(gpu_rows) != 1 or any(len(row) != 4 for row in gpu_rows):
        raise RuntimeError("one declared actual GPU required; ambiguous query rejected")
    no_competitors = all(len(row) == 3 and int(row[0]) == cuda_owner_pid for row in process_rows)
    utilization = int(gpu_rows[0][2])
    load_per_cpu = os.getloadavg()[0] / len(after)
    passed = (all(row["passed"] for row in cpu_rows.values()) and no_competitors
              and utilization == 0 and load_per_cpu <= caps["maximum_load1_per_onlineCPU"])
    return dict(passed=passed, CPU=cpu_rows, GPU=query.stdout, GPU_processes=processes.stdout,
                GPU_idle_percent=utilization, owner_PID=cuda_owner_pid,
                load1_per_onlineCPU=load_per_cpu, onlineCPUs=len(after))


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
    phase_times, trials, owner_results, host_guards = {}, [], {}, []
    original_task_affinities = task_affinities()

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
        archive_start = time.monotonic()
        path = output / name
        np.savez_compressed(path, **arrays)
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        begin("NPZ_load")
        with np.load(path, allow_pickle=False) as replay:
            for field, value in arrays.items():
                check("archive_readback_" + name + "/" + field, not np.array_equal(replay[field], value))
            finish("NPZ_load")
        checkpoints.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
        phase_times.setdefault("archive_s", []).append(dict(name=name, elapsed_s=time.monotonic() - archive_start))
        progress()

    def watchdog(signum, frame):
        raise TimeoutError("prospective inner wall limit")

    def pin_all_tasks(cpuset):
        # Numba pools launched while CPU1 is current can inherit {30}.
        # Changing just the main thread later does not admit four real CPUs.
        actual = task_affinities()
        for tid, mask in actual.items():
            original_task_affinities.setdefault(tid, mask)
            try:
                os.sched_setaffinity(tid, cpuset)
            except ProcessLookupError:
                continue
        actual = task_affinities()
        for tid, mask in actual.items():
            check('actual_thread_affinity_' + str(tid), mask != cpuset)
        return {str(tid): sorted(mask) for tid, mask in actual.items()}

    previous_affinity = os.sched_getaffinity(0)
    try:
        if not set(caps["CPU4"]) <= previous_affinity:
            raise RuntimeError("actual permitted affinity does not admit four declared CPUs")
        pin_all_tasks({caps["CPU1"]})
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

        helper = literal_module(HERE / 'native31_helpers.py', 'native31_resident_literal_helper')
        runtime_context = dict(np=np, check=check, begin=begin, finish=finish, load=load, plain=plain,
            roles=roles, fixed=fixed, gates=gates, variational=variational, tracking_module=tracking_module,
            optimization=optimization, ci_module=ci_module, csf_signature=csf_signature,
            enumerate_jj_csfs=enumerate_jj_csfs, get_shared_orbital_basis=get_shared_orbital_basis,
            CIStateAverage=CIStateAverage, make_nuclear_potential=make_nuclear_potential,
            output=output, write=write, save=save, ci_level_blocks=ci_level_blocks, adapter_module=adapter_module)
        host_guards.append(dict(phase='before_public_setup', **idle_guard(caps)))
        phase = 'literal_public_setup'
        progress()
        tick = time.monotonic()
        point = helper.prepare_native31_point(runtime_context)
        phase_times['public_setup_s'] = time.monotonic() - tick
        phase = 'public_HCI_tracking_RDM_validation'
        progress()
        tick = time.monotonic()
        reference = helper.fresh_native31_reference(runtime_context, point)
        phase_times['reference_with_validation_archive_s'] = time.monotonic() - tick
        for name in ('public_HCI_s', 'tracking_s', 'normalized_RDM_s'):
            phase_times[name] = reference[name]
        ci, functional, waves, data = (reference['ci'], point['functional'], point['waves'], point['data'])
        signed_layout, full_weights = reference['signed_layout'], reference['full_weights']
        saved_g = point['saved_g']
        saved_reference = dict(radial=saved_g['full_normalized_radial'],
            raw_coordinate=saved_g['full_normalized_raw_coordinate'], tangent=saved_g['full_normalized_tangent'],
            pg=saved_g['full_normalized_preconditioned_tangent'])
        original_band = variational._convol_case1_semiseparable_t
        original_potential = variational.CSFOrbitalFunctional._pair_potential
        public_packets, public_sources, public_band_times = [], [0], []

        def public_potential(instance, rank, a, b, **kwargs):
            begin('ordinary_source_potential')
            job = signed_layout.jobs[public_sources[0]]
            check('ordinary_source_order_' + str(public_sources[0]), instance is not functional
                  or (rank, job[1], job[2]) != job or a is not waves[job[1]] or b is not waves[job[2]])
            value = original_potential(instance, rank, a, b, **kwargs)
            public_sources[0] += 1
            finish('ordinary_source_potential')
            return value

        def public_band(*args, **kwargs):
            begin('ordinary_band')
            tick = time.monotonic()
            value = original_band(*args, **kwargs)
            public_band_times.append(time.monotonic() - tick)
            component = len(public_packets)
            left, right = divmod(component % 4, 2)
            orders = data.get('ksp', data['ks']), data.get('ksq', data['ks'])
            check('ordinary_component_shape_' + str(component), value.shape != (123, int(args[7])))
            check('ordinary_component_order_' + str(component), (int(args[6]), int(args[7]), int(args[8])) != (123, orders[left], orders[right]))
            public_packets.append(value)
            finish('ordinary_band')
            return value

        restored.extend(((variational.CSFOrbitalFunctional, '_pair_potential', original_potential),
                         (variational, '_convol_case1_semiseparable_t', original_band)))
        variational.CSFOrbitalFunctional._pair_potential = public_potential
        variational._convol_case1_semiseparable_t = public_band
        phase = 'public_fullsigned_gradient_anchor'
        progress()
        begin('public_full_gradient')
        tick = time.monotonic()
        public_raw = functional.gradient_from_weights(waves, full_weights, record_diagnostics=False)
        phase_times['public_raw_gradient_with_forward_observer_s'] = time.monotonic() - tick
        tick = time.monotonic()
        ordinary = helper.pack_native31_gradient(runtime_context, point, public_raw)
        phase_times['public_projection_metric_s'] = time.monotonic() - tick
        finish('public_full_gradient')
        variational.CSFOrbitalFunctional._pair_potential = original_potential
        variational._convol_case1_semiseparable_t = original_band
        phase_times['public_band_call_sum_with_first_JIT_s'] = sum(public_band_times)
        check('ordinary_complete706_2824', public_sources[0] != 706 or len(public_packets) != 2824)
        saved_bands = np.zeros((706, 4, 123, max(data.get('ksp', data['ks']), data.get('ksq', data['ks']))))
        for index, value in enumerate(public_packets):
            saved_bands[index // 4, index % 4, :, :value.shape[1]] = value
        public_packets.clear()
        del public_raw
        for field in ordinary:
            check('ordinary_vs_saved438/' + field, np.max(abs(ordinary[field] - saved_reference[field])), gates['pg'] if field == 'pg' else gates['g'])
        save('ordinary_complete_gradient.npz', **ordinary, bands=saved_bands)
        binding_fields = load(roles['adapter_binding'])
        native_binding = adapter_module.NativeBinding(**{k: tuple(tuple(v) for v in value) if k == 'public_file_sha256' else value
                   for k, value in binding_fields.items() if k != 'scope'})
        owners, cold_values, cold_packets = {}, {}, {}
        CPU1_engine = None
        for name, backend, threads in (('NumBa_CPU1', 'cpu', 1), ('NumBa_CPU4', 'cpu', 4), ('CUDA', 'cuda', 1)):
            phase = 'resident_owner_' + name
            progress()
            pin_all_tasks(set(caps['CPU4']) if threads == 4 else {caps['CPU1']})
            cold_context_s = None
            if backend == 'cuda':
                from numba import cuda
                begin('CUDA_context')
                tick = time.monotonic()
                context = cuda.current_context()
                free, total = context.get_memory_info()
                cold_context_s = time.monotonic() - tick
                finish('CUDA_context')
                check('CUDA_actual_free_payload', free < caps['device_payload_bytes'])
                write(output / 'CUDA_context_admission.json', dict(free_bytes=int(free), total_bytes=int(total),
                    device_name=str(context.device.name), compute_capability=context.device.compute_capability,
                    context_s=cold_context_s))
            begin('private_owner_factory')
            tick = time.monotonic()
            owner = adapter_module.NativeBatchGradient(functional, waves, full_weights, ci,
                binding=native_binding, backend=backend, threads=threads, expected_weights_sha256=signed_layout.weights_sha256)
            construction_s = time.monotonic() - tick
            finish('private_owner_factory')
            if threads == 4:
                check('CPU4_same_NumBa_factory_plan', owner.executor.nb is not CPU1_engine.nb
                    or owner.executor.plan.identity != CPU1_engine.plan.identity
                    or owner.executor.factor_value_sha256 != CPU1_engine.factor_value_sha256)
                owner.executor.kernels = CPU1_engine.kernels
            elif backend == 'cpu':
                CPU1_engine = owner.executor
            owners[name] = owner
            check(name + '/device_payload_upper', owner.executor.preflight['owned_device_bytes_upper'] > caps['device_payload_bytes'])
            owner_results[name] = dict(construction_s=construction_s, context_s=cold_context_s,
                CPU4_JIT_reused=threads == 4, factor_value_sha256=owner.executor.factor_value_sha256,
                resident_identity=owner.executor.resident_identity, preflight=owner.executor.preflight)
        pin_all_tasks({caps['CPU1']})
        check('three_independent_owned_executors', len({id(v.executor) for v in owners.values()}) != 3)
        check('three_same_factor_values', len({v.executor.factor_value_sha256 for v in owners.values()}) != 1)
        primitive_total = dict(density_band_requests=0, band_expansions=0, signed_term_incidence=0,
            pair_actions=0, one_body_actions=0, scan_kernel=0, row_kernel=0)

        def one_call(name, category, ordinal, profile=False):
            nonlocal phase
            phase = category + '/' + str(ordinal) + '/' + name
            progress()
            owner = owners[name]
            threads = 4 if name == 'NumBa_CPU4' else 1
            current_task_masks = pin_all_tasks(set(caps['CPU4']) if threads == 4 else {caps['CPU1']})
            check(phase + '/actual_affinity', os.sched_getaffinity(0) != (set(caps['CPU4']) if threads == 4 else {caps['CPU1']}))
            mask = nb.get_num_threads()
            check(phase + '/entry_mask_one', mask != 1)
            helper.validate_reference_bytes(runtime_context, point, reference, phase + '/before')
            packets, packet_times = [], []
            original_chunks = owner.executor.chunks
            before_adapter, before_engine = owner.counters.copy(), owner.executor.counters.copy()
            def collect(snapshot):
                iterator = iter(original_chunks(snapshot))
                while True:
                    tick = time.monotonic() if profile else None
                    try:
                        jobs, packet = next(iterator)
                    except StopIteration:
                        break
                    if profile:
                        packet_times.append(time.monotonic() - tick)
                    packets.append((jobs, packet))
                    yield jobs, packet
            owner.executor.chunks = collect
            begin('private_full_gradient')
            before_getters = calls['factor_getter']['completed'], calls['hS_getter']['completed'], calls['H']['completed'], calls['CI']['completed']
            try:
                nb.set_num_threads(threads)
                check(phase + '/actual_NumBa_mask', nb.get_num_threads() != threads)
                tick = time.monotonic()
                raw = owner.gradient_from_weights(waves, full_weights, ci,
                    expected_weights_sha256=signed_layout.weights_sha256, record_diagnostics=False)
                raw_s = time.monotonic() - tick
                tick = time.monotonic()
                values = helper.pack_native31_gradient(runtime_context, point, raw)
                metric_s = time.monotonic() - tick
            finally:
                owner.executor.chunks = original_chunks
                nb.set_num_threads(mask)
            finish('private_full_gradient')
            complete_s = raw_s + metric_s
            validation_start = time.monotonic()
            check(phase + '/mask_restored_one', nb.get_num_threads() != 1)
            after_getters = calls['factor_getter']['completed'], calls['hS_getter']['completed'], calls['H']['completed'], calls['CI']['completed']
            check(phase + '/no_new_factor_hS_H_CI', after_getters != before_getters)
            band_report = helper.validate_native31_gradient(runtime_context, point, reference, phase, values, ordinary, packets, saved_bands)
            for field in values:
                check(phase + '/same_saved438_' + field, np.max(abs(values[field] - saved_reference[field])), gates['pg'] if field == 'pg' else gates['g'])
                if name in cold_values:
                    check(phase + '/same_backend_cold_' + field, np.max(abs(values[field] - cold_values[name][field])), gates['pg'] if field == 'pg' else gates['g'])
            if name in cold_packets:
                check(phase + '/same_backend_cold_allbands_bitwise', any(not np.array_equal(a[1], b[1])
                    for a, b in zip(packets, cold_packets[name], strict=True)))
            else:
                cold_values[name], cold_packets[name] = values, packets
            adapter_delta = {k: owner.counters[k] - before_adapter[k] for k in before_adapter}
            engine_delta = {k: owner.executor.counters[k] - before_engine[k] for k in before_engine}
            expected_adapter = dict(gradient=1, density_band_requests=992, band_expansions=2824,
                signed_term_incidence=39064, pair_actions=7727, one_body_actions=31, H_owner_validation=1)
            for field, expected in expected_adapter.items():
                check(phase + '/primitive_' + field, adapter_delta[field] != expected)
                if field in primitive_total:
                    primitive_total[field] += adapter_delta[field]
            for field in ('trials', 'scan_kernel', 'row_kernel'):
                expected = 1 if field == 'trials' else 6
                check(phase + '/primitive_' + field, engine_delta[field] != expected)
                if field in primitive_total:
                    primitive_total[field] += engine_delta[field]
            if name == 'CUDA':
                for field, expected in dict(density_H2D_bytes=8785152, job_H2D_bytes=90368,
                     band_D2H_bytes=25009344, stream_sync=6, factor_H2D_bytes=0).items():
                    check(phase + '/CUDA_' + field, engine_delta[field] != expected)
            else:
                check(phase + '/CPU_no_device_transfers', any(engine_delta[k] for k in ('density_H2D_bytes', 'job_H2D_bytes', 'band_D2H_bytes', 'stream_sync', 'factor_H2D_bytes')))
            validation_s = time.monotonic() - validation_start
            tick = time.monotonic()
            bands_array = np.concatenate([packet for _, packet in packets], axis=0)
            save(category + '_' + str(ordinal) + '_' + name + '.npz', **values, bands=bands_array)
            archive_s = time.monotonic() - tick
            row = dict(backend=name, category=category, ordinal=ordinal, full_gradient_metric_s=complete_s,
                raw_gradient_s=raw_s, original_metric_s=metric_s, validation_s=validation_s, archive_s=archive_s,
                NumBa_mask=threads, affinity=sorted(os.sched_getaffinity(0)), all_task_affinities=current_task_masks, bands=band_report,
                adapter_delta=adapter_delta, engine_delta=engine_delta,
                warmup_packet_transferinclusive_s=packet_times, profile_enabled=profile,
                first_call_JIT_reused=name == 'NumBa_CPU4' and category == 'first')
            if profile:
                row['profile'] = dict(packet_band_transferinclusive_s=sum(packet_times),
                    raw_remainder_density_identity_CPUactions_signedadd_s=raw_s - sum(packet_times),
                    fraction_packet_within_raw=sum(packet_times) / raw_s,
                    device_kernel_transfer_duration_separately_available=False,
                    timings_uncontrolled_and_not_scored=True)
            trials.append(row)
            pin_all_tasks({caps['CPU1']})
            progress()
            write(output / 'completed_trials.json', trials)
            return row

        for name in ('NumBa_CPU1', 'NumBa_CPU4', 'CUDA'):
            one_call(name, 'first', 0)
        for index, kernel in enumerate(owners['CUDA'].executor.kernels):
            ptx = kernel.inspect_asm()
            text = '\n'.join(ptx.values()) if isinstance(ptx, dict) else str(ptx)
            (output / ('CUDA_kernel_' + str(index) + '.ptx')).write_text(text)
            check('CUDA_no_atomic_FMA_' + str(index), 'fma.' in text or 'atom.' in text or 'red.' in text)
        for name in ('NumBa_CPU1', 'NumBa_CPU4', 'CUDA'):
            one_call(name, 'explicit_warmup', 0, profile=True)
        for index, order in enumerate(contract['schedule']['warm_triplets']):
            host_guards.append(dict(phase='triplet_' + str(index) + '_before', **idle_guard(caps, cuda_owner_pid=os.getpid())))
            for name in order:
                one_call(name, 'warm', index)
            host_guards.append(dict(phase='triplet_' + str(index) + '_after', **idle_guard(caps, cuda_owner_pid=os.getpid())))
            write(output / 'host_guards.json', host_guards)
        pin_all_tasks({caps['CPU1']})
        for name, owner in owners.items():
            owner_results[name].update(adapter_calls=owner.counters, engine_calls=owner.executor.counters)
            check(name + '/complete_seven_private_gradients', owner.counters['gradient'] != 7 or owner.executor.counters['trials'] != 7)
        cuda_calls = owners['CUDA'].executor.counters
        for field, expected in dict(factor_H2D_bytes=38930976, density_H2D_bytes=61496064,
                job_H2D_bytes=632576, band_D2H_bytes=175065408, stream_sync=43,
                scan_kernel=42, row_kernel=42).items():
            check('complete_CUDA_' + field, cuda_calls[field] != expected)
        for name, expected in (('H', 1), ('CI', 1), ('public_full_gradient', 1), ('private_full_gradient', 21),
                               ('private_owner_factory', 3), ('NPZ_load', 29), ('CUDA_context', 1), ('optimizer', 0)):
            check('complete_actual_call_' + name, calls[name]['completed'] != expected)
        for field, expected in dict(density_band_requests=20832, band_expansions=59304, signed_term_incidence=820344,
            pair_actions=162267, one_body_actions=651, scan_kernel=126, row_kernel=126).items():
            check('all_private_primitives_' + field, primitive_total[field] != expected)
        statistics_rows = {}
        all_host_guards = all(row['passed'] for row in host_guards)
        for target in ('NumBa_CPU4', 'CUDA'):
            ratios, efficiency = [], []
            for index in range(5):
                selected = {r['backend']: r['full_gradient_metric_s'] for r in trials if r['category'] == 'warm' and r['ordinal'] == index}
                ratios.append(selected['NumBa_CPU1'] / selected[target])
                if target == 'NumBa_CPU4':
                    efficiency.append(selected['NumBa_CPU1'] / (4 * selected[target]))
            median = statistics.median(ratios)
            statistics_rows[target] = dict(within_triplet_ratios=ratios, median_ratio=median,
                logical_worker_efficiency=efficiency, median_logical_worker_efficiency=statistics.median(efficiency) if efficiency else None,
                candidate_scoped_speed_after_clean_closed_resource_receipt=all_host_guards and median >= 1.05,
                private_complete_gradient_scoped_speed_qualified=False,
                closed_resource_receipt_still_required=True,
                all_host_guards_passed=all_host_guards, predeclared_median_threshold=1.05)
        metadata(arrays=True)
        phase = 'completed'
        write(output / 'result.json', dict(status='native31_resident_gradient_timing_completed',
            all_declared_science_controls_passed=True, checks=controls, calls=calls, request_ledgers=request_ledgers,
            phase_times=phase_times, owners=owner_results, trials=trials, host_guards=host_guards,
            private_primitives=primitive_total, statistics=statistics_rows, archives=checkpoints,
            root_execution_binding_sha256=sha(binding_path), contract_sha256=sha(HERE / 'contract.json'),
            source_map_sha256=fixed['source_map_sha256'], actual_native31_directional_prerequisite=binding['actual_native31_directional_prerequisite'],
            all_host_guards_passed=all_host_guards, original_stationarity_passed=False, expanded_MCDHF_optimized=False,
            full_H_CI_on_GPU=False, full_optimizer_speedup_qualified=False, public_default_adoption=False,
            controlled_speedup_pending_closed_guardian_resource_receipt=True, elapsed_s=time.monotonic() - start))
    except BaseException as exc:
        write(output / 'result.json', dict(status='failed_partial', all_declared_science_controls_passed=False,
            failure_type=type(exc).__name__, failure=str(exc), traceback=traceback.format_exc(), phase=phase,
            checks=controls, calls=calls, request_ledgers=request_ledgers, phase_times=phase_times,
            trials=trials, owners=owner_results, host_guards=host_guards, archives=checkpoints,
            root_execution_binding_sha256=sha(binding_path), elapsed_s=time.monotonic() - start,
            controlled_speedup_qualified=False, original_stationarity_passed=False))
        raise
    finally:
        signal.alarm(0)
        for module, name, function in reversed(restored):
            setattr(module, name, function)
        if nb is not None and previous_NumBa_mask is not None:
            nb.set_num_threads(previous_NumBa_mask)
        restored_task_masks = {}
        for tid, mask in original_task_affinities.items():
            try:
                os.sched_setaffinity(tid, mask)
                restored_task_masks[str(tid)] = sorted(os.sched_getaffinity(tid))
            except ProcessLookupError:
                restored_task_masks[str(tid)] = 'thread_exited'
        os.sched_setaffinity(0, previous_affinity)
        write(output / 'thread_affinity_restore.json', dict(original_affinity=sorted(previous_affinity),
            restored_affinity=sorted(os.sched_getaffinity(0)), original_NumBa_mask=previous_NumBa_mask,
            original_task_affinities={str(tid): sorted(mask) for tid, mask in original_task_affinities.items()},
            restored_task_affinities=restored_task_masks,
            restored_NumBa_mask=nb.get_num_threads() if nb is not None and previous_NumBa_mask is not None else None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', type=Path)
    args = parser.parse_args()
    if args.execute is None:
        contract, _, _, deferred = metadata(arrays=False)
        print(json.dumps(dict(mode='SOURCE_JSON_ONLY', deferred_array_roles=deferred,
            fixed_source=contract['fixed']['source_map_sha256'], numerical_calls=0,
            controlled_window_and_actual_closed_receipts_required=True), indent=2))
    else:
        execute(args.execute)


if __name__ == '__main__':
    main()
