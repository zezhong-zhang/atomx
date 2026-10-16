"""Closed-only SOURCE/JSON scalar audit; default never reads actual outputs."""

import argparse
import hashlib
import json
import math
import os
import resource
import signal
import statistics
import time
from pathlib import Path

HERE = Path(__file__).absolute().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").is_file())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def execute(binding_path):
    binding = load(binding_path)
    contract = load(HERE / "contract.json")
    if (binding.get("closed_JSON_scalar_audit_authorized_by_root") is not True
            or binding.get("actual_foreground_closed_declared_by_root") is not True
            or binding.get("script_sha256") != sha(__file__)
            or binding.get("contract_sha256") != sha(HERE / "contract.json")
            or set(binding.get("closed_actual_roles", {})) != {"closed_receipt", "result"}):
        raise ValueError("exact post-closure root binding and supplied actual SHA roles required")
    for role, row in binding["closed_actual_roles"].items():
        if set(row) != {"path", "sha256"} or row["path"] != contract["actual_roles_deferred"][role] or len(row["sha256"]) != 64:
            raise ValueError("exact closed receipt/result roles required; no mutable polling")
    for row in contract["prepared_inputs"].values():
        if sha(ROOT / row["path"]) != row["sha256"]:
            raise ValueError("frozen prepared source changed")
    # Stdlib-only audit has no CUDA virtual-mapping allowance.
    memory_cap = contract["resources"]["RSS_bytes"]
    resource.setrlimit(resource.RLIMIT_AS, (memory_cap, memory_cap))
    receipt_role = binding["closed_actual_roles"]["closed_receipt"]
    if sha(ROOT / receipt_role["path"]) != receipt_role["sha256"]:
        raise ValueError("supplied closed receipt changed")
    receipt = load(ROOT / receipt_role["path"])
    if (receipt.get("status") != "foreground_closed" or type(receipt.get("exit_code")) is not int
            or receipt.get("root_seal_sha256") != contract["fixed"]["timing_root_execution_binding_sha256"]):
        raise ValueError("foreground still open or actual receipt belongs to another attempt")
    result_role = binding["closed_actual_roles"]["result"]
    if sha(ROOT / result_role["path"]) != result_role["sha256"]:
        raise ValueError("supplied closed result changed")
    result = load(ROOT / result_role["path"])
    output = HERE / "actual"
    output.mkdir(exist_ok=False)
    write(output / "ONE_ATTEMPT.json", dict(root_binding_sha256=sha(binding_path), actual_roles=binding["closed_actual_roles"]))
    start = time.monotonic()
    original_affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {contract["resources"]["CPU"]})
    def timeout(signum, frame):
        raise TimeoutError("closed-only scalar budget exhausted")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(contract["resources"]["wall_limit_seconds"])
    checks, ancillary, observations = [], {}, {}
    def check(name, passed):
        checks.append(dict(name=name, passed=bool(passed)))
        if not passed:
            write(output / "first_failure.json", checks[-1])
            raise AssertionError(name)
    try:
        check("actual_result_root_identity", result["root_execution_binding_sha256"] == contract["fixed"]["timing_root_execution_binding_sha256"])
        prepared = contract["prepared_inputs"]
        check("receipt_source_identity", receipt["driver_sha256"] == prepared["timing_driver"]["sha256"]
              and receipt["launcher_sha256"] == prepared["timing_launcher"]["sha256"]
              and receipt["contract_sha256"] == prepared["timing_contract"]["sha256"])
        timing = load(ROOT / prepared["timing_contract"]["path"])
        all_complete = result["status"] == "native31_resident_gradient_timing_completed"
        science = result.get("all_declared_science_controls_passed") is True
        check("declared_science_status", science == all_complete)
        controls = result["checks"]
        for index, row in enumerate(controls):
            check("recorded_live_gate_truth_" + str(index), finite(row["error"]) and finite(row["limit"])
                  and type(row["passed"]) is bool and row["passed"] == (row["error"] <= row["limit"]))
        passed_live = sum(row["passed"] for row in controls)
        failed_live = len(controls) - passed_live
        check("all_complete_means_no_failed_live_gate", not all_complete or failed_live == 0)
        observations.update(status=result["status"], live_controls=len(controls), live_passed=passed_live,
                            live_failed=failed_live, phase=result.get("phase"), original_result_never_reclassified=True)
        caps = timing["resources"]["calls"]
        check("all_counter_names_preserved", set(result["calls"]) == set(caps))
        for name, row in result["calls"].items():
            check("call_bound_" + name, type(row["started"]) is int and type(row["completed"]) is int
                  and 0 <= row["completed"] <= row["started"] <= caps[name])
        for name, row in result["request_ledgers"].items():
            check("request_cache_" + name, all(type(v) is int and v >= 0 for v in row.values())
                  and row["hits"] + row["misses"] == row["completed"] <= row["requests"])
        base = Path(contract["actual_roles_deferred"]["result"]).parent
        for name in ("thread_affinity_restore.json", "runtime_admission.json", "completed_trials.json", "host_guards.json"):
            path = ROOT / base / name
            if path.is_file():
                ancillary[name] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path), value=load(path))
            else:
                ancillary[name] = dict(path=str(path.relative_to(ROOT)), unavailable=True)
        restore = ancillary["thread_affinity_restore.json"].get("value")
        restore_ok = False
        if restore:
            restore_ok = (restore["original_affinity"] == restore["restored_affinity"]
                          and restore["original_NumBa_mask"] == restore["restored_NumBa_mask"])
            for tid, original in restore["original_task_affinities"].items():
                restored = restore["restored_task_affinities"].get(tid)
                restore_ok = restore_ok and (restored == original or restored == "thread_exited")
        resource_ok = (receipt["exit_code"] == 0 and receipt["resource_stop"] is None
                       and receipt["peak_process_group_RSS_bytes"] <= timing["resources"]["maximum_RSS_bytes"])
        if all_complete:
            check("complete_source_epoch", result["source_map_sha256"] == contract["fixed"]["f043_source_map_sha256"])
            check("complete_contract_epoch", result["contract_sha256"] == prepared["timing_contract"]["sha256"])
            check("complete_primitive_calls", all(result["calls"][k]["completed"] == v for k, v in
                  dict(H=1, CI=1, public_full_gradient=1, private_full_gradient=21, private_owner_factory=3,
                       NPZ_load=29, CUDA_context=1, optimizer=0).items()))
            check("closed_trial_sidecar", ancillary["completed_trials.json"].get("value") == result["trials"])
            check("closed_host_sidecar", ancillary["host_guards.json"].get("value") == result["host_guards"])
            check("finally_thread_restoration", restore_ok)
            trials = result["trials"]
            expected_order = [("first", 0, n) for n in ("NumBa_CPU1", "NumBa_CPU4", "CUDA")]
            expected_order += [("explicit_warmup", 0, n) for n in ("NumBa_CPU1", "NumBa_CPU4", "CUDA")]
            expected_order += [("warm", i, n) for i, names in enumerate(timing["schedule"]["warm_triplets"]) for n in names]
            check("exact21trial_order", [(r["category"], r["ordinal"], r["backend"]) for r in trials] == expected_order)
            sums = dict(density_band_requests=0, band_expansions=0, signed_term_incidence=0, pair_actions=0,
                        one_body_actions=0, scan_kernel=0, row_kernel=0)
            for index, row in enumerate(trials):
                workers = 4 if row["backend"] == "NumBa_CPU4" else 1
                cpus = timing["resources"]["CPU4"] if workers == 4 else [timing["resources"]["CPU1"]]
                check("trial_affinity_mask_" + str(index), row["NumBa_mask"] == workers and row["affinity"] == cpus
                      and bool(row["all_task_affinities"]) and all(v == cpus for v in row["all_task_affinities"].values()))
                check("trial_finite_scope_" + str(index), all(finite(row[k]) and row[k] >= 0 for k in
                      ("raw_gradient_s", "original_metric_s", "full_gradient_metric_s", "validation_s", "archive_s"))
                      and row["full_gradient_metric_s"] == row["raw_gradient_s"] + row["original_metric_s"]
                      and row["full_gradient_metric_s"] > 0)
                a, e = row["adapter_delta"], row["engine_delta"]
                check("trial_full_primitives_" + str(index), all(a[k] == v for k, v in dict(gradient=1,
                      density_band_requests=992, band_expansions=2824, signed_term_incidence=39064, pair_actions=7727,
                      one_body_actions=31, H_owner_validation=1).items()) and e["trials"] == 1
                      and e["scan_kernel"] == e["row_kernel"] == 6)
                check("trial_allbands_" + str(index), row["bands"]["all_bands"] == 2824
                      and 0 <= row["bands"]["bitwise_bands"] <= 2824
                      and finite(row["bands"]["max_scaled_band_error"])
                      and 0 <= row["bands"]["max_scaled_band_error"] <= timing["gates"]["band_scaled"])
                cuda = row["backend"] == "CUDA"
                check("trial_device_counts_" + str(index), all(e[k] == (v if cuda else 0) for k, v in
                      dict(density_H2D_bytes=8785152, job_H2D_bytes=90368, band_D2H_bytes=25009344,
                           stream_sync=6, factor_H2D_bytes=0).items()))
                check("trial_profile_only_warmup_" + str(index), row["profile_enabled"] == (row["category"] == "explicit_warmup")
                      and len(row["warmup_packet_transferinclusive_s"]) == (6 if row["profile_enabled"] else 0))
                for k in sums:
                    sums[k] += a[k] if k in a else e[k]
            check("primitive_total_from_actual_trials", sums == result["private_primitives"])
            for name, owner in result["owners"].items():
                check("seven_complete_owner_calls_" + name, owner["adapter_calls"]["gradient"] == 7
                      and owner["engine_calls"]["trials"] == 7
                      and owner["engine_calls"]["scan_kernel"] == owner["engine_calls"]["row_kernel"] == 42)
            host_rows = result["host_guards"]
            check("all11host_samples_preserved", len(host_rows) == 11)
            for index, row in enumerate(host_rows):
                cpu_passed = []
                for cpu in timing["resources"]["CPU4"]:
                    cpurow = row["CPU"][str(cpu)]
                    cpupass = (cpurow["busy_fraction"] <= timing["resources"]["host_guards"]["maximum_idle_busy_fraction_each_used_CPU"]
                               and cpurow["system_fraction"] <= timing["resources"]["host_guards"]["maximum_system_iowait_irq_softirq_steal_fraction_each_CPU"])
                    check("host_CPU_truth_" + str(index) + "/" + str(cpu), cpurow["passed"] == cpupass
                          and all(finite(cpurow[k]) and cpurow[k] >= 0 for k in ("busy_fraction", "system_fraction")))
                    cpu_passed.append(cpupass)
                processes = [r.strip().split(", ") for r in row["GPU_processes"].splitlines() if r.strip()]
                competitors_absent = all(len(r) == 3 and int(r[0]) == row["owner_PID"] for r in processes)
                gpu_rows = [r.strip().split(", ") for r in row["GPU"].splitlines() if r.strip()]
                check("host_GPU_record_" + str(index), len(gpu_rows) == 1 and len(gpu_rows[0]) == 4
                      and int(gpu_rows[0][2]) == row["GPU_idle_percent"])
                actual_hostpass = (all(cpu_passed) and competitors_absent and row["GPU_idle_percent"] == 0
                                   and row["load1_per_onlineCPU"] <= timing["resources"]["maximum_load1_per_onlineCPU"])
                check("host_guard_truth_" + str(index), row["passed"] == actual_hostpass)
            all_host = all(r["passed"] for r in host_rows)
            check("all_host_truth", result["all_host_guards_passed"] == all_host)
            statistics_rows = {}
            for backend in ("NumBa_CPU4", "CUDA"):
                ratios = []
                for i in range(5):
                    pair = {r["backend"]: r["full_gradient_metric_s"] for r in trials if r["category"] == "warm" and r["ordinal"] == i}
                    ratios.append(pair["NumBa_CPU1"] / pair[backend])
                median = statistics.median(ratios)
                stored = result["statistics"][backend]
                check("paired_ratios_median_" + backend, stored["within_triplet_ratios"] == ratios and stored["median_ratio"] == median
                      and stored["predeclared_median_threshold"] == 1.05 and stored["all_host_guards_passed"] == all_host)
                check("conditional_candidate_speed_" + backend, stored["candidate_scoped_speed_after_clean_closed_resource_receipt"] == (all_host and median >= 1.05)
                      and stored["private_complete_gradient_scoped_speed_qualified"] is False
                      and stored["closed_resource_receipt_still_required"] is True)
                if backend == "NumBa_CPU4":
                    eff = [r / 4 for r in ratios]
                    check("logical_worker_efficiency", stored["logical_worker_efficiency"] == eff
                          and stored["median_logical_worker_efficiency"] == statistics.median(eff))
                statistics_rows[backend] = dict(within_triplet_ratios=ratios, median_ratio=median,
                    ratio_threshold_passed=median >= 1.05, all_host_guards_passed=all_host,
                    closed_resource_gate_passed=resource_ok, actual_finally_affinity_masks_restored=restore_ok,
                    private_complete_gradient_vs_privateCPU1_speed_qualified=science and all_host and resource_ok and restore_ok and median >= 1.05,
                    no_public_default_or_E2E_claim=True)
            observations["paired_statistics"] = statistics_rows
        else:
            observations["partial_actual_calls"] = result["calls"]
            observations["partial_actual_trials"] = result.get("trials", [])
            observations["host_guards"] = result.get("host_guards", [])
            observations["paired_statistics"] = dict(unavailable_or_partial_preserved=True, controlled_speed_qualified=False)
        # Ancillary JSON hashes are recorded after closed admission. No NPZ or
        # scientific array bytes are read or rehashed by this independent audit.
        provenance = {k: {f: v for f, v in r.items() if f != "value"} for k, r in ancillary.items()}
        write(output / "completion.json", dict(status="CLOSED_JSON_SCALAR_AUDIT_COMPLETED", all_metadata_checks_passed=True,
            checks=checks, checks_count=len(checks), actual_roles=binding["closed_actual_roles"], ancillary_JSON=provenance,
            root_binding_sha256=sha(binding_path), script_sha256=sha(__file__), contract_sha256=sha(HERE / "contract.json"),
            closed_receipt_exit=receipt["exit_code"], resource_stop=receipt["resource_stop"], observations=observations,
            science_controls_inherited_from_actual_live_record=science, independent_numerical_array_replay=False,
            numerical_libraries_imported=0, NPZ_reads_hashes=0, HCI_gradient_eigen_GPU_queries=0,
            repeated_benchmark=0, original_scientific_stationarity_or_experimental_FAILURE_preserved=True,
            elapsed_scalar_audit_s=time.monotonic() - start))
    except BaseException as exc:
        write(output / "completion.json", dict(status="CLOSED_METADATA_INCONSISTENCY_PRESERVED", all_metadata_checks_passed=False,
            failure_type=type(exc).__name__, failure=str(exc), checks=checks, observations=observations,
            actual_roles=binding["closed_actual_roles"], original_result_never_reclassified=True,
            numerical_libraries_imported=0, NPZ_reads_hashes=0, repeated_benchmark=0))
        raise
    finally:
        signal.alarm(0)
        os.sched_setaffinity(0, original_affinity)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", type=Path)
    args = parser.parse_args()
    if args.execute is None:
        print("SOURCE_ONLY: actual receipt/result remain unread until supplied exact CLOSED root binding")
    else:
        execute(args.execute)


if __name__ == "__main__":
    main()
