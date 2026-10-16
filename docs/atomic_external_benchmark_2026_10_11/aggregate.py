"""Reproduce the scoped CPU table from complete, admitted benchmark epochs."""

import csv
import hashlib
import json
import statistics
from pathlib import Path

OWNER = Path(__file__).resolve().parent
INPUTS = {}


def read(name):
    path = OWNER / name
    raw = path.read_bytes()
    INPUTS[name] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def save(name, value):
    (OWNER / name).write_text(json.dumps(value, indent=2) + "\n")


def row(name, workers, times, ratios, controlled, scope, threads=None):
    if not times or not ratios:
        raise ValueError("A complete paired group is required")
    speedup = statistics.median(ratios) if controlled else None
    return {
        "workload": name,
        "workers": workers,
        "rounds": len(times),
        "seconds_median": statistics.median(times),
        "seconds_min": min(times),
        "seconds_max": max(times),
        "speedup": speedup,
        "speedup_min": min(ratios) if controlled else None,
        "speedup_max": max(ratios) if controlled else None,
        "efficiency": speedup / workers if controlled else None,
        "efficiency_min": min(ratios) / workers if controlled else None,
        "efficiency_max": max(ratios) / workers if controlled else None,
        "actual_threads_peak": threads,
        "process_thread_efficiency": (
            speedup / threads if controlled and threads else None
        ),
        "controlled": controlled,
        "scope": scope,
        "statistic": "median of within-round paired T1/Tp; times separate",
        "observed_times_seconds": times,
        "observed_paired_speedups": ratios,
    }


def main():
    read("contract.json")
    rows = []
    finite = read("sd/finite_action001/result.corrected.json")
    observations = read("sd/finite_action001/observations.json")
    assert finite["full_numerical_gate"]
    for workers in (1, 2, 4, 8):
        selected = [x for x in observations if x["backend"] == workers]
        ratios = [
            next(
                a["per_five_column_map_seconds"]
                for a in observations
                if a["round"] == x["round"] and a["backend"] == 1
            )
            / x["per_five_column_map_seconds"]
            for x in selected
        ]
        rows.append(
            row(
                "Atomx Na+ full SD action",
                workers,
                [x["per_five_column_map_seconds"] for x in selected],
                ratios,
                all(x["timing_guard_pass"] for x in observations),
                finite["scope"],
            )
        )

    yb_results = [read(f"sd/yb_round{r:03d}/result.json") for r in range(3)]
    yb_observations = []
    for r, result in enumerate(yb_results):
        verification = read(f"sd/yb_round{r:03d}/full_output_verification.json")
        assert verification["all_original_arrays_preserved"]
        assert result["precision_pass"] and result["all_workers_bitwise_equal"]
        assert not result["current_source_changes"]
        assert not result["original_input_changes"]
        samples = read(f"sd/yb_round{r:03d}/observations.json")
        yb_observations.append([x for x in samples if not x["warmup"]])
    for workers in (1, 2, 4, 8):
        times = [
            next(x["seconds"] for x in samples if x["workers"] == workers)
            for samples in yb_observations
        ]
        ratios = [
            next(x["seconds"] for x in samples if x["workers"] == 1) / t
            for samples, t in zip(yb_observations, times, strict=True)
        ]
        rows.append(
            row(
                "Atomx Yb term2 stress stage",
                workers,
                times,
                ratios,
                all(x["timing_guard_pass"] for x in yb_results),
                yb_results[0]["scope"],
            )
        )

    ambit = read("ambit/result.corrected.json")
    assert ambit["matched"]["all_precision_gates_pass"]
    scaling = ambit["ambit_native_scaling"]
    assert scaling["all_full_signed_files_bitwise_equal"]
    serial_times = []
    for workers in (2, 4, 8):
        pairs = scaling["results"][str(workers)]["pairs"]
        serial_times.extend(x["samples"]["1"]["child_wall_seconds"] for x in pairs)
        times = [x["samples"][str(workers)]["child_wall_seconds"] for x in pairs]
        ratios = [
            x["samples"]["1"]["child_wall_seconds"] / t
            for x, t in zip(pairs, times, strict=True)
        ]
        rows.append(
            row(
                "AMBiT BeII full CLI",
                workers,
                times,
                ratios,
                all(x["controlled_timing"] and x["scientific_gate"] for x in pairs),
                scaling["timing_scope"],
            )
        )
    rows.append(
        row(
            "AMBiT BeII full CLI",
            1,
            serial_times,
            [1.0] * len(serial_times),
            all(
                x["controlled"] for x in rows if x["workload"] == "AMBiT BeII full CLI"
            ),
            scaling["timing_scope"],
        )
    )

    fscc = read("fscc/run003/result.json")
    raw = read("fscc/run003/raw_samples.json")["backend"]
    assert all(fscc["gates"].values())
    fscc_pairs = []
    for workers in (1, 2):
        samples = [
            next(x for x in raw if x["directory"].endswith(f"scaling_r{r}_p{workers}"))
            for r in range(5)
        ]
        ratios = [
            next(
                x["wall_seconds"]
                for x in raw
                if x["directory"].endswith(f"scaling_r{r}_p1")
            )
            / sample["wall_seconds"]
            for r, sample in enumerate(samples)
        ]
        if workers == 2:
            fscc_pairs = ratios
        rows.append(
            row(
                "EXP-T finite CCSDT backend",
                workers,
                [x["wall_seconds"] for x in samples],
                ratios,
                all(x["controlled_timing"] for x in samples),
                "4-core/12-spinor fixed full backend solve; private waitpid/watchdog; includes CLI/I/O, excludes integral export",
                max(
                    x["actual_runtime"]["observed_backend_max_threads"] for x in samples
                ),
            )
        )

    wrapper_deltas = [
        next(
            x["wall_seconds"]
            for x in raw
            if x["directory"].endswith(f"wrapper_pair_r{r}_wrapper")
        )
        - next(
            x["wall_seconds"]
            for x in raw
            if x["directory"].endswith(f"wrapper_pair_r{r}_bare")
        )
        for r in range(5)
    ]
    gpu = read("gpu/summary.json")
    experiment = read("experimental_precision_context.json")
    rejected_p4 = read("fscc/thread_preflight002/p4/runtime_thread_cap_rejection.json")
    rejected_scaling = {p: dict(fscc["backend_scaling"][p]) for p in ("4", "8")}
    rejected_scaling["4"]["first_cap_crossing_thread_count"] = 9
    rejected_scaling["4"]["observed_total_threads_peak"] = rejected_p4[
        "observed_backend_max_threads"
    ]
    summary = {
        "cpu": rows,
        "matched_AMBiT": {
            k: ambit["matched"][k]
            for k in (
                "orbitals",
                "integrals",
                "signed_matrix_entries",
                "components",
                "max_error_Ha",
                "all_precision_gates_pass",
                "all_environment_gates_pass",
                "statistics",
                "timing_scope",
                "experimental_accuracy_claim",
            )
        },
        "finite_SD": {
            k: finite[k]
            for k in (
                "scope",
                "maximum_signed_error_Ha",
                "maximum_relative_error",
                "scaling",
                "setup_seconds",
                "scf_and_virtual_seconds",
                "peak_rss_bytes",
            )
        },
        "Yb_complete_stage": {
            "maximum_signed_error_Ha": max(
                x["maximum_signed_error_Ha"] for x in yb_results
            ),
            "cold_setup_seconds": [x["setup_seconds"] for x in yb_results],
            "peak_rss_bytes": max(x["peak_rss_bytes"] for x in yb_results),
            "complete_shape": yb_results[0]["shape"],
            "full_core_electrons": 68,
            "full_virtual_orbitals": 414,
            "full_response": True,
            "amplitude_input": "Archived complete signed Gaussian stress numerator/response on the full physical orbital space; zero core seed, not a converged Yb SD wavefunction.",
            "whole_SD_solve_speedup_claim": False,
        },
        "FSCC_accuracy": {
            model: {
                variant: {
                    k: value[k]
                    for k in (
                        "nspinors",
                        "noccupied",
                        "core_determinants",
                        "attachment_determinants",
                        "model_dimension",
                        "core_projected_rank123_max_Ha",
                        "attachment_projected_rank012_max_Ha",
                        "heff_signed_projection_max_Ha",
                        "core_unretained_rank4_max_Ha",
                        "attachment_unretained_rank34_max_Ha",
                        "full_attachment_bloch_max_Ha",
                        "attachment_fci_method_error_max_Ha",
                        "fock_occupied_virtual_max_Ha",
                        "gates",
                        "exact_two_core_gate",
                    )
                }
                for variant, value in variants.items()
            }
            for model, variants in fscc["accuracy"].items()
        },
        "FSCC_resource_rejections": rejected_scaling,
        "FSCC_paired_speedups": fscc_pairs,
        "private_wrapper_paired_extra_seconds": wrapper_deltas,
        "private_wrapper_median_paired_extra_seconds": statistics.median(
            wrapper_deltas
        ),
        "private_wrapper_difference_of_median_times_seconds": fscc[
            "wrapper_same_backend_overhead"
        ]["median_wrapper_extra_seconds"],
        "GPU": gpu,
        "experimental_baseline": experiment,
        "highest_level_reached": False,
        "independent_standalone_pCI_claim": False,
        "whole_production_software_rank": None,
    }
    save("metrics.json", summary)
    fields = [k for k in rows[0] if not k.startswith("observed_")]
    with (OWNER / "performance.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: x[k] for k in fields} for x in rows)
    save(
        "aggregation_provenance.json",
        {
            "inputs_sha256": INPUTS,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "complete_groups_required": True,
            "paired_statistic": "Median of same-round T1/Tp; error ranges show all observations, not confidence intervals.",
            "actual_process_threads_scope": "FSCC 10 ms /proc sampled lower bound; AMBiT program team size and SD compile/runtime contract are recorded in their owners.",
            "historical_experiment_not_rerun": True,
        },
    )


if __name__ == "__main__":
    main()
