"""Source-bound real five-state Yb II level/splitting/observed-line audit."""

import argparse
import hashlib
import json
import os
import resource
import shutil
import signal
import sqlite3
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OWNER = HERE.parent
REPO = OWNER.parents[1]
KEYS = ((6, -1), (6, 1), (6, -2), (5, 2), (5, -3))
DEFAULT_NORM_RUN = OWNER / "parallel_six_norm001/run002"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def resolve_norm_run(norm_run):
    """Select an actual in-repository producer, preserving its strict proof."""
    path = Path(norm_run)
    if not path.is_absolute():
        path = REPO / path
    path = path.resolve(strict=True)
    if not path.is_relative_to(REPO) or not path.is_dir():
        raise ValueError("norm producer must be an in-repository run directory")
    return path


def main(out, norm_run=DEFAULT_NORM_RUN):
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        (out / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )

    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    signal.signal(
        signal.SIGALRM,
        lambda *_: (_ for _ in ()).throw(TimeoutError("strict NIST120s cap")),
    )
    signal.alarm(120)
    root_run = OWNER / "yb_full_monovalent006"
    compare = OWNER / "yb_monovalent_experiment003"
    norm_run = resolve_norm_run(norm_run)
    archive = REPO / "output/nist_spectra/spectra.sqlite"
    source_paths = [Path(__file__)] + [
        REPO / name
        for name in (
            "tools/atomic_nist_audit.py",
            "tests/test_atomic_nist_audit.py",
            "tools/export_nist_atlas.py",
            "src/atomx/reference_data/nist.py",
            "src/atomx/reference_data/matching.py",
            "src/atomx/constants.py",
            "src/atomx/config/parsing.py",
            "src/atomx/config/subshell.py",
        )
    ]
    norm_sources = json.loads((norm_run / "source_sha256.json").read_text())
    source_paths = list(
        dict.fromkeys(source_paths + [REPO / name for name in norm_sources])
    )
    input_paths = [
        archive,
        OWNER / "yb_full_run008/yb_full_inputs.npz",
        OWNER / "yb_full_run008/yb_full_solution.npz",
    ]
    input_paths += [
        root_run / name
        for name in (
            "contract.json",
            "resource.json",
            "core_replay.json",
            "source_sha256.json",
            "source_postcheck.json",
            "gpu_source_postcheck.json",
            "gpu_result.json",
            "numeric64_result.json",
        )
    ]
    input_paths += [compare / name for name in ("comparison.json", "reference.json")]
    input_paths += [
        norm_run / name
        for name in (
            "result.json",
            "contract.json",
            "source_sha256.json",
            "input_sha256.json",
            "source_and_input_postcheck.json",
            "fresh_finite_anchor.json",
            "resource.json",
            "progress.json",
        )
    ]
    for n, k in KEYS:
        input_paths += [
            root_run / f"root_{n}_{k}{suffix}" for suffix in (".json", ".npz")
        ]
        input_paths += [root_run / f"cpu_root_recheck_{n}_{k}.json"]
    sources = {str(p.relative_to(REPO)): digest(p) for p in source_paths}
    inputs = {str(p.relative_to(REPO)): digest(p) for p in input_paths}
    archive_bytes = (
        sum(p.stat().st_size for p in source_paths + input_paths) + 16 * 1024**2
    )
    save(
        "contract.json",
        dict(
            hypothesis="Existing source-certified full five roots retain their real failed accuracy gates; a common-identity audit distinguishes evaluated levels, splittings and genuinely observed lines without invented uncertainties or missing-as-zero.",
            fixed="Original Z70 Yb II V68/Fermi Dirac-Coulomb complete ns123/l3/E<5Ha wavefunctions, actual five certified root files, local ASD5.12 snapshot and original1% gate; no roots/physics/weights are changed.",
            required_keys=KEYS,
            norm_producer_directory=str(norm_run.relative_to(REPO)),
            source_sha256=sources,
            input_sha256=inputs,
            distinguishing_outcomes="4/4 scored excitations and2/2splittings vs2/4 genuinely observed target lines; all unknown uncertainties and unmatched/ambiguous lines remain unknown; actual percentages must reproduce original comparison exactly.",
            scientific_gates="Complete whole-panel identity and root/norm certificates; strict unique species/config/J/parity; no ground score; literal observed-vs-Ritz and air-vacuum convention; historical percentages preserved; source/input unchanged.",
            timeout_s=120,
            threads=1,
            GPU=False,
            reserved_bytes=4 * 1024**3,
            address_bytes=4 * 1024**3,
            estimated_resident_bytes=1024**3,
            disk_archive_reservation_bytes=archive_bytes,
            stop="Any identity/certificate/source/resource/provenance/acceptance failure rejects and archives failure, without replacing missing data or shrinking states.",
        ),
    )
    if shutil.disk_usage(out).free < archive_bytes:
        raise MemoryError(
            "full scientific input archive exceeds free disk; no truncation"
        )
    save("source_sha256.json", sources)
    save("input_sha256.json", inputs)
    for category, paths in (("source", source_paths), ("input", input_paths)):
        for path in paths:
            target = out / category / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    sys.path[:0] = [str(REPO), str(REPO / "src")]
    from atomx.reference_data.nist import normalise_configuration
    from tools.atomic_nist_audit import SpectralState, audit_spectrum
    from tools.export_nist_atlas import match_yb

    if any(sources[name] != value for name, value in norm_sources.items()):
        raise ValueError(
            "current norm consumer/finite anchor/helper sources differ from actual whole-array norm audit"
        )
    historical = json.loads((compare / "comparison.json").read_text())
    reference = json.loads((compare / "reference.json").read_text())
    producer = json.loads((root_run / "contract.json").read_text())
    core_resource = json.loads((root_run / "resource.json").read_text())
    norm = json.loads((norm_run / "result.json").read_text())
    norm_inputs = json.loads((norm_run / "input_sha256.json").read_text())
    finite_anchor = json.loads((norm_run / "fresh_finite_anchor.json").read_text())
    if (
        finite_anchor["status"] != "pass"
        or finite_anchor["cases"] != 104
        or finite_anchor["component_checks"] != 832
        or max(finite_anchor["maxima"].values()) > 1e-11
    ):
        raise ValueError("fresh signed complete finite anchor certificate failed")
    if (
        norm["status"] != "pass"
        or not norm["full_Yb_norm_executed"]
        or not all(norm["gates"].values())
        or len(norm["rows"]) != 5
        or sum(len(row["components"]) for row in norm["rows"]) != 18
        or historical["relative_error_gate"] != 0.01
        or not historical["no_fitted_shift"]
        or historical["highest_level_reached"]
    ):
        raise ValueError(
            "complete original-state/newnorm and unrelaxed accuracy gates required"
        )
    for key, value in norm_inputs.items():
        if key in inputs and inputs[key] != value:
            raise ValueError("norm audit and state comparator scientific inputs differ")
    if (
        inputs[str((OWNER / "yb_full_run008/yb_full_inputs.npz").relative_to(REPO))]
        != producer["input_sha256"]
        or inputs[
            str((OWNER / "yb_full_run008/yb_full_solution.npz").relative_to(REPO))
        ]
        != producer["seed_sha256"]
    ):
        raise ValueError("original full radial/core inputs changed")
    for name in (
        "source_postcheck.json",
        "gpu_source_postcheck.json",
        "gpu_result.json",
        "numeric64_result.json",
    ):
        if json.loads((root_run / name).read_text())["status"] != "pass":
            raise ValueError("original whole-panel producer execution failed")
    roots = [json.loads((root_run / f"root_{n}_{k}.json").read_text()) for n, k in KEYS]
    physical_fields = (
        "hamiltonian",
        "binding",
        "occupied_keys",
        "virtual_keys",
        "feature_identity",
        "equation",
        "amplitude_representation",
        "physical_source_sha256",
    )
    common = {name: roots[0]["provenance"][name] for name in physical_fields}
    if any(
        {name: root["provenance"][name] for name in physical_fields} != common
        for root in roots
    ):
        raise ValueError(
            "all five roots must carry one unchanged physical producer identity"
        )
    if norm["common_physical_identity"] != common:
        raise ValueError(
            "new whole-array norm and theory energy physical identities differ"
        )
    plan = core_resource["standalone_plan"]
    residuals = []
    for key, root in zip(KEYS, roots, strict=True):
        proof = json.loads(
            (root_run / f"cpu_root_recheck_{key[0]}_{key[1]}.json").read_text()
        )
        if (
            root["column_key"] != list(key)
            or proof["status"] != "pass"
            or max(proof["raw_residual_Ha"], proof["energy_residual_Ha"]) > 1e-10
            or root["channels"] != plan["channels"][str(key)]
            or root["single_keys"] != plan["single_keys"][str(key)]
        ):
            raise ValueError(
                "original whole root coordinate/residual certificate failed"
            )
        residuals.append(
            dict(
                key=key,
                raw_residual_Ha=proof["raw_residual_Ha"],
                energy_residual_Ha=proof["energy_residual_Ha"],
                scope="stationarity audits, not standard theoretical uncertainty",
            )
        )
    identity = dict(
        hamiltonian=dict(
            name=common["hamiltonian"],
            producer_binding=common["binding"],
            source=common["physical_source_sha256"],
        ),
        nucleus=dict(
            Z=70,
            profile="Fermi",
            exact_potential_payload_sha256=producer["input_sha256"],
            isotope_resolved_experimental_assignment="not supplied",
        ),
        core=dict(
            occupied=common["occupied_keys"],
            seed_sha256=producer["seed_sha256"],
            raw_amplitudes=core_resource["core_plan"]["raw_amplitudes"],
        ),
        space=dict(
            virtual=common["virtual_keys"],
            geometry=common["feature_identity"],
            complete_coordinates=plan["amplitudes"],
        ),
        energy_zero=dict(
            theory="difference of common-H attachment energies",
            key=KEYS[0],
            reference="unique evaluated YbII 6s zero",
        ),
        units="Ha",
        normalization=dict(
            representation=common["amplitude_representation"],
            physical_metric=norm["scope"],
            consumed_whole_norm_sha256=inputs[
                str((norm_run / "result.json").relative_to(REPO))
            ],
        ),
        approximation_scope=dict(
            equation=common["equation"],
            missing="nonlinear/full triples and enlarged-space convergence; common Breit/QED/recoil absent; configuration weights unverified",
        ),
    )
    states = []
    for key, root in zip(KEYS, roots, strict=True):
        label = next(r for r in reference["levels"] if r["key"] == list(key))
        if label["twoJ"] != 2 * abs(key[1]) - 1 or (
            1 if label["parity"] == "even" else -1
        ) != (-1) ** (abs(key[1]) - (key[1] < 0)):
            raise ValueError("declared nominal kappa and resolved J/parity disagree")
        states.append(
            SpectralState(
                key,
                70,
                1,
                label["configuration"],
                label["twoJ"],
                1 if label["parity"] == "even" else -1,
                root["attachment_energy_Ha"],
                identity,
            )
        )
    frozen_db = out / "input" / archive.relative_to(REPO)
    db = sqlite3.connect(frozen_db.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    db.execute("BEGIN")
    metadata = dict(db.execute("select key,value from metadata"))
    snapshot = {
        name: [
            dict(r)
            for r in db.execute(
                f"select * from {name} where z=70 and charge=1",
            )
        ]
        for name in ("levels", "lines", "source_files")
    }
    db.close()
    save(
        "reference_snapshot.json",
        dict(
            metadata=metadata,
            source_database_sha256=inputs[str(archive.relative_to(REPO))],
            **snapshot,
        ),
    )
    report = audit_spectrum(
        states,
        snapshot["levels"],
        snapshot["lines"],
        zero_key=KEYS[0],
        required_keys=KEYS,
        splitting_pairs=((KEYS[1], KEYS[2]), (KEYS[3], KEYS[4])),
        transition_targets=tuple(
            (KEYS[0], key, "E1" if key[0] == 6 else "E2") for key in KEYS[1:]
        ),
        relative_goal=0.01,
    )
    existing = match_yb(historical["rows"], snapshot["levels"])
    for row in report["excitation"][1:]:
        old = next(r for r in historical["rows"] if r["key"] == list(row["key"]))
        atlas = next(
            r
            for r in existing
            if r["two_j"] == states[KEYS.index(row["key"])].two_j
            and r["configuration"]
            == normalise_configuration(states[KEYS.index(row["key"])].configuration)
        )
        if (
            abs(row["calculated_cm1"] - old["computed_cm1"]) > 1e-9
            or abs(row["reference_cm1"] - old["experimental_cm1"]) > 1e-9
            or abs(row["relative_error"] - atlas["error_percent"] / 100) > 1e-12
            or row["goal_pass"] != atlas["goal_pass"]
        ):
            raise ValueError("actual comparison/ASD/atlas gate or percentages changed")
    for row in report["splittings"]:
        old = next(
            r
            for r in historical["splittings"]
            if r["lower"] == list(row["lower_key"])
            and r["upper"] == list(row["upper_key"])
        )
        if (
            abs(row["calculated_cm1"] - old["computed_cm1"]) > 1e-9
            or abs(row["reference_cm1"] - old["experimental_cm1"]) > 1e-9
            or abs(row["relative_error"] - old["relative_error"]) > 1e-12
            or row["goal_pass"] != old["gate_pass"]
        ):
            raise ValueError(
                "original complete splitting failures or numerical acceptance changed"
            )
    report["provenance"] = dict(
        source_sha256=sources,
        input_sha256=inputs,
        original_producer_source_bound=True,
        new_roots_computed=False,
        fullnorm_current_source_actual=True,
        norm_producer_directory=str(norm_run.relative_to(REPO)),
        reference_metadata=metadata,
        reference_source_files=snapshot["source_files"],
    )
    report["numerical_stationarity"] = residuals
    save("audit.json", report)
    save("iteration_requests.json", report["iteration_requests"])
    unchanged = all(
        digest(REPO / name) == value for name, value in {**sources, **inputs}.items()
    )
    if not unchanged:
        raise ValueError(
            "source/scientific input changed during current comparator execution"
        )
    save(
        "source_and_input_postcheck.json",
        dict(status="pass", sources=len(sources), inputs=len(inputs)),
    )
    result = dict(
        status="pass",
        gates=dict(
            actual_full_theory_same_H_core_space_zero=True,
            current_whole_array_norm_consumed=True,
            strict_unique_state_matching=True,
            observed_not_Ritz_or_level_derived=True,
            multipole_species_medium_coverage_preserved=True,
            missing_uncertainty_never_zero=True,
            original1percent_accuracy_failures_preserved=True,
            whole_state_gap_requests_emitted=True,
            source_and_input_unchanged=True,
        ),
        scientific_accuracy_gates=dict(
            excitation=report["metrics"]["excitation_cm1"]["complete_and_all_pass"],
            splitting=report["metrics"]["splitting_cm1"]["complete_and_all_pass"],
            observed_wavelength=report["metrics"]["observed_wavelength_nm"][
                "complete_and_all_pass"
            ],
        ),
        coverage=report["coverage"],
        metrics=report["metrics"],
        seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        card_summary="YbII原完整五态同H/空间/零点审计通过；4能级、2精细分裂仍保留原失败，真实观测线仅2/4，缺失不确定度和E2观测线不计零误差；已输出必须全五态联合计算的gap队列，未产生新谱或精度提升。",
        highest_level_reached=False,
    )
    save("result.json", result)
    signal.alarm(0)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "scientific_accuracy_gates",
                    "seconds",
                    "peak_rss_bytes",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--norm-run",
        type=Path,
        default=DEFAULT_NORM_RUN,
        help="Actual whole-array norm proof directory, repo-relative or absolute; defaults to the historical parallel_six_norm001/run002 producer.",
    )
    args = parser.parse_args()
    try:
        main(args.output, norm_run=args.norm_run)
    except Exception as error:
        if args.output.is_dir():
            failure = dict(type=type(error).__name__, message=str(error))
            (args.output / "failure.json").write_text(
                json.dumps(failure, indent=2) + "\n"
            )
            (args.output / "result.json").write_text(
                json.dumps(
                    dict(
                        status="fail",
                        gates={"declared": False},
                        card_summary=str(error),
                        highest_level_reached=False,
                    ),
                    indent=2,
                )
                + "\n"
            )
        raise
