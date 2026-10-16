"""Paired frozen-orbital Yb II DC/static/dynamic Breit experiment, no SD splice."""

import argparse
import hashlib
import json
import os
import pickle
import resource
import signal
import tarfile
import time
import traceback
from pathlib import Path

import numpy as np

OWNER = Path(__file__).resolve().parent
REPO = OWNER.parents[1]
BASE = REPO / "docs/rare_earth_yb_spectrum_2026_10_08/results/v68_default"
REFERENCE = OWNER / "yb_monovalent_experiment003/comparison.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def sources():
    paths = sorted((REPO / "src/atomx").rglob("*.py")) + [Path(__file__)]
    return {str(path.relative_to(REPO)): sha(path) for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="yb_dynamic_breit001")
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name:
        raise ValueError("run name must be a single new directory name")
    out = OWNER / args.run_name
    out.mkdir(exist_ok=False)
    initial_sources = sources()
    inputs = {
        str(path.relative_to(REPO)): sha(path)
        for path in [
            BASE / name
            for name in (
                "result.pkl",
                "orbitals_workspace.npz",
                "structure.json",
                "core.json",
                "status.json",
            )
        ]
        + [REFERENCE]
    }
    contract = dict(
        hypothesis="At common fixed Yb II spinors, finite-frequency transverse exchange has "
        "a measured excitation increment distinct from static Breit; neither is "
        "assumed to repair missing correlation.",
        fixed="Z70, Fermi A175 empirical radius, 69 electrons, V68 reference, "
        "24 identical radial orbitals and five one-dimensional J/parity blocks; no fit",
        frequency="Real Coulomb gauge, symmetric average of the two vertex orbital differences; "
        "core eigenenergies and each actual valence eigenenergy from the saved DF solve",
        coverage="Frozen-orbital CI expectation/rediagonalization, not frequency-relaxed DF, "
        "not native reduced SD, not screened/multi-photon QED, not dressed rates",
        numerical_gates=dict(
            CI_absolute_residual_Ha=1e-8,
            DC_replay_Ha=1e-9,
            zero_frequency_matrix_Ha=1e-9,
            matching_overlap_squared=0.99,
        ),
        experiment_gate=dict(
            original_frozen_core_relative_goal=0.05,
            reference="NIST Handbook M67; no supplied level uncertainties",
        ),
        budget=dict(
            wall_seconds=900,
            virtual_address_bytes=3 * 1024**3,
            declared_cached_array_bound="128 entries per transverse array cache, "
            "12 FP64 cell arrays per entry; scalar cache 8192; no orbital deletion",
        ),
        stopping="Any numerical, source, input, matching or resource failure stops and is retained; "
        "experimental failure is recorded without changing the model or gates",
        source_sha256=initial_sources,
        input_sha256=inputs,
        load_average=list(os.getloadavg()),
        timing="Live-load wall observations only; no speedup claim",
        references=[
            "https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=920054",
            "https://github.com/flexible-atomic-code/fac/blob/master/faclib/radial.c",
        ],
    )
    save(out / "contract.json", contract)
    with tarfile.open(out / "source_snapshot.tar.gz", "w:gz") as archive:
        for name in initial_sources:
            archive.add(REPO / name, arcname=name)
    start = time.monotonic()

    def timeout(*_):
        raise TimeoutError("900-second declared paired Breit cap")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(900)
    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    try:
        from atomx.constants import HARTREE_CM1
        from atomx.physics.csf.matrix import build_ci_matrix
        from atomx.physics.csf.radial_snapshot import CIRadialSnapshot
        from atomx.potentials.nuclear import make_nuclear_potential

        if not json.loads((BASE / "status.json").read_text())["passed"]:
            raise ValueError("passed baseline required")
        with (BASE / "result.pkl").open("rb") as stream:
            baseline = pickle.load(stream)
        structure = json.loads((BASE / "structure.json").read_text())
        levels = structure["levels"]
        energies = {}
        for csf in baseline.csfs:
            for subshell in csf.subshells:
                key = subshell.n, subshell.kappa
                if subshell.occupation != 1:
                    energies[key] = baseline.orbital_energies[subshell.label]
        for level in levels:
            csf = baseline.csfs[level["dominant_CSF_index"]]
            active = [(s.n, s.kappa) for s in csf.subshells if s.occupation == 1]
            if len(active) != 1:
                raise ValueError("one valence spinor per baseline state required")
            energies[active[0]] = level["valence_energy_hartree"]
        if set(energies) != set(baseline.orbitals_dict):
            raise ValueError("actual finite-frequency orbital map is incomplete")
        with np.load(BASE / "orbitals_workspace.npz") as payload:
            for i, key in enumerate(map(tuple, payload["orbital_keys"])):
                np.testing.assert_array_equal(
                    payload["P"][i], baseline.orbitals_dict[key][0]
                )
                np.testing.assert_array_equal(
                    payload["Q"][i], baseline.orbitals_dict[key][1]
                )
        ws = baseline.bsp_data
        snapshot = CIRadialSnapshot(
            baseline.orbitals_dict,
            ws,
            70,
            ws["ns"],
            nuclear_potential=make_nuclear_potential(70, "fermi"),
        )
        keys = tuple(sorted(baseline.orbitals_dict))
        reference = {
            tuple(row["key"]): row["experimental_cm1"]
            for row in json.loads(REFERENCE.read_text())["rows"]
        }
        controls, matrices, measured = {}, {}, {}
        for mode, options in (
            ("DC", {}),
            ("static", dict(include_breit=True)),
            (
                "zero_frequency",
                dict(include_breit=True, breit_orbital_energies={k: 0.0 for k in keys}),
            ),
            ("dynamic", dict(include_breit=True, breit_orbital_energies=energies)),
        ):
            stage = time.monotonic()
            ci = snapshot.solve(baseline.csfs, keys, **options)
            # Independently rebuild the actual requested full Hamiltonian.
            matrix = build_ci_matrix(
                baseline.csfs,
                baseline.orbitals_dict,
                keys,
                ws,
                70,
                ws["ns"],
                nuclear_potential=snapshot._nuclear_potential,
                radial_snapshot=snapshot,
                **options,
            )
            residual = np.linalg.norm(
                matrix @ ci.mixing_coefficients
                - ci.mixing_coefficients * ci.state_energies,
                axis=0,
            )
            overlap = abs(baseline.mixing_coefficients.T @ ci.mixing_coefficients) ** 2
            matches = np.argmax(overlap, axis=1)
            weights = overlap[np.arange(len(matches)), matches]
            if (
                len(set(matches)) != len(levels)
                or min(weights) < 0.99
                or max(residual) >= 1e-8
            ):
                raise ValueError(
                    "actual Hamiltonian residual or unique state-matching gate failed"
                )
            reordered = ci.state_energies[matches]
            for old, new in enumerate(matches):
                csf = baseline.csfs[int(np.argmax(abs(ci.mixing_coefficients[:, new])))]
                if (csf.total_J2, csf.parity) != (
                    levels[old]["two_J"],
                    levels[old]["parity"],
                ):
                    raise ValueError("symmetry identity changed")
            measured[mode] = reordered
            matrices[mode] = matrix
            controls[mode] = dict(
                total_energies_Ha=reordered.tolist(),
                max_CI_residual_Ha=float(max(residual)),
                reference_overlaps_squared=weights.tolist(),
                source_provenance=ci.provenance,
                wall_seconds_uncontrolled=time.monotonic() - stage,
            )
            print(
                mode,
                "residual",
                max(residual),
                "seconds",
                time.monotonic() - stage,
                flush=True,
            )
        replay = float(max(abs(measured["DC"] - baseline.state_energies)))
        zero = float(np.max(abs(matrices["zero_frequency"] - matrices["static"])))
        if replay > 1e-9 or zero > 1e-9:
            raise ValueError("DC replay or zero-frequency static operator gate failed")
        rows = []
        ground = next(i for i, level in enumerate(levels) if level["label"] == "6s_1/2")
        for i, level in enumerate(levels):
            csf = baseline.csfs[level["dominant_CSF_index"]]
            key = next((s.n, s.kappa) for s in csf.subshells if s.occupation == 1)
            excitation = {
                mode: float((e[i] - e[ground]) * HARTREE_CM1)
                for mode, e in measured.items()
            }
            exp = reference[key]
            static_shift = excitation["static"] - excitation["DC"]
            dynamic_shift = excitation["dynamic"] - excitation["static"]
            rows.append(
                dict(
                    label=level["label"],
                    key=list(key),
                    two_J=level["two_J"],
                    parity=level["parity"],
                    excitation_cm1=excitation,
                    static_minus_DC_cm1=static_shift,
                    dynamic_minus_static_cm1=dynamic_shift,
                    dynamic_minus_DC_cm1=static_shift + dynamic_shift,
                    experimental_cm1=exp,
                    dynamic_minus_experiment_cm1=excitation["dynamic"] - exp,
                    dynamic_relative_error=None
                    if exp == 0
                    else excitation["dynamic"] / exp - 1,
                )
            )
        source_unchanged = sources() == initial_sources
        input_unchanged = all(
            sha(REPO / name) == value for name, value in inputs.items()
        )
        if not source_unchanged or not input_unchanged:
            raise ValueError("source/input identity changed during paired run")
        result = dict(
            status="pass: paired numerical/operator gates; experiment evaluated separately",
            DC_replay_max_Ha=replay,
            zero_frequency_matrix_max_Ha=zero,
            modes=controls,
            rows=rows,
            original_experimental_goal_pass=all(
                abs(r["dynamic_relative_error"]) <= 0.05
                for r in rows
                if r["dynamic_relative_error"] is not None
            ),
            source_unchanged=source_unchanged,
            input_unchanged=input_unchanged,
            wall_seconds_uncontrolled=time.monotonic() - start,
            peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            contract_sha256=sha(out / "contract.json"),
            scope=contract["coverage"],
            highest_level_reached=False,
            card_summary="Yb⁺ 五态同轨道配对通过数值门；动态相对静态的激发能增量：\n"
            + "\n".join(
                f"• {r['label']}：{r['dynamic_minus_static_cm1']:+.4f} cm⁻¹"
                for r in rows
                if r["label"] != "6s_1/2"
            )
            + "\n这是冻结核 CI 因素实验；未叠加到现有 DC-SD 能级，也未认证寿命或完整 QED。",
        )
        save(out / "result.json", result)
        np.savez_compressed(out / "matrices.npz", **matrices)
        save(
            out / "source_and_input_postcheck.json",
            dict(source_sha256=sources(), input_sha256=inputs, passed=True),
        )
    except Exception:
        save(
            out / "failure.json",
            dict(
                traceback=traceback.format_exc(), wall_seconds=time.monotonic() - start
            ),
        )
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
