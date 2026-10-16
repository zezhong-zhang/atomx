"""One fixed full4f residual/action profile before any optimization."""

import cProfile
import hashlib
import io
import json
import os
import pstats
import resource
import signal
import tarfile
import time
import traceback
from pathlib import Path

import numpy as np

from atomx.methods.dirac_fock import SCFResult
from atomx.methods.tensor_coupled_cluster import (
    CCSDTTensorOperator,
    TensorCoreResult,
    resource_plan,
)
from atomx.physics.csf.spinor_integrals import SpinorIntegrals, build_spinor_integrals
from atomx.potentials.nuclear import make_nuclear_potential

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").is_file())
OLD = HERE.parent / "native_tensor_ccsdt001"
DHF = HERE.parent / "yb_dhf_table001"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    source = json.loads((OLD / "source.json").read_text())
    actual = {
        str(p.relative_to(ROOT)): sha(p)
        for p in sorted((ROOT / "src/atomx").rglob("*.py"))
    }
    assert actual == source["files"]
    for name in ("source.json", "source.tar.gz"):
        (HERE / ("before_" + name)).write_bytes((OLD / name).read_bytes())
    with tarfile.open(HERE / "before_source.tar.gz") as a:
        a.extractall(HERE / "before_tree")
    contract = dict(
        hypothesis="Adding both complete4f subshells reveals nonzero new core-channel residuals at the embedded old solved T and measures actual complete native tensor work. Profile before changing contractions; this off-shell probe is not a new spectrum.",
        fixed=dict(
            Z=70,
            radial="same ns180/ks9/R500/RKB/A171 native bank",
            occupied=[[5, -1], [5, 1], [4, 3], [4, -4]],
            virtual=[[6, -1], [5, 2], [5, -3]],
            correlated_core_electrons=18,
            frozen_electrons=50,
            operator="DC only; preserve frozen contractions and original common68e HF energy origin",
            core_amplitudes="embed all1324 previous solved4core amplitudes, zero new4f amplitudes; no numerical screening",
            right="deterministic seed62014, all even-parity M=1/2 R3 coordinates, normalized with nonzero6s bare component",
            equations="all231core/341EA terms, fullSDT andR3",
            source_map_sha256=source["source_map_sha256"],
        ),
        resources=dict(
            cpu=27,
            threads=1,
            AS_bytes=12 * 1024**3,
            seconds=300,
            tensor_MiB=4096,
            load1_max=8,
            available_min_bytes=32 * 1024**3,
            scalar_plan=resource_plan(18, 12),
        ),
        gates=dict(
            repartition_fock_Ha=1e-10,
            repartition_effective_h_Ha=1e-10,
            repartition_frozen_Ha=1e-10,
            repartition_ERI_Ha=1e-10,
            embedded_old_projected_residual_Ha=1e-10,
            embedded_old_energy_Ha=1e-10,
            new4f_residual_min_Ha=1e-8,
        ),
        stopping="First failed source/input/physics/resource gate stops. No automatic retries or model/cut/tolerance reduction. Residual/action profile only; no full18e core or EA solve in this probe.",
        inputs={
            str(p.relative_to(ROOT)): sha(p)
            for p in [
                Path(__file__),
                OLD / "actual001/CCSDT_core.npz",
                OLD / "actual001/operator.json",
                OLD / "actual001/integrals.npz",
                DHF / "actual004/core_ns180.h5",
                DHF / "actual004/blocks_ns180.npz",
            ]
        },
    )
    save("probe_contract.json", contract)
    assert os.getloadavg()[0] <= 8
    available = next(
        int(x.split()[1]) * 1024
        for x in Path("/proc/meminfo").read_text().splitlines()
        if x.startswith("MemAvailable:")
    )
    assert available >= 32 * 1024**3
    os.sched_setaffinity(0, {27})
    resource.setrlimit(resource.RLIMIT_AS, (12 * 1024**3, 12 * 1024**3))

    def alarm(*_):
        raise TimeoutError("predeclared300-second probe cap")

    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(300)
    report = dict(
        status="RUNNING", controls=[], source_map_sha256=source["source_map_sha256"]
    )

    def gate(name, value, cap=None):
        measured = float(value) if cap is not None else bool(value)
        passed = measured <= cap if cap is not None else measured
        report["controls"].append(
            dict(name=name, measured=measured, cap=cap, passed=bool(passed))
        )
        assert passed, (name, measured, cap)

    try:
        radial = SCFResult.load(DHF / "actual004/core_ns180.h5")
        radial._nuclear_potential = make_nuclear_potential(70, "fermi", 171)
        with np.load(DHF / "actual004/blocks_ns180.npz") as z:
            orbitals = {
                tuple(map(int, k)): (p, q)
                for k, p, q in zip(
                    z["all_orbital_keys"], z["all_P"], z["all_Q"], strict=True
                )
            }
        occupied = ((5, -1), (5, 1), (4, 3), (4, -4))
        virtual = ((6, -1), (5, 2), (5, -3))
        frozen = tuple(k for k in radial.orbitals_by_subshell() if k not in occupied)
        gate(
            "50frozen_electrons_15complete_radials",
            sum(2 * abs(k) for n, k in frozen) == 50 and len(frozen) == 15,
        )
        original = build_spinor_integrals(
            orbitals,
            radial._bsp_data,
            70,
            180,
            occupied_keys=occupied,
            virtual_keys=virtual,
            frozen_keys=frozen,
            nuclear_potential=radial._nuclear_potential,
            max_integral_bytes=64 * 1024**2,
            metric_tolerance=1e-10,
        )
        origin = original.reference_energy_Ha
        identity = dict(original.identity)
        identity.update(
            probe="complete4f14 active;18correlated/50frozen/12virtual",
            energy_origin_Ha=origin,
        )
        obj = SpinorIntegrals(
            original.labels,
            18,
            original.one_body_Ha,
            original.fock_Ha,
            original.packed_eri_Ha,
            original.frozen_energy_Ha - origin,
            0.0,
            identity,
        )
        gate("30spinors18occupied", len(obj.labels) == 30 and obj.noccupied == 18)
        ledger = json.loads((OLD / "actual001/operator.json").read_text())
        with np.load(OLD / "actual001/integrals.npz") as z:
            old = SpinorIntegrals(
                tuple(map(tuple, ledger["labels"])),
                4,
                z["h"],
                z["fock"],
                z["g"],
                ledger["shifted_frozen_energy_Ha"],
                0.0,
                ledger["identity"],
            )
        indices = [obj.labels.index(x) for x in old.labels]
        gate("same_origin", abs(origin - ledger["absolute_energy_origin_Ha"]), 1e-10)
        gate(
            "same_repartitioned_Fock",
            np.max(abs(obj.fock_Ha[np.ix_(indices, indices)] - old.fock_Ha)),
            1e-10,
        )
        shifted_h = obj.one_body_Ha[np.ix_(indices, indices)].copy()
        for p, i in enumerate(indices):
            for q, j in enumerate(indices):
                shifted_h[p, q] += sum(obj.eri(i, a, j, a) for a in range(4, 18))
        gate("same_old_effective_h", np.max(abs(shifted_h - old.one_body_Ha)), 1e-10)
        shifted_frozen = (
            obj.frozen_energy_Ha
            + np.trace(obj.one_body_Ha[4:18, 4:18])
            + 0.5 * sum(obj.eri(i, j, i, j) for i in range(4, 18) for j in range(4, 18))
        )
        gate(
            "same_old_frozen_constant",
            abs(shifted_frozen - old.frozen_energy_Ha),
            1e-10,
        )
        gate(
            "same_all_old_ERI",
            max(
                abs(
                    obj.eri(indices[p], indices[q], indices[r], indices[s])
                    - old.eri(p, q, r, s)
                )
                for p in range(16)
                for q in range(16)
                for r in range(16)
                for s in range(16)
            ),
            1e-10,
        )
        save(
            "operator.json",
            dict(
                labels=obj.labels,
                identity=identity,
                absolute_energy_origin_Ha=origin,
                shifted_frozen_energy_Ha=obj.frozen_energy_Ha,
            ),
        )
        np.savez(
            HERE / "integrals.npz",
            h=obj.one_body_Ha,
            fock=obj.fock_Ha,
            g=obj.packed_eri_Ha,
        )
        op = CCSDTTensorOperator(obj, memory_MiB=4096)
        oldop = CCSDTTensorOperator(old)
        with np.load(OLD / "actual001/CCSDT_core.npz") as z:
            oldvector = z["amplitudes"]
        vector = np.zeros(op.core.size)
        old_indices = []
        new_channel = []
        for (ns, np_, nh, _), (ss, sp, sh, _) in zip(
            op.core.blocks, oldop.core.blocks, strict=True
        ):
            lookup = {
                (tuple(p), tuple(h)): ss.start + i
                for i, (p, h) in enumerate(zip(sp, sh, strict=True))
            }
            for i, (p, h) in enumerate(zip(np_, nh, strict=True)):
                key = (tuple(p), tuple(h))
                if key in lookup:
                    vector[ns.start + i] = oldvector[lookup[key]]
                    old_indices.append(ns.start + i)
                else:
                    new_channel.append(ns.start + i)
        gate("1324_old_coordinates_embedded", len(old_indices) == 1324)
        gate(
            "T_norm_unchanged",
            abs(np.linalg.norm(vector) - np.linalg.norm(oldvector)),
            1e-14,
        )
        right = np.zeros(op.attachment.size)
        rng = np.random.default_rng(62014)
        active = []
        for section, particles, holes, _ in op.attachment.blocks:
            for i, (p, h) in enumerate(zip(particles, holes, strict=True)):
                p_labels = [obj.labels[op.o + int(a)] for a in p]
                h_labels = [obj.labels[int(a)] for a in h]
                m2 = sum(x[2] for x in p_labels) - sum(x[2] for x in h_labels)
                parity = (
                    sum(x[1] if x[1] > 0 else -x[1] - 1 for x in p_labels + h_labels)
                    % 2
                )
                if m2 == 1 and parity == 0:
                    active.append(section.start + i)
        right[active] = rng.normal(scale=0.01, size=len(active))
        right[1] += 1.0
        right /= np.linalg.norm(right)
        np.savez(
            HERE / "fixed_inputs.npz",
            core=vector,
            right=right,
            old_indices=old_indices,
            new_channel=new_channel,
            active_EA=active,
        )
        profiler = cProfile.Profile()
        profiler.enable()
        start = time.perf_counter()
        residual = op.core_residual(vector)
        core_seconds = time.perf_counter() - start
        energy = op.core_energy(vector)
        gate(
            "embedded_old_retained_equations", np.max(abs(residual[old_indices])), 1e-10
        )
        oldenergy = json.loads((OLD / "actual001/result.json").read_text())["cores"][
            "CCSDT"
        ]["energy_Ha"]
        gate("embedded_old_energy", abs(energy - oldenergy), 1e-10)
        gate(
            "nonzero_new4f_channel_residual", np.max(abs(residual[new_channel])) > 1e-8
        )
        trial = TensorCoreResult(
            vector, energy, energy, float(np.max(abs(residual))), 0, op.operator_sha256
        )
        start = time.perf_counter()
        product = op.attachment_action(trial, right)
        ea_seconds = time.perf_counter() - start
        profiler.disable()
        profiler.dump_stats(HERE / "before_profile.prof")
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats("cumtime").print_stats(35)
        (HERE / "before_profile.txt").write_text(stream.getvalue())
        np.savez(HERE / "before_values.npz", core_residual=residual, EA_action=product)
        report.update(
            status="PASS_FULL4F_OFFSHELL_PROBE",
            core_residual_seconds_profiled=core_seconds,
            EA_action_seconds_profiled=ea_seconds,
            new4f_residual_by_rank={
                str(k + 1): float(np.max(abs(residual[s][np.any(h >= 4, axis=1)])))
                for k, (s, p, h, shape) in enumerate(op.core.blocks)
            },
            core_energy_Ha=energy,
            retained_old_coordinates=len(old_indices),
            new_channel_coordinates=len(new_channel),
            active_EA_Mhalf_even=len(active),
            core_T3_norm=float(np.linalg.norm(vector[op.core.blocks[2][0]])),
            operator_sha256=op.operator_sha256,
            peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        )
    except Exception:
        report.update(status="FAILED", exception=traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        save("probe_result.json", report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
