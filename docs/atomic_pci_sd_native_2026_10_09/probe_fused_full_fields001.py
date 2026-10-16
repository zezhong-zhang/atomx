"""All five complete frozen native term2 field/response/mode controls.

The unchanged GPU baseline's full native affine maps were certified against
CPU in cuda_valence002. This is an explicitly transitive arithmetic comparison,
not another native COMMON initialization or fresh whole SD/physical solve.
"""

import argparse
import hashlib
import itertools
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from probe_batched_projection001 import FrozenFused, FrozenNative

from atomx.numerics.bspline.coulomb_features import PolynomialCellCoulombFeatures

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(run_name):
    out = OWNER / run_name
    out.mkdir(exist_ok=False)
    parent = OWNER / "cuda_valence002"
    resource = json.loads((parent / "resource.json").read_text())
    manifest = json.loads((parent / "backend/manifest.json").read_text())
    dims = manifest["dimensions"]
    shape = (dims["NXX"], dims["NXX"], manifest["valence_sd"]["nvh"])
    host_bound = int(
        8 * (16 * np.prod(shape) + 4 * dims["NHF"] * dims["NX"] * dims["NK"])
    )
    contract = dict(
        hypothesis="fused exactly disjoint angular rows preserve complete original term2/analytic-response/mode outputs across all five external partial waves",
        fixed="archived68core/414virtual/ns123/Fermi/l<=3/5Ha fields and all native rows, unchanged packed numerator/response and per-column energy",
        gate_Ha=1e-11,
        host_cap_bytes=32 * 1024**3,
        host_payload_bound_bytes=host_bound,
        device_cap_bytes=2 * 1024**3,
        role="transitive complete frozen-field arithmetic; original CUDA002 full CPU/GPU map anchor, not fresh COMMON/map/solve/controlled-timing/public adoption",
        load=os.getloadavg(),
        stop="first complete signed/finite/identity/cap failure, no screening/gate alteration; retain arrays",
    )
    (out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    if host_bound > contract["host_cap_bytes"]:
        raise MemoryError("complete five-column replay exceeds host cap")
    files = [
        Path(__file__),
        *(
            OWNER / name
            for name in (
                "probe_batched_projection001.py",
                "cuda_fused_angular_control.py",
                "cuda_batched_projection_control.py",
                "cuda_full_term2_control.py",
                "cuda_density_control.py",
                "cuda_valence_bridge.py",
            )
        ),
        ROOT / "src/atomx/numerics/bspline/coulomb_features.py",
    ]
    hashes = {}
    for path in files:
        target = out / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(path.relative_to(ROOT))] = digest(path)
    (out / "source_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    physical = parent / "physical_inputs.npz"
    with np.load(physical, allow_pickle=False) as data:
        workspace = {
            name.removeprefix("workspace_"): data[name].copy()
            for name in data.files
            if name.startswith("workspace_")
        }
    workspace = {k: v.item() if v.ndim == 0 else v for k, v in workspace.items()}
    feature = PolynomialCellCoulombFeatures(workspace)
    library = parent / "backend/native_core_sd.so"
    binding = {
        str(path.relative_to(ROOT)): digest(path)
        for path in (
            physical,
            library,
            parent / "result.json",
            parent / "source_sha256.json",
            parent / "resource.json",
        )
    }
    records = []
    for ordinal, key in enumerate(((6, -1), (6, 1), (6, -2), (5, 2), (5, -3)), 1):
        label = f"{key[0]}_{key[1]}"
        context = parent / "cuda" / f"context_{2 * ordinal:06d}"
        field_path, plan_path = (
            context / f"cuda_full_fields_{label}.npz",
            context / f"cuda_full_plan_{label}.json",
        )
        plan = json.loads(plan_path.read_text())
        column_path = parent / f"column_{label}.npz"
        column = json.loads((parent / f"column_{label}.json").read_text())
        if digest(library) != plan["library_sha256"] or plan["state"] != list(key):
            raise ValueError("complete original field/library identity differs")
        with np.load(field_path, allow_pickle=False) as fields:
            starts, ends, rows = fields["starts"] - 1, fields["ends"], fields["rows"].T
        selections = set()
        for _, group in itertools.groupby(rows, lambda row: tuple(row[:3])):
            for _, core in itertools.groupby(group, lambda row: tuple(row[3:7])):
                selections.add(tuple(dict.fromkeys(int(row[7]) - 1 for row in core)))
        widths = [sum(int(ends[i] - starts[i]) for i in group) for group in selections]
        fused_bound = (
            plan["device_bound_bytes"]
            + 8 * (dims["NHF"] * sum(widths) + 4 * dims["NXX"] * max(widths))
            + 24 * len(rows)
        )
        # Both source-bound controls live in this process's original 2GiB pool.
        if plan["device_bound_bytes"] + fused_bound > contract["device_cap_bytes"]:
            raise MemoryError(
                "combined complete old/fused control payload exceeds original device cap"
            )
        with np.load(column_path, allow_pickle=False) as data:
            packed, packed_prime = data["numerator"].copy(), data["prime"].copy()
        numerator, prime = (np.zeros(shape, order="F") for _ in range(2))
        channels = resource["valence_plan"]["channels"][str(key)]
        offset = 0
        for c, (_, im, inn, _) in enumerate(channels):
            a, b = int(starts[im]), int(ends[im])
            u, v = int(starts[inn]), int(ends[inn])
            count = (b - a) * (v - u)
            for full, vector in ((numerator, packed), (prime, packed_prime)):
                full[a:b, u:v, c] = vector[offset : offset + count].reshape(
                    (b - a, v - u), order="F"
                )
            offset += count
        if offset != len(packed) or offset != len(packed_prime):
            raise ValueError("complete packed/Fortran field coordinates differ")
        case = out / label
        case.mkdir()
        controls = []
        for name, cls in (("original_gpu", FrozenNative), ("fused_gpu", FrozenFused)):
            directory = case / name
            directory.mkdir()
            options = {"projection_policy": "reachable"} if cls is FrozenFused else {}
            control = cls(
                feature, dims, directory, SimpleNamespace(_name=str(library)), **options
            )
            control.field_path, control.field_plan = field_path, plan
            controls.append(control)
        outputs = {}
        errors = {}
        for name, control, response in (
            ("original_full", controls[0], True),
            ("fused_full", controls[1], True),
            ("fused_value", controls[1], False),
            ("fused_restored", controls[1], True),
        ):
            value, derivative = (np.zeros(shape, order="F") for _ in range(2))
            control.compute(
                column["energy_Ha"],
                numerator,
                prime,
                value,
                derivative,
                response=response,
            )
            if not all(np.all(np.isfinite(a)) for a in (value, derivative)):
                raise AssertionError("complete five-column fused outputs nonfinite")
            outputs[name + "_value"], outputs[name + "_prime"] = value, derivative
        for name in ("fused_full", "fused_value", "fused_restored"):
            errors[name + "_value"] = float(
                np.max(abs(outputs[name + "_value"] - outputs["original_full_value"]))
            )
            if name != "fused_value":
                errors[name + "_prime"] = float(
                    np.max(
                        abs(outputs[name + "_prime"] - outputs["original_full_prime"])
                    )
                )
        if max(errors.values()) > contract["gate_Ha"]:
            raise AssertionError(
                "complete original/fused five-column signed/mode gate failed"
            )
        np.savez_compressed(
            case / "outputs.npz", numerator=numerator, prime=prime, **outputs
        )
        row = dict(
            key=key,
            energy_Ha=column["energy_Ha"],
            complete_signed_errors_Ha=errors,
            angular_rows=len(rows),
            combined_device_payload_bound_bytes=int(
                plan["device_bound_bytes"] + fused_bound
            ),
            load_after=os.getloadavg(),
        )
        records.append(row)
        for path in (field_path, plan_path, column_path):
            binding[str(path.relative_to(ROOT))] = digest(path)
        (out / "partial_result.json").write_text(json.dumps(records, indent=2) + "\n")
        (out / "input_binding.json").write_text(json.dumps(binding, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        del controls, control, outputs, numerator, prime, value, derivative
    (out / "result.json").write_text(
        json.dumps(
            dict(
                status="pass",
                columns=records,
                scope=contract["role"],
                controlled_timing=None,
                public_adopted=False,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    try:
        main(args.run_name)
    except BaseException as error:
        (OWNER / args.run_name / "failure.json").write_text(
            json.dumps(dict(type=type(error).__name__, message=str(error)), indent=2)
            + "\n"
        )
        raise
