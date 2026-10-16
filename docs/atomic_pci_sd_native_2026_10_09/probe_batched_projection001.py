"""Replay complete frozen native fields against retained original CPU term2.

This is an arithmetic replay, not a new native COMMON action or physical solve.
All 414 virtuals, 22538 angular rows and full analytic responses are retained.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cuda_batched_projection_control import NativeBatchedCUDAControl  # noqa: E402
from cuda_full_term2_control import NativeCUDAFullTerm2Control  # noqa: E402
from cuda_fused_angular_control import NativeFusedCUDAControl  # noqa: E402

from atomx.numerics.bspline.coulomb_features import (  # noqa: E402
    PolynomialCellCoulombFeatures,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FrozenFields:
    def prepare_fields(self, shape):
        if shape != self.expected_shape:
            raise ValueError("complete original frozen field shape differs")
        if self.state is not None:
            return
        with np.load(self.field_path, allow_pickle=False) as data:
            phi, energies, core_e, starts, ends, rows, coeff = (
                data[key].copy()
                for key in (
                    "phi",
                    "energies",
                    "core_e",
                    "starts",
                    "ends",
                    "rows",
                    "coefficients",
                )
            )
        d = self.d
        if (
            phi.shape != (d["NHF"], d["NX"], d["NK"])
            or energies.shape != (d["NX"], d["NK"])
            or core_e.shape != (d["NN"],)
            or starts.shape != (d["NK"],)
            or ends.shape != starts.shape
            or rows.shape != (13, self.field_plan["plan_rows"])
            or coeff.shape != (rows.shape[1],)
            or not all(np.all(np.isfinite(a)) for a in (phi, energies, core_e, coeff))
            or not np.all((starts >= 1) & (starts <= ends) & (ends <= d["NX"]))
        ):
            raise ValueError("all source-bound frozen native fields/rows required")
        bound = self.density.preflight(d["NXX"] ** 2) + 8 * (
            d["NHF"] * d["NX"] * d["NK"]
            + 6 * np.prod(shape)
            + 8 * d["NHF"] * d["NXX"]
            + 3 * d["NXX"] ** 2
        )
        if bound != self.field_plan["device_bound_bytes"] or bound > self.cap:
            raise MemoryError("frozen complete payload differs or exceeds cap")
        self.starts, self.ends = tuple(starts - 1), tuple(ends)
        self.phi = tuple(
            self.cp.asarray(np.ascontiguousarray(phi[:, a:b, i]))
            for i, (a, b) in enumerate(zip(self.starts, self.ends, strict=True))
        )
        self.energies = tuple(
            self.cp.asarray(energies[a:b, i])
            for i, (a, b) in enumerate(zip(self.starts, self.ends, strict=True))
        )
        self.core_e, self.rows, self.coeff = core_e, rows.T.copy(), coeff
        self.state = tuple(self.field_plan["state"])


class FrozenNative(FrozenFields, NativeCUDAFullTerm2Control):
    def prepare(self, shape):
        self.prepare_fields(shape)


class FrozenBatched(FrozenFields, NativeBatchedCUDAControl):
    def prepare(self, shape):
        self.prepare_fields(shape)
        self.prepare_projection()


class FrozenFused(FrozenFields, NativeFusedCUDAControl):
    def prepare(self, shape):
        self.prepare_fields(shape)
        self.prepare_projection()


def main(run_name, reachable_projection=False, fused_angular=False):
    if fused_angular and not reachable_projection:
        raise ValueError("explicit reachable projection required for fused control")
    out = OWNER / run_name
    out.mkdir(exist_ok=False)
    parent = OWNER / "valence_term2_cuda009"
    plan = json.loads((parent / "cuda_full_plan_6_-1.json").read_text())
    manifest = json.loads((parent / "backend/manifest.json").read_text())
    d = manifest["dimensions"]
    shape = (d["NXX"], d["NXX"], manifest["valence_sd"]["nvh"])
    host_bound = 8 * (12 * np.prod(shape) + 4 * d["NHF"] * d["NX"] * d["NK"])
    contract = dict(
        hypothesis="one complete concatenated uv/d_uv field projection preserves every original ordered angular row and full analytic response",
        fixed="retained original68core/414virtual/ns123/Fermi/l<=3/5Ha native fields, all22538 angular rows, same numerator and derivative",
        gate_Ha=1e-11,
        host_cap_bytes=32 * 1024**3,
        host_payload_bound_bytes=int(host_bound),
        device_cap_bytes=2 * 1024**3,
        exactly_reachable_projection=reachable_projection,
        fused_disjoint_angular=fused_angular,
        load_before=os.getloadavg(),
        scope="complete frozen-field arithmetic replay with independent retained original CPU outputs; not a fresh COMMON map, solve, controlled timing or public adoption",
        stop="first complete signed/mode/finite/cap failure; preserve failures and all arrays; never alter physical space or gate",
    )
    (out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    if host_bound > contract["host_cap_bytes"]:
        raise MemoryError("complete replay host payload exceeds cap")
    paths = [
        Path(__file__),
        OWNER / "cuda_batched_projection_control.py",
        OWNER / "cuda_full_term2_control.py",
        OWNER / "cuda_valence_bridge.py",
        OWNER / "cuda_density_control.py",
        ROOT / "src/atomx/numerics/bspline/coulomb_features.py",
    ]
    if fused_angular:
        paths.append(OWNER / "cuda_fused_angular_control.py")
    source = {}
    for path in paths:
        target = out / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source[str(path.relative_to(ROOT))] = digest(path)
    (out / "source_sha256.json").write_text(json.dumps(source, indent=2) + "\n")
    library_path = parent / "backend/native_core_sd.so"
    if digest(library_path) != plan["library_sha256"]:
        raise ValueError("original frozen native library identity differs")
    field_path = parent / "cuda_full_fields_6_-1.npz"
    input_path = OWNER / "yb_full_run008/yb_full_inputs.npz"
    output_path = parent / "term2_6_-1.npz"
    parent_resource = json.loads((parent / "resource.json").read_text())
    if digest(input_path) != parent_resource["input_identity"]["input_sha256"]:
        raise ValueError("original polynomial workspace identity differs")
    bindings = {
        str(path.relative_to(ROOT)): digest(path)
        for path in (
            library_path,
            field_path,
            input_path,
            output_path,
            parent / "cuda_full_plan_6_-1.json",
            parent / "reference_binding.json",
        )
    }
    (out / "input_sha256.json").write_text(json.dumps(bindings, indent=2) + "\n")
    with np.load(input_path, allow_pickle=False) as data:
        workspace = {
            k.removeprefix("workspace_"): data[k].copy()
            for k in data.files
            if k.startswith("workspace_")
        }
    workspace = {k: v.item() if v.ndim == 0 else v for k, v in workspace.items()}
    feature = PolynomialCellCoulombFeatures(workspace)
    with np.load(output_path, allow_pickle=False) as data:
        numerator, prime, original, original_prime = (
            np.array(data[key], dtype=np.float64, order="F", copy=True)
            for key in ("numerator", "prime", "original", "original_prime")
        )
    energy = next(
        row["energy_Ha"]
        for row in json.loads((parent / "partial_result.json").read_text())
        if row["key"] == [6, -1]
    )
    controls = []
    for label, cls in (
        ("baseline", FrozenNative),
        ("batched", FrozenFused if fused_angular else FrozenBatched),
    ):
        directory = out / label
        directory.mkdir()
        options = (
            {"projection_policy": "reachable" if reachable_projection else "all"}
            if cls in (FrozenBatched, FrozenFused)
            else {}
        )
        control = cls(
            feature, d, directory, SimpleNamespace(_name=str(library_path)), **options
        )
        control.field_path, control.field_plan = field_path, plan
        controls.append(control)
    outputs, records = {}, []
    for name, control, response in (
        ("baseline_full", controls[0], True),
        ("batched_full", controls[1], True),
        ("batched_value", controls[1], False),
        ("batched_restored", controls[1], True),
    ):
        value, derivative = (np.zeros(shape, order="F") for _ in range(2))
        control.compute(energy, numerator, prime, value, derivative, response=response)
        errors = [float(np.max(abs(value - original)))]
        if response:
            errors.append(float(np.max(abs(derivative - original_prime))))
        if max(errors) > contract["gate_Ha"] or not all(
            np.all(np.isfinite(a)) for a in (value, derivative)
        ):
            raise AssertionError("complete original CPU/batched CUDA replay failed")
        outputs[name + "_value"], outputs[name + "_prime"] = value, derivative
        row = dict(name=name, signed_errors_Ha=errors, load_after=os.getloadavg())
        records.append(row)
        print(json.dumps(row), flush=True)
    mode = float(
        np.max(abs(outputs["batched_full_value"] - outputs["batched_value_value"]))
    )
    restored = max(
        float(
            np.max(
                abs(
                    outputs["batched_full" + suffix]
                    - outputs["batched_restored" + suffix]
                )
            )
        )
        for suffix in ("_value", "_prime")
    )
    if max(mode, restored) > 1e-11:
        raise AssertionError("complete batched mode recovery failed")
    np.savez_compressed(out / "outputs.npz", **outputs)
    (out / "result.json").write_text(
        json.dumps(
            dict(
                status="pass",
                cases=records,
                mode_error_Ha=mode,
                restored_error_Ha=restored,
                energy_Ha=energy,
                shape=shape,
                angular_rows=plan["plan_rows"],
                physical_columns=1,
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
    parser.add_argument("--reachable-projection", action="store_true")
    parser.add_argument("--fused-angular", action="store_true")
    args = parser.parse_args()
    main(args.run_name, args.reachable_projection, args.fused_angular)
