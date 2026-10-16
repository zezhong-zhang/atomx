"""Compare complete certified fixed-column data in the unchanged Yb space.

Array agreement is a same-model numerical check, not physical convergence or
experimental precision. Source manifests remain distinct, explicitly bound.
"""

import argparse
import hashlib
import json
import math
import zipfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np

OWNER = Path(__file__).resolve().parent
KEYS = ((6, -1), (6, 1), (6, -2), (5, 2), (5, -3))
PHYSICAL = (
    "hamiltonian",
    "binding",
    "physical_source_sha256",
    "occupied_keys",
    "virtual_keys",
    "frozen_keys",
    "feature_identity",
    "equation",
    "amplitude_representation",
    "backend_pin",
    "core_amplitudes_sha256",
    "valence_source_sha256",
    "energy_reference",
    "response",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive_array_extent(path):
    """Read complete array headers before allocating root or screening data."""
    maximum = 0
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            if not item.filename.endswith(".npy"):
                raise ValueError("complete numerical archive contains non-array data")
            with archive.open(item) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("unsupported numerical archive header version")
                if dtype != np.dtype(np.float64):
                    raise ValueError("complete real float64 archive required")
                extent = math.prod(shape)
                if stream.tell() + extent * dtype.itemsize != item.file_size:
                    raise ValueError("complete numerical archive byte extent differs")
                maximum = max(maximum, extent)
    return maximum


def compare(left, right, output, *, screening=False):
    left, right, output = OWNER / left, OWNER / right, OWNER / output
    if output.exists():
        raise FileExistsError("fresh comparison record required")
    contracts = [json.loads((d / "contract.json").read_text()) for d in (left, right)]
    for name in (
        "fixed",
        "external_columns",
        "energy_convention",
        "input_sha256",
        "seed_file_sha256",
        "raw_residual_gate_Ha",
        "source_gate_Ha",
    ):
        if contracts[0][name] != contracts[1][name]:
            raise ValueError("complete fixed input/state/energy/gate contract differs")
    if contracts[0]["external_columns"] != [list(k) for k in KEYS]:
        raise ValueError("all five original external columns required")
    largest = max(
        max(
            json.loads((d / "resource.json").read_text())["valence_plan"][
                "raw_amplitudes"
            ].values()
        )
        for d in (left, right)
    )
    array_paths = [
        d / f"valence_{key[0]}_{key[1]}.npz" for d in (left, right) for key in KEYS
    ]
    if screening:
        array_paths += [d / "sigma2.npz" for d in (left, right)]
    largest = max(largest, *(archive_array_extent(f) for f in array_paths))
    bound = 8 * largest * 6
    if bound > 1024**3:
        raise MemoryError("complete comparison arrays exceed declared1GiB host cap")
    gate, raw_gate = 1e-11, 1e-10
    files = [
        d / n
        for d in (left, right)
        for n in ("contract.json", "source_sha256.json", "resource.json")
    ]
    rows = []
    for key in KEYS:
        label = f"{key[0]}_{key[1]}"
        records = [
            json.loads((d / f"valence_{label}.json").read_text()) for d in (left, right)
        ]
        for name in ("column", "energy_Ha", "row_keys", "channels"):
            if records[0][name] != records[1][name]:
                raise ValueError("complete column/energy/coordinate identity differs")
        for name in PHYSICAL:
            if records[0]["provenance"][name] != records[1]["provenance"][name]:
                raise ValueError(f"complete physical identity differs: {name}")
        for record in records:
            if (
                record["provenance"]["solver"]
                != "matrix-free GMRES and actual homogeneous-operator refinement"
                or not np.isfinite(record["raw_residual_Ha"])
                or not np.isfinite(record["response_residual_Ha"])
                or max(record["raw_residual_Ha"], record["response_residual_Ha"])
                > raw_gate
            ):
                raise ValueError(
                    "fresh actual public root/response certification required"
                )
        arrays = [d / f"valence_{label}.npz" for d in (left, right)]
        before = [digest(f) for f in arrays]
        errors = {}
        with ExitStack() as stack:
            data = [stack.enter_context(np.load(f, allow_pickle=False)) for f in arrays]
            names = {
                "numerators",
                "denominators",
                "sigma1",
                "numerator_derivative",
                "sigma1_derivative",
            }
            if any(set(a.files) != names for a in data):
                raise ValueError("every complete signed root/response array required")
            for name in sorted(names):
                a, b = data[0][name], data[1][name]
                if (
                    a.dtype != np.float64
                    or b.dtype != a.dtype
                    or a.shape != b.shape
                    or not all(np.all(np.isfinite(v)) for v in (a, b))
                ):
                    raise ValueError("complete real finite arrays required")
                errors[name] = float(np.max(abs(a - b)))
            if not np.array_equal(data[0]["denominators"], data[1]["denominators"]):
                raise ValueError("complete column denominators differ")
        if before != [digest(f) for f in arrays] or max(errors.values()) > gate:
            raise AssertionError("complete immutable signed numerical gate failed")
        rows.append(
            dict(
                column=key,
                energy_Ha=records[0]["energy_Ha"],
                complete_signed_errors_Ha=errors,
                raw_root_Ha=[r["raw_residual_Ha"] for r in records],
                raw_prime_Ha=[r["response_residual_Ha"] for r in records],
            )
        )
        files += arrays + [d / f"valence_{label}.json" for d in (left, right)]
    screening_errors = None
    if screening:
        results = [json.loads((d / "result.json").read_text()) for d in (left, right)]
        if any(
            r["status"]
            != "full fixed-energy columns passed, directed screening computed"
            for r in results
        ):
            raise ValueError("complete directed production result required")
        records = [json.loads((d / "sigma2.json").read_text()) for d in (left, right)]
        for name in ("channels", "virtual_keys", "valence_keys", "groups"):
            if records[0][name] != records[1][name]:
                raise ValueError("complete directed screening coordinates differ")
        screening_errors = {}
        screening_paths = [d / "sigma2.npz" for d in (left, right)]
        before = [digest(f) for f in screening_paths]
        with ExitStack() as stack:
            data = [
                stack.enter_context(np.load(d / "sigma2.npz", allow_pickle=False))
                for d in (left, right)
            ]
            names = {"radial_Ha", "joint_derivative", "energy_parameters_Ha"}
            if any(set(a.files) != names for a in data):
                raise ValueError("complete screening/joint response arrays required")
            for name in sorted(names):
                a, b = data[0][name], data[1][name]
                if (
                    a.dtype != np.float64
                    or b.dtype != a.dtype
                    or a.shape != b.shape
                    or not all(np.all(np.isfinite(v)) for v in (a, b))
                ):
                    raise ValueError("complete real finite directed arrays required")
                screening_errors[name] = float(np.max(abs(a - b)))
            if not np.array_equal(
                data[0]["energy_parameters_Ha"], data[1]["energy_parameters_Ha"]
            ):
                raise ValueError("screening column parameters differ")
        if (
            before != [digest(f) for f in screening_paths]
            or max(screening_errors.values()) > gate
        ):
            raise AssertionError("complete directed screening signed gate failed")
        files += [
            d / n
            for d in (left, right)
            for n in ("sigma2.json", "sigma2.npz", "result.json")
        ]
    result = dict(
        status="pass",
        scope="complete matched numerical operators after fresh public root/response certification; no physical spectrum, convergence, controlled timing or experimental claim",
        left=left.name,
        right=right.name,
        gate_Ha=gate,
        raw_gate_Ha=raw_gate,
        host_cap_bytes=1024**3,
        complete_array_payload_bound_bytes=bound,
        columns=rows,
        screening_signed_errors_Ha=screening_errors,
        input_sha256={str(f.relative_to(OWNER)): digest(f) for f in files},
        comparator_source_sha256=digest(__file__),
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                status="pass",
                columns=len(rows),
                max_column_error_Ha=max(
                    max(r["complete_signed_errors_Ha"].values()) for r in rows
                ),
                screening_errors_Ha=screening_errors,
            )
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--screening", action="store_true")
    args = parser.parse_args()
    compare(args.left, args.right, args.output, screening=args.screening)
