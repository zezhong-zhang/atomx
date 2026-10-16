"""Source-bound complete EXP-T amplitude/Bloch audit in a finite 10-spinor space.

This reader implements only the pinned little-endian, compression-off ATXDG002
format. It rejects other formats, external block files and unexpected extents.
It preserves complex values and reconstructs every permutation-related block.
The determinant lift is a diagnostic oracle, not a scalable FSCC solver.
"""

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import os
import resource
import struct
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm

REPO = Path(__file__).resolve().parents[3]
OWNER = REPO / "docs/atomic_pci_sd_native_2026_10_09"
REFERENCE = OWNER / "expt_compact_public_source001/src/rcc"
DEFAULT_CASE = OWNER / "transverse_precision007/tight"
BACKEND = OWNER / "expt_compact_public_source001-build/expt.x"
BACKEND_SHA = "1605f4d9daf0adb5ecc543fc785da4cd5e32d96121260dc00e9951bdbdd0d33f"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Reader:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def raw(self, count):
        if count < 0 or self.pos + count > len(self.data):
            raise ValueError("truncated diagram extent")
        value = self.data[self.pos : self.pos + count]
        self.pos += count
        return value

    def number(self, fmt):
        return struct.unpack("<" + fmt, self.raw(struct.calcsize("<" + fmt)))[0]

    def ints(self, count):
        return np.frombuffer(self.raw(4 * count), dtype="<i4").copy()


def decode_diagram(path, nspinors=10):
    """Decode complete bounded tensors using pinned diagram_get permutations."""
    if not 1 <= nspinors <= 12 or path.stat().st_size > 8 * 1024**2:
        raise ValueError("finite audit diagram exceeds declared bounds")
    reader = Reader(path.read_bytes())
    name = reader.raw(64).split(b"\0", 1)[0].decode("ascii")
    rank, symmetry, only_unique = [reader.number("i") for _ in range(3)]
    if rank not in (2, 4, 6):
        raise ValueError("unsupported tensor rank")
    qparts, valence, t3space, order = [reader.ints(rank) for _ in range(4)]
    if reader.raw(8) != b"ATXDG002":
        raise ValueError("requires explicit uncompressed ATXDG002")
    count = reader.number("Q")
    if count > 4096:
        raise ValueError("too many finite diagram blocks")
    blocks = []
    for _ in range(count):
        block_id, block_rank, arithmetic = (
            reader.number("q"), reader.number("i"), reader.number("i")
        )
        unique, sign, equal = [reader.number("i") for _ in range(3)]
        from_unique, to_unique, block_numbers = [reader.ints(16) for _ in range(3)]
        if block_rank != rank or arithmetic not in (0, 1) or unique not in (0, 1):
            raise ValueError("invalid block metadata")
        indices = []
        for _ in range(rank):
            extent = reader.number("i")
            if not 1 <= extent <= nspinors:
                raise ValueError("invalid finite axis extent")
            axis = reader.ints(extent)
            if len(set(axis)) != len(axis) or np.any(axis < 0) or np.any(axis >= nspinors):
                raise ValueError("invalid spinor identity")
            indices.append(axis)
        size, storage = reader.number("Q"), reader.number("i")
        shape = tuple(map(len, indices))
        if size != math.prod(shape):
            raise ValueError("block size does not match complete axes")
        if storage == 0:
            dtype = "<f8" if arithmetic == 0 else "<c16"
            data = np.frombuffer(reader.raw(size * np.dtype(dtype).itemsize), dtype=dtype)
            data = data.reshape(shape).astype(complex)
            if not np.all(np.isfinite(data)):
                raise ValueError("nonfinite amplitude")
        elif storage == 2 and not unique:
            data = None
        else:
            raise ValueError("external/compressed block storage is unsupported")
        blocks.append(dict(id=block_id, unique=unique, sign=sign, equal=equal,
                           from_unique=from_unique[:rank], to_unique=to_unique[:rank],
                           key=tuple(map(int, block_numbers[:rank])), indices=indices,
                           data=data))
    if reader.pos != len(reader.data):
        raise ValueError("unexpected trailing diagram bytes")
    index = {block["key"]: block for block in blocks}
    if len(index) != len(blocks):
        raise ValueError("duplicate block identity")
    result = np.zeros((nspinors,) * rank, complex)
    present = np.zeros((nspinors,) * rank, bool)
    for block in blocks:
        values = block["data"]
        if not block["unique"]:
            perm = block["to_unique"]
            if set(perm) != set(range(rank)):
                raise ValueError("invalid permutation metadata")
            key = tuple(block["key"][i] for i in perm)
            base = index.get(key)
            if base is None or not base["unique"] or base["data"] is None:
                raise ValueError("missing unique permutation parent")
            if any(not np.array_equal(base["indices"][j], block["indices"][i])
                   for j, i in enumerate(perm)):
                raise ValueError("permutation changed spinor identity")
            values = block["sign"] * base["data"].transpose(np.argsort(perm))
        ix = np.ix_(*block["indices"])
        if np.any(present[ix]):
            raise ValueError("overlapping diagram blocks")
        result[ix], present[ix] = values, True
    return result, dict(name=name, rank=rank, symmetry=symmetry,
                        only_unique=only_unique, qparts=qparts.tolist(),
                        valence=valence.tolist(), t3space=t3space.tolist(),
                        order=order.tolist(), blocks=int(count),
                        coordinates=int(np.count_nonzero(present)),
                        nonzero=int(np.count_nonzero(result)), sha256=sha(path))


def load_oracle():
    spec = importlib.util.spec_from_file_location("fscc012_oracle", OWNER / "probe_bloch_oracle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cluster_terms(tensors, attachment=False):
    """EXP-T hole/valence inputs precede particle outputs in normal order."""
    terms = []
    for level, tensor in enumerate(tensors, 1):
        factor = 1 / (math.factorial(level) * math.factorial(level - int(attachment)))
        for coordinate in zip(*np.nonzero(tensor)):
            incoming, outgoing = coordinate[:level], coordinate[level:]
            operations = tuple((int(p), True) for p in outgoing)
            operations += tuple((int(i), False) for i in reversed(incoming))
            terms.append((operations, factor * tensor[coordinate]))
    return terms


def audit(case):
    oracle = load_oracle()
    integrals = oracle.load_integrals(case)
    n = len(integrals.labels)
    if n != 10 or integrals.noccupied != 2:
        raise ValueError("fixed audit requires complete ten-spinor two-core case")
    tensors, metadata = {}, {}
    for name in ("t1c", "t2c", "t3c", "s1c", "s2c", "s3c", "hp", "ph", "pphh"):
        tensors[name], metadata[name] = decode_diagram(case / "scratch" / (name + ".dg"), n)
    fock_decode = max(float(np.max(abs(tensors["hp"][:2, 2:] - integrals.fock_Ha[:2, 2:]))),
                      float(np.max(abs(tensors["ph"][2:, :2] - integrals.fock_Ha[2:, :2]))))
    masks2, h2 = oracle.determinant_matrix(integrals, 2)
    masks3, h3 = oracle.determinant_matrix(integrals, 3)
    core_terms = cluster_terms([tensors["t1c"], tensors["t2c"], tensors["t3c"]])
    t2 = oracle.excitation_matrix(masks2, core_terms)
    t3 = oracle.excitation_matrix(masks3, core_terms)
    transformed2 = expm(-t2) @ h2 @ expm(t2)
    reference = masks2.index(3)
    core_energy = transformed2[reference, reference]
    core_residual = transformed2[:, reference].copy()
    core_residual[reference] -= core_energy
    exact_core, exact_arrays = oracle.diagnose(integrals)
    core_cluster_error = float(np.max(abs(t2 - exact_arrays["core_cluster_2"])))
    hbar = expm(-t3) @ h3 @ expm(t3)
    s = oracle.excitation_matrix(masks3, cluster_terms(
        [tensors["s1c"], tensors["s2c"], tensors["s3c"]], attachment=True))
    model = [masks3.index(3 | (1 << v)) for v in range(2, n)]
    omega = (np.eye(len(masks3)) + s)[:, model]
    target = np.load(case / "model_space_matrix.npz", allow_pickle=False)["matrix_Ha"]
    heff = (hbar @ omega)[model] - core_energy * np.eye(len(model))
    residual = hbar @ omega - omega @ (target + core_energy * np.eye(len(model)))
    rank_rows = [[i for i, mask in enumerate(masks3)
                  if 2 - (mask & 3).bit_count() == rank] for rank in range(3)]
    fov_terms = [(((i, True), (v, False)), integrals.fock_Ha[i, v])
                 for i in range(2) for v in range(2, n)]
    f_deexcitation = oracle.excitation_matrix(masks3, fov_terms)
    t2_only = oracle.excitation_matrix(masks3, cluster_terms([np.zeros_like(tensors["t1c"]), tensors["t2c"]]))
    fov_t2_commutator = (f_deexcitation @ t2_only - t2_only @ f_deexcitation)[:, model]
    fov_t2_open_tensor = np.einsum("jv,jiab->viab", integrals.fock_Ha, tensors["t2c"])
    fov_t2_open_tensor[:2] = 0
    fov_t2_open = oracle.excitation_matrix(masks3, cluster_terms(
        [np.zeros_like(tensors["s1c"]), fov_t2_open_tensor], attachment=True))[:, model]
    metrics = dict(
        status="unclosed_equation_audit", highest_level_reached=False,
        fock_decode_max_Ha=fock_decode,
        core_cluster_max=core_cluster_error,
        core_projected_residual_max_Ha=float(np.max(abs(core_residual))),
        core_energy_Ha=float(core_energy.real),
        intermediate_normalization_max=float(np.max(abs(omega[model] - np.eye(len(model))))),
        backend_wave_bloch_residual_max_Ha=float(np.max(abs(residual))),
        backend_wave_bloch_by_core_holes_Ha=[float(np.max(abs(residual[rows]))) for rows in rank_rows],
        backend_heff_minus_wave_projection_max_Ha=float(np.max(abs(target - heff))),
        wave_minus_exact_max=float(np.max(abs(omega - exact_arrays["omega"]))),
        backend_strict_heff_max_Ha=float(np.max(abs(target - exact_arrays["exact_heff_Ha"]))),
        q1_residual_minus_Fov_T2_commutator_Ha=float(np.max(abs(
            (residual - fov_t2_commutator)[rank_rows[1]]))),
        q1_residual_plus_Fov_T2_commutator_Ha=float(np.max(abs(
            (residual + fov_t2_commutator)[rank_rows[1]]))),
        Fov_T2_commutator_max_Ha=float(np.max(abs(fov_t2_commutator))),
        residual_minus_open_Fov_T2_max_Ha=float(np.max(abs(residual - fov_t2_open))),
        residual_plus_open_Fov_T2_max_Ha=float(np.max(abs(residual + fov_t2_open))),
        open_Fov_T2_max_Ha=float(np.max(abs(fov_t2_open))),
        gates=dict(oracle_Ha=1e-11, strict_backend_Ha=1e-10, normalization=1e-12),
        decoder_metadata=metadata,
        scope="Complete finite source-bound actual signed amplitude audit, no production repair or scalability claim",
    )
    metrics["decoder_and_core_gate_pass"] = bool(fock_decode < 1e-14 and core_cluster_error < 1e-12
                                                and metrics["core_projected_residual_max_Ha"] < 1e-11)
    metrics["strict_backend_gate_pass"] = metrics["backend_strict_heff_max_Ha"] <= 1e-10
    arrays = dict(backend_tcore2=t2, backend_tcore3=t3, backend_omega=omega,
                  backend_hbar3_Ha=hbar, backend_heff_Ha=target, wave_heff_Ha=heff,
                  backend_bloch_residual_Ha=residual,
                  exact_omega=exact_arrays["omega"], exact_heff_Ha=exact_arrays["exact_heff_Ha"],
                  full_h2_Ha=h2, full_h3_Ha=h3, model_indices=model,
                  Fov_T2_commutator=fov_t2_commutator,
                  open_Fov_T2=fov_t2_open,
                  **{name: tensor for name, tensor in tensors.items() if name.startswith(("t", "s"))})
    return metrics, arrays


def source_inputs(case):
    files = [Path(__file__), OWNER / "probe_bloch_oracle.py",
             REPO / "src/atomx/methods/fscc.py",
             REPO / "src/atomx/physics/csf/spinor_integrals.py",
             REPO / "tests/test_transverse_spinor_integrals.py", BACKEND]
    files += [REFERENCE / part for part in (
        "engine/diagram.c", "engine/diagram.h", "engine/block.c", "engine/block.h",
        "io/io.c", "include/comdef.h", "include/options.h", "models/sector01.c",
        "models/sector01_ccsdt.c")]
    files += [p for p in case.iterdir() if p.is_file()]
    files += list((case / "scratch").glob("*.dg"))
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = source_inputs(args.case)
    binding = {str(p.relative_to(REPO)): sha(p) for p in files}
    if sha(BACKEND) != BACKEND_SHA:
        raise ValueError("pinned executable identity changed")
    contract = dict(
        hypothesis="Actual signed backend core and attachment amplitudes distinguish a projection/energy formula defect from missing open-sector equation terms; do not infer a specific diagram from eigenvalues.",
        fixed="Unchanged saved ten-spinor two-core/eight-active Hamiltonian, full 120 determinant rows and eight model columns, same native orthonormal metric, compression-off ATXDG002 signed amplitudes.",
        distinguishing="Core reconstruction and complete Fock decode must pass before interpreting P/Q1/Q2 Bloch residuals. A localized residual identifies affected operator coordinates, not yet a named missing diagram.",
        gates=dict(oracle_Ha=1e-11, strict_backend_Ha=1e-10, normalization=1e-12),
        stop="One complete amplitude audit; no gate relaxation, Fov deletion, fitted correction, backend edits or radial scans. Single BLAS thread, 4 GiB address cap, 120 seconds.",
        source_and_inputs=binding, load=os.getloadavg(),
        available_bytes=os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"),
    )
    (args.output / "contract.json").write_text(json.dumps(contract, indent=2))
    with tarfile.open(args.output / "source_inputs.tar.gz", "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(REPO))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    start = time.perf_counter()
    try:
        metrics, arrays = audit(args.case)
        assert all(sha(REPO / p) == value for p, value in binding.items())
        np.savez_compressed(args.output / "full_arrays.npz", **arrays)
        metrics.update(source_and_inputs=binding, inputs_unchanged=True,
                       arrays_sha256=sha(args.output / "full_arrays.npz"),
                       archive_sha256=sha(args.output / "source_inputs.tar.gz"),
                       observed_phase_wall_seconds=time.perf_counter() - start,
                       peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        (args.output / "result.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps({k: v for k, v in metrics.items() if k not in ("decoder_metadata", "source_and_inputs")}))
    except Exception as error:
        (args.output / "failure.json").write_text(json.dumps(
            dict(status="failed_audit", error=type(error).__name__, message=str(error))))
        raise


if __name__ == "__main__":
    main()
