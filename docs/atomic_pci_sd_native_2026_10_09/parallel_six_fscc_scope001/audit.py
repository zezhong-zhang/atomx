"""Bounded determinant projection of complete signed, actual CCSDT tensors."""

import hashlib
import importlib.util
import json
import math
import os
import tarfile
from pathlib import Path

import numpy as np
from scipy.linalg import expm

REPO = Path(__file__).resolve().parents[3]
OWNER = REPO / "docs/atomic_pci_sd_native_2026_10_09"
BACKEND = (
    REPO
    / "docs/atomx_hourly_progress_2026_10_10/parallel001/public_backend_source-build/expt.x"
)
SOURCE = BACKEND.parent.parent / "public_backend_source"
CASE = OWNER / "transverse_precision007/tight"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def helpers():
    reader = load(OWNER / "parallel_fscc012/probe.py", "signed_fscc_reader")
    return reader, reader.load_oracle()


def source_archive(output, extra=()):
    paths = [
        *REPO.joinpath("src/atomx").rglob("*.py"),
        *extra,
        Path(__file__),
        BACKEND,
        BACKEND.parent / "atomx_backend.json",
        OWNER / "parallel_fscc012/probe.py",
        OWNER / "probe_bloch_oracle.py",
        REPO / "tests/test_transverse_spinor_integrals.py",
        REPO / "tests/test_spinor_integrals_fscc.py",
    ]
    paths.extend(SOURCE.joinpath("src/rcc").rglob("*.c"))
    paths.extend(SOURCE.joinpath("src/rcc").rglob("*.h"))
    paths = sorted(set(p.resolve() for p in paths if p.is_file()))
    binding = {str(p.relative_to(REPO)): sha(p) for p in paths}
    with tarfile.open(output / "source_inputs.tar.gz", "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(REPO)))
    return binding


def audit_case(case):
    from atomx.methods.fscc import load_integrals

    reader, oracle = helpers()
    a = load_integrals(case)
    n, o = len(a.labels), a.noccupied
    if n > 12 or math.comb(n, o + 1) > 1000:
        raise ValueError("declared finite complete determinant cap exceeded")
    mc, hc = oracle.determinant_matrix(a, o)
    ma, ha = oracle.determinant_matrix(a, o + 1)
    tensors, decoder = {}, {}
    for name in ("t1c", "t2c", "t3c", "s1c", "s2c", "s3c"):
        tensors[name], decoder[name] = reader.decode_diagram(
            case / "scratch" / (name + ".dg"), n
        )
    terms = reader.cluster_terms([tensors[k] for k in ("t1c", "t2c", "t3c")])
    tc = oracle.excitation_matrix(mc, terms)
    ta = oracle.excitation_matrix(ma, terms)
    ec, ea = expm(tc), expm(ta)
    hbc, hba = expm(-tc) @ hc @ ec, expm(-ta) @ ha @ ea
    ref = (1 << o) - 1
    ri = mc.index(ref)
    energy = hbc[ri, ri]
    core_residual = hbc[:, ri].copy()
    core_residual[ri] -= energy
    sa = oracle.excitation_matrix(
        ma,
        reader.cluster_terms(
            [tensors[k] for k in ("s1c", "s2c", "s3c")], attachment=True
        ),
    )
    pi = np.array([ma.index(ref | (1 << v)) for v in range(o, n)])
    omega = (np.eye(len(ma)) + sa)[:, pi]
    matrix = np.load(case / "model_space_matrix.npz")["matrix_Ha"]
    total = matrix + energy * np.eye(len(pi))
    residual = hba @ omega - omega @ total
    core_ranks = np.array([o - (mask & ref).bit_count() for mask in mc])
    attachment_ranks = np.array([o - (mask & ref).bit_count() for mask in ma])
    core_retained = (core_ranks > 0) & (core_ranks <= 3)
    attachment_retained = attachment_ranks <= 2
    wave = ea @ omega
    metric = wave.conj().T @ wave
    roots_c, roots_a = np.linalg.eigvalsh(hc), np.linalg.eigvalsh(ha)
    projected = (hba @ omega)[pi] - energy * np.eye(len(pi))
    metrics = dict(
        nspinors=n,
        noccupied=o,
        core_determinants=len(mc),
        attachment_determinants=len(ma),
        model_dimension=len(pi),
        complete_signed_columns=len(pi),
        decoder_metadata=decoder,
        core_energy_Ha=float(energy.real),
        core_projected_rank123_max_Ha=float(np.max(abs(core_residual[core_retained]))),
        core_unretained_rank4_max_Ha=float(
            np.max(abs(core_residual[core_ranks > 3]), initial=0)
        ),
        attachment_projected_rank012_max_Ha=float(
            np.max(abs(residual[attachment_retained]))
        ),
        attachment_unretained_rank34_max_Ha=float(
            np.max(abs(residual[attachment_ranks > 2]), initial=0)
        ),
        full_attachment_bloch_max_Ha=float(np.max(abs(residual))),
        heff_signed_projection_max_Ha=float(np.max(abs(matrix - projected))),
        intermediate_normalization_max=float(np.max(abs(omega[pi] - np.eye(len(pi))))),
        physical_metric_min_eigenvalue=float(np.linalg.eigvalsh(metric)[0]),
        core_fci_method_error_Ha=float(abs(energy - roots_c[0])),
        attachment_fci_method_error_max_Ha=float(
            np.max(abs(np.sort(np.linalg.eigvals(total).real) - roots_a[: len(pi)]))
        ),
        attachment_retained_rows=int(np.count_nonzero(attachment_retained)),
        core_retained_rows=int(np.count_nonzero(core_retained)),
        fock_occupied_virtual_max_Ha=float(np.max(abs(a.fock_Ha[:o, o:]))),
        truncation_contract="Core T1/T2/T3 and attachment S1/S2/S3; connected T4/S4 are absent. Full FCI difference is reported as method error and never relaxed into the projected equation gate.",
    )
    arrays = dict(
        core_masks=mc,
        attachment_masks=ma,
        model_indices=pi,
        full_core_hamiltonian_Ha=hc,
        full_attachment_hamiltonian_Ha=ha,
        core_cluster=tc,
        attachment_cluster=ta,
        core_hbar_Ha=hbc,
        attachment_hbar_Ha=hba,
        omega=omega,
        wave_operator=wave,
        physical_metric=metric,
        backend_heff_Ha=matrix,
        total_heff_Ha=total,
        core_residual_Ha=core_residual,
        attachment_residual_Ha=residual,
        core_ranks=core_ranks,
        attachment_ranks=attachment_ranks,
        exact_core_roots_Ha=roots_c,
        exact_attachment_roots_Ha=roots_a,
        **tensors,
    )
    return metrics, arrays


def solve(a, output):
    from atomx.methods.fscc import compute

    environment = os.environ.copy()
    libraries = (
        REPO.parent
        / "atomx-references/2026-10-09-ci-fscc/toolchain/usr/lib/x86_64-linux-gnu"
    )
    environment["LD_LIBRARY_PATH"] = ":".join(
        (
            str(libraries),
            str(libraries / "openblas-serial"),
            environment.get("LD_LIBRARY_PATH", ""),
        )
    )
    return compute(
        a,
        executable=BACKEND,
        directory=output,
        model="CCSDT",
        sector="0h1p",
        active_indices=tuple(range(a.noccupied, len(a.labels))),
        tolerance=1e-13,
        residual_bound_tolerance_Ha=1e-11,
        memory_bytes=512 * 1024**2,
        tile_size=4,
        threads=1,
        timeout_seconds=120,
        environment=environment,
        tensor_symmetry="C1",
    )


def dump(path, data):
    path.write_text(json.dumps(data, indent=2))


def archive_scientific_inputs(output):
    paths = [p for p in (output / "fscc").rglob("*") if p.is_file()]
    paths.extend(
        p for p in output.iterdir() if p.is_file() and p.suffix in (".npz", ".json")
    )
    with tarfile.open(output / "scientific_inputs.tar.gz", "w:gz") as archive:
        for path in sorted(paths):
            archive.add(path, arcname=str(path.relative_to(output)))
    return sha(output / "scientific_inputs.tar.gz")
