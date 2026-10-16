"""Matched frozen-H actual LCCSD/CCSD/CCSDT/FCI discriminator; private evidence."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import resource
import shutil
import signal
import tarfile
import time
from pathlib import Path

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np  # noqa: E402
from scipy.linalg import expm  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OWNER = Path(__file__).resolve().parent
RUN = OWNER / "run001"
OLD = REPO / "docs/atomic_pci_sd_native_2026_10_09"
INPUT = REPO / "docs/atomic_external_benchmark_2026_10_11/fscc/run003/input/four_core"
BACKEND = REPO / "docs/atomx_hourly_progress_2026_10_10/parallel001/public_backend_source-build/expt.x"
BACKEND_SOURCE = BACKEND.parent.parent / "public_backend_source"
LOCK = REPO / "docs/atomic_external_benchmark_2026_10_11/timing.lock"
CAP = 4 * 1024**3
GATE = 1e-11


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def machine():
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return dict(load=list(os.getloadavg()), affinity=sorted(os.sched_getaffinity(0)),
                available_bytes=int(memory["MemAvailable"].split()[0]) * 1024,
                free_disk_bytes=shutil.disk_usage(OWNER).free)


def helpers():
    reader = load(OLD / "parallel_fscc012/probe.py", "method_rank_decoder")
    return reader, reader.load_oracle()


def unchanged():
    contract = json.loads((RUN / "contract.json").read_text())
    return {path: sha(REPO / path) == digest for path, digest in contract["source_and_inputs"].items()}


def prepare():
    RUN.mkdir(exist_ok=False)
    files = [Path(__file__), *REPO.joinpath("src/atomx").rglob("*.py"), BACKEND,
             BACKEND.parent / "atomx_backend.json", OLD / "parallel_fscc012/probe.py",
             OLD / "probe_bloch_oracle.py", OLD / "parallel_six_fscc_scope001/native.py",
             REPO / "tests/test_transverse_spinor_integrals.py",
             REPO / "tests/test_spinor_integrals_fscc.py",
             *(INPUT / name for name in ("ATOMX_ONE", "ATOMX_TWO", "integrals.json"))]
    files.extend(BACKEND_SOURCE.joinpath("src").rglob("*.c"))
    files.extend(BACKEND_SOURCE.joinpath("src").rglob("*.h"))
    files = sorted(set(path.resolve() for path in files))
    binding = {str(path.relative_to(REPO)): sha(path) for path in files}
    before = machine()
    contract = dict(
        hypothesis="At fixed complete signed one-/two-body H and orbital space, actual method-rank changes distinguish nonlinear SD and connected triples from orbital-cutoff error. Finite residual correctness is separate from finite FCI method error.",
        fixed=dict(input=str(INPUT.relative_to(REPO)), Z=4, nucleus="point; original native producer default; nuclear binding preserved in frozen integral identity",
                   occupied="1s2 2s2, four electrons; original two-core VN orbitals without reoptimization",
                   virtual="3s1/2 doublet, 2p1/2 doublet, 2p3/2 quartet: all eight magnetic spinors",
                   nspinors=12, core_determinants=495, attachment_determinants=792,
                   hamiltonian="Dirac-Coulomb + real symmetric-two-vertex orbital-difference finite-frequency transverse exchange; signed ERI/Fock; no scalar Breit/QED addition",
                   reference_mask=15, frozen_energy_Ha=0, units="Hartree", correlation_space_changes=0,
                   nucleus_orbital_frequency_changes=0),
        methods=[
            dict(name="LCCSD", entry="atomx.methods.linearized_sd.solve", scope="Actual linked finite H+[H,T1+T2] solve on all 200 orthonormal magnetic SD determinants. Dense public correctness anchor, not native pCI CoreSD HPC. No attachment LCCSD method is substituted."),
            dict(name="CCSD", entry="atomx.methods.fscc.compute", model="CCSD", sector="0h1p", scope="Actual native nonlinear T1/T2 and S1/S2 equations, complete eight-column model. Triples diagrams are not decoded or fed as a surrogate CCSD solve."),
            dict(name="CCSDT", entry="atomx.methods.fscc.compute", model="CCSDT", sector="0h1p", scope="Actual source-bound repaired T1/T2/T3 and S1/S2/S3 equations; connected higher ranks absent."),
            dict(name="FCI", entry="existing independent fermionic determinant_matrix + numpy.linalg.eigh", scope="Complete fixed-4/5-electron determinant H only, same frozen signed integrals; finite oracle, not experimental truth.")],
        gates=dict(core_retained_equation_max_Ha=GATE, attachment_retained_equation_max_Ha=GATE,
                   signed_HEFF_projection_max_Ha=1e-10, intermediate_normalization_max=1e-12,
                   native_vs_independent_core_energy_max_Ha=1e-10,
                   FCI_full_eigen_residual_max_Ha=GATE,
                   model_projector_condition_max=1e6, minimum_core_FCI_overlap=0.5,
                   root_group_energy_tolerance_Ha=1e-9, minimum_attachment_group_principal_overlap=0.5,
                   FCI_method_error="report signed and absolute, never required zero for truncated methods"),
        matching="Closed core: lowest complete FCI state, accepted only with reference and solved-wave overlap >=0.5. Attachment: compare the complete lowest eight FCI states only when the full original eight-index projector is nonsingular; verify complete degenerate energy groups and their physical projectors. Never splice individual roots between runs.",
        resources=dict(address_cap_bytes=CAP, backend_memory_bytes=512 * 1024**2,
                       threads=1, timeout_seconds_per_stage=120,
                       shared_timing_lock=str(LOCK.relative_to(REPO)),
                       largest_complete_matrix_bytes=16 * 792**2,
                       largest_rank6_decode_bytes=16 * 12**6,
                       conservative_working_payload_bytes=1024**3,
                       performance_claim=False),
        stop="One predeclared FCI, LCCSD, CCSD and CCSDT stage each. Preserve any failure; no changed H, thresholds, source, rank, orbital scope or root selection to recover passing results.",
        source_and_inputs=binding, resource_preflight=before,
    )
    dump(RUN / "contract.json", contract)
    (RUN / "fixed_inputs").mkdir()
    for name in ("ATOMX_ONE", "ATOMX_TWO", "integrals.json"):
        shutil.copy2(INPUT / name, RUN / "fixed_inputs" / name)
    with tarfile.open(RUN / "source_inputs.tar.gz", "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(REPO)))
    dump(RUN / "preparation.json", dict(contract_sha256=sha(RUN / "contract.json"),
         source_archive_sha256=sha(RUN / "source_inputs.tar.gz"), source_and_inputs_unchanged=unchanged()))
    print("prospective fixed-H contract written", flush=True)


def fci():
    from atomx.methods.fscc import load_integrals
    _, oracle = helpers()
    a = load_integrals(RUN / "fixed_inputs")
    mc, hc = oracle.determinant_matrix(a, 4)
    ma, ha = oracle.determinant_matrix(a, 5)
    if not np.array_equal(hc, hc.conj().T) or not np.array_equal(ha, ha.conj().T):
        raise ValueError("original frozen full signed H must be exactly Hermitian; no projection permitted")
    ec, vc = np.linalg.eigh(hc)
    ea, va = np.linalg.eigh(ha)
    ref = mc.index(15)
    pi = np.array([ma.index(15 | (1 << index)) for index in range(4, 12)])
    rc = float(np.max(abs(hc @ vc - vc * ec)))
    ra = float(np.max(abs(ha @ va - va * ea)))
    reference_overlap = float(abs(vc[ref, 0]))
    condition = float(np.linalg.cond(va[pi, :8]))
    arrays = dict(core_masks=mc, attachment_masks=ma, full_core_hamiltonian_Ha=hc,
                  full_attachment_hamiltonian_Ha=ha, exact_core_roots_Ha=ec,
                  exact_attachment_roots_Ha=ea, exact_core_vectors=vc,
                  exact_attachment_vectors=va, model_indices=pi,
                  labels=np.asarray(a.labels), one_body_Ha=a.one_body_Ha,
                  fock_Ha=a.fock_Ha, packed_eri_Ha=a.packed_eri_Ha)
    np.savez_compressed(RUN / "fci_arrays.npz", **arrays)
    return dict(method="FCI", core_energy_Ha=float(ec[0]), core_full_eigen_residual_max_Ha=rc,
                attachment_full_eigen_residual_max_Ha=ra, core_reference_overlap=reference_overlap,
                attachment_lowest8_model_projector_condition=condition,
                attachment_lowest8_energies_Ha=ea[:8].tolist(),
                fock_occupied_virtual_max_Ha=float(np.max(abs(a.fock_Ha[:4, 4:]))),
                frozen_integral_identity=dict(a.identity),
                gates=dict(full_eigen_equations=max(rc, ra) <= GATE,
                           reference_branch=reference_overlap >= .5,
                           complete_model_projection=condition < 1e6),
                arrays_sha256=sha(RUN / "fci_arrays.npz"))


def lccsd():
    from atomx.methods.linearized_sd import solve
    _, oracle = helpers()
    saved = np.load(RUN / "fci_arrays.npz")
    masks = saved["core_masks"].tolist()
    h = saved["full_core_hamiltonian_Ha"]
    if np.any(h.imag != 0):
        raise ValueError("public finite LCCSD is real-only; original complex H cannot be changed")
    ref_index = masks.index(15)
    ranks = np.array([4 - (mask & 15).bit_count() for mask in masks])
    selected = np.concatenate(([ref_index], np.flatnonzero((ranks > 0) & (ranks <= 2))))
    vectors = [((masks[index], 1.),) for index in selected]
    solved = solve(h.real[np.ix_(selected, selected)], vectors, 15,
                   reference_identity="same immutable four-core 12-spinor signed-H reference; " + sha(RUN / "fixed_inputs/ATOMX_ONE") + ":" + sha(RUN / "fixed_inputs/ATOMX_TWO"),
                   residual_tolerance_Ha=GATE, memory_bytes=512 * 1024**2)
    operators = []
    for mask, amplitude in zip((masks[index] for index in selected[1:]), solved.amplitudes, strict=True):
        holes = [i for i in range(4) if not mask & (1 << i)]
        particles = [i for i in range(4, 12) if mask & (1 << i)]
        ops = tuple((p, True) for p in particles) + tuple((i, False) for i in reversed(holes))
        target, sign = oracle.apply(15, ops)
        if target != mask or not sign:
            raise ValueError("independent excitation identity/sign failed")
        operators.append((ops, amplitude / sign))
    t = oracle.excitation_matrix(masks, operators)
    linear = h + h @ t - t @ h
    energy = linear[ref_index, ref_index]
    residual = linear[:, ref_index].copy()
    residual[ref_index] -= energy
    trial = (np.eye(len(masks)) + t)[:, ref_index]
    overlap = float(abs(np.vdot(saved["exact_core_vectors"][:, 0], trial)) / np.linalg.norm(trial))
    trial_residual = h @ trial - energy * trial
    retained = float(np.max(abs(residual[(ranks > 0) & (ranks <= 2)])))
    delta = float(energy.real - saved["exact_core_roots_Ha"][0])
    np.savez_compressed(RUN / "lccsd_arrays.npz", core_masks=masks, core_ranks=ranks,
                        sd_indices=selected, amplitudes=solved.amplitudes,
                        native_equation_matrix_Ha=solved.equation_matrix_Ha,
                        singles_product_subtraction_Ha=solved.singles_product_subtraction_Ha,
                        core_cluster=t, full_linearized_H_Ha=linear,
                        core_residual_Ha=residual, linearized_trial_wave=trial,
                        trial_schrodinger_residual_Ha=trial_residual)
    return dict(method="LCCSD", core_energy_Ha=float(energy.real), core_energy_native_Ha=solved.total_energy_Ha,
                native_equation_residual_Ha=solved.equation_residual_Ha,
                core_retained_rank12_residual_max_Ha=retained,
                core_unretained_rank34_linearized_residual_max_Ha=float(np.max(abs(residual[ranks > 2]))),
                linearized_trial_schrodinger_residual_max_Ha=float(np.max(abs(trial_residual))),
                core_fci_signed_method_error_Ha=delta, core_fci_abs_method_error_Ha=abs(delta),
                core_FCI_wave_overlap=overlap, retained_coordinate_count=len(selected) - 1,
                native_provenance=dict(solved.provenance),
                residual_scope="Q_SD(H+[H,T1+T2])|reference>, not exponential or physical Schrödinger residual. (1+T)|reference> is diagnostic only; linked LCCSD energy is not its Rayleigh quotient.",
                attachment="No genuine matched-H LCCSD attachment entry was executed; do not substitute core + separate valence energy.",
                gates=dict(retained_equations=retained <= GATE,
                           native_independent_energy=abs(energy - solved.total_energy_Ha) <= 1e-10,
                           reference_branch=overlap >= .5),
                arrays_sha256=sha(RUN / "lccsd_arrays.npz"))


def group_indices(energies):
    groups = []
    for index, energy in enumerate(energies):
        if not groups or abs(energy - energies[groups[-1][0]]) > 1e-9:
            groups.append([])
        groups[-1].append(index)
    return groups


def coupled(model):
    from atomx.methods.fscc import compute, load_integrals
    reader, oracle = helpers()
    a = load_integrals(RUN / "fixed_inputs")
    environment = os.environ.copy()
    libraries = REPO.parent / "atomx-references/2026-10-09-ci-fscc/toolchain/usr/lib/x86_64-linux-gnu"
    environment["LD_LIBRARY_PATH"] = ":".join((str(libraries), str(libraries / "openblas-serial"), environment.get("LD_LIBRARY_PATH", "")))
    environment.update(OMP_DYNAMIC="FALSE", OMP_THREAD_LIMIT="1", OMP_MAX_ACTIVE_LEVELS="1")
    directory = RUN / model.lower()
    solved = compute(a, executable=BACKEND, directory=directory, model=model, sector="0h1p",
                     active_indices=tuple(range(4, 12)), tolerance=1e-13,
                     residual_bound_tolerance_Ha=GATE, maxiter=200,
                     memory_bytes=512 * 1024**2, tile_size=4, threads=1,
                     timeout_seconds=120, environment=environment, tensor_symmetry="C1")
    if any(sha(directory / name) != sha(RUN / "fixed_inputs" / name) for name in ("ATOMX_ONE", "ATOMX_TWO")):
        raise ValueError("actual backend signed integral files differ from frozen inputs")
    rank = 2 if model == "CCSD" else 3
    tensors, metadata = {}, {}
    for stem in ("t", "s"):
        for level in range(1, rank + 1):
            name = f"{stem}{level}c"
            tensors[name], metadata[name] = reader.decode_diagram(directory / "scratch" / (name + ".dg"), 12)
    saved = np.load(RUN / "fci_arrays.npz")
    mc, ma = saved["core_masks"].tolist(), saved["attachment_masks"].tolist()
    hc, ha = saved["full_core_hamiltonian_Ha"], saved["full_attachment_hamiltonian_Ha"]
    terms = reader.cluster_terms([tensors[f"t{level}c"] for level in range(1, rank + 1)])
    tc, ta = oracle.excitation_matrix(mc, terms), oracle.excitation_matrix(ma, terms)
    ec, ea = expm(tc), expm(ta)
    hbc, hba = expm(-tc) @ hc @ ec, expm(-ta) @ ha @ ea
    ri, pi = mc.index(15), saved["model_indices"]
    energy = hbc[ri, ri]
    rc = hbc[:, ri].copy()
    rc[ri] -= energy
    sa = oracle.excitation_matrix(ma, reader.cluster_terms([tensors[f"s{level}c"] for level in range(1, rank + 1)], attachment=True))
    omega = (np.eye(len(ma)) + sa)[:, pi]
    matrix = solved.model_space_matrix_Ha
    total = matrix + energy * np.eye(len(pi))
    ra = hba @ omega - omega @ total
    cr = np.array([4 - (mask & 15).bit_count() for mask in mc])
    ar = np.array([4 - (mask & 15).bit_count() for mask in ma])
    core_retained, attachment_retained = (cr > 0) & (cr <= rank), ar <= rank - 1
    wave = ea @ omega
    roots, eigenvectors = np.linalg.eig(total)
    order = np.argsort(roots.real)
    roots, eigenvectors = roots[order], eigenvectors[:, order]
    physical_vectors = wave @ eigenvectors
    exact_roots, exact_vectors = saved["exact_attachment_roots_Ha"][:8], saved["exact_attachment_vectors"][:, :8]
    groups, exact_groups = group_indices(roots.real), group_indices(exact_roots)
    group_match = [len(group) for group in groups] == [len(group) for group in exact_groups]
    overlaps = []
    if group_match:
        for group, exact_group in zip(groups, exact_groups, strict=True):
            left = np.linalg.qr(physical_vectors[:, group])[0]
            right = np.linalg.qr(exact_vectors[:, exact_group])[0]
            overlaps.append(float(np.linalg.svd(left.conj().T @ right, compute_uv=False)[-1]))
    core_wave = ec[:, ri]
    core_overlap = float(abs(np.vdot(saved["exact_core_vectors"][:, 0], core_wave)) / np.linalg.norm(core_wave))
    core_delta = float(energy.real - saved["exact_core_roots_Ha"][0])
    attachment_delta = roots.real - exact_roots
    # Pairing transitions with their own core removes no physics by splicing.
    transition_delta = attachment_delta - core_delta
    retained_c = float(np.max(abs(rc[core_retained])))
    retained_a = float(np.max(abs(ra[attachment_retained])))
    projection = float(np.max(abs(matrix - ((hba @ omega)[pi] - energy * np.eye(len(pi))))))
    normalization = float(np.max(abs(omega[pi] - np.eye(len(pi)))))
    arrays = dict(core_masks=mc, attachment_masks=ma, model_indices=pi,
                  core_cluster=tc, attachment_cluster=ta, core_hbar_Ha=hbc,
                  attachment_hbar_Ha=hba, omega=omega, wave_operator=wave,
                  physical_metric=wave.conj().T @ wave, backend_heff_Ha=matrix,
                  total_heff_Ha=total, core_residual_Ha=rc, attachment_residual_Ha=ra,
                  core_ranks=cr, attachment_ranks=ar, attachment_roots_Ha=roots,
                  attachment_eigenvectors=eigenvectors, attachment_physical_vectors=physical_vectors,
                  core_wave=core_wave, core_physical_schrodinger_residual_Ha=hc @ core_wave - energy * core_wave,
                  attachment_physical_schrodinger_residual_Ha=ha @ physical_vectors - physical_vectors * roots,
                  **tensors)
    np.savez_compressed(RUN / (model.lower() + "_arrays.npz"), **arrays)
    return dict(method=model, native_backend_result=str((directory / "result.json").relative_to(REPO)),
                decoder_metadata=metadata, actual_model_keyword=(directory / "cc.inp").read_text().split("model ")[1].splitlines()[0],
                core_energy_Ha=float(energy.real), core_energy_native_Ha=solved.core_energy_Ha,
                raw_Jacobi_update_bound_Ha=solved.equation_residual_bound_Ha,
                core_retained_max_Ha=retained_c,
                core_omitted_max_Ha=float(np.max(abs(rc[cr > rank]), initial=0)),
                attachment_retained_max_Ha=retained_a,
                attachment_omitted_max_Ha=float(np.max(abs(ra[ar >= rank]), initial=0)),
                attachment_full_Bloch_max_Ha=float(np.max(abs(ra))),
                signed_HEFF_projection_max_Ha=projection, intermediate_normalization_max=normalization,
                core_fci_signed_method_error_Ha=core_delta, core_fci_abs_method_error_Ha=abs(core_delta),
                attachment_energies_Ha=roots.real.tolist(),
                attachment_fci_signed_method_errors_Ha=attachment_delta.tolist(),
                attachment_fci_abs_method_error_max_Ha=float(np.max(abs(attachment_delta))),
                attachment_minus_own_core_fci_signed_method_errors_Ha=transition_delta.tolist(),
                attachment_minus_own_core_fci_abs_method_error_max_Ha=float(np.max(abs(transition_delta))),
                core_FCI_wave_overlap=core_overlap, attachment_group_sizes=[len(group) for group in groups],
                FCI_attachment_group_sizes=[len(group) for group in exact_groups],
                attachment_group_min_principal_overlaps=overlaps,
                core_retained_rows=int(np.count_nonzero(core_retained)),
                attachment_retained_rows=int(np.count_nonzero(attachment_retained)),
                residual_scope=f"Independent Q_core(rank1..{rank}) e^-T H e^T |reference> and Q_attachment(core-hole rank0..{rank-1}) full signed eight-column Bloch residual; omitted rows reported separately.",
                gates=dict(core_retained_equations=retained_c <= GATE,
                           attachment_retained_equations=retained_a <= GATE,
                           signed_HEFF_projection=projection <= 1e-10,
                           intermediate_normalization=normalization <= 1e-12,
                           native_independent_core_energy=abs(energy - solved.core_energy_Ha) <= 1e-10,
                           reference_branch=core_overlap >= .5,
                           complete_attachment_root_groups=bool(group_match and overlaps and min(overlaps) >= .5),
                           real_roots=bool(np.max(abs(roots.imag)) <= 1e-10)),
                arrays_sha256=sha(RUN / (model.lower() + "_arrays.npz")))


def stage(name):
    resource.setrlimit(resource.RLIMIT_AS, (CAP, CAP))
    before = machine()
    if before["available_bytes"] < CAP + 1024**3 or before["free_disk_bytes"] < 1024**3:
        raise MemoryError("predeclared physical resource headroom unavailable")
    started = time.perf_counter()
    with LOCK.open("a") as lock:
        print("waiting shared timing.lock", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        def expire(_number, _frame):
            raise TimeoutError("predeclared 120-second stage cap exceeded")
        signal.signal(signal.SIGALRM, expire)
        signal.alarm(120)
        stage_started = time.perf_counter()
        try:
            result = fci() if name == "fci" else lccsd() if name == "lccsd" else coupled(name.upper())
        except Exception as error:
            result = dict(method=name.upper(), status="raw_failure_preserved", error=type(error).__name__,
                          message=str(error), gates=dict(execution=False))
        finally:
            signal.alarm(0)
            elapsed = time.perf_counter() - stage_started
            fcntl.flock(lock, fcntl.LOCK_UN)
    result.update(stage_wall_seconds=elapsed, elapsed_with_lock_wait_seconds=time.perf_counter() - started,
                  peak_self_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  peak_child_RSS_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024,
                  resource_before=before, resource_after=machine(), performance_claim=False,
                  RLIMIT_AS_bytes=list(resource.getrlimit(resource.RLIMIT_AS)),
                  thread_environment={key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")},
                  source_and_inputs_unchanged=unchanged(), contract_sha256=sha(RUN / "contract.json"))
    result["gates"]["source_inputs_unchanged"] = all(result["source_and_inputs_unchanged"].values())
    result["status"] = "finite_equations_and_matching_pass" if all(result["gates"].values()) else result.get("status", "scientific_gate_failed_preserved")
    dump(RUN / (name + "_result.json"), result)
    print(name, result["status"], result["gates"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "fci", "lccsd", "ccsd", "ccsdt"))
    argument = parser.parse_args()
    prepare() if argument.stage == "prepare" else stage(argument.stage)
