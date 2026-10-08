"""Small, bounded API demonstrations; these do not certify physical spectra.

Run from the Atomx checkout with its existing .venv. BLAS threads are pinned
before imports. A failed check stops the demonstration rather than changing
the model or relaxing its tolerance.
"""
import os

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OUT = ROOT / "foundation_results"

CONTRACT = {
    "purpose": "software_contract_demonstrations_only",
    "hypothesis": "The audited atomic and PySCF provider interfaces can run and preserve declared units, electron counts and orbital identity.",
    "not_claimed": ["converged QED", "molecular XAS", "periodic crystal XAS", "experimental validation"],
    "fixed_conditions": "one BLAS thread; explicit uncorrected atomic Hamiltonian; restricted RHF for finite molecular/embedding models",
    "acceptance": {"pyscf_scf_converged": True, "electron_count_absolute_error": 1e-8, "MO_metric_residual": 1e-8, "core_dominance_min": 0.8},
    "resource_scope": "three small examples; no correlated CI space, periodic supercell, or full AO four-index ERI requested",
    "stopping_rule": "stop on exception or contract failure; do not change model or gates",
}


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run():
    OUT.mkdir(exist_ok=True)
    save("contract.json", CONTRACT)
    from atomx import Atom
    from atomx.providers.pyscf_adapter import from_scf
    from atomx.providers.pyscf_embedding import rocksalt_embedding, build_embedded_scf, core_orbital_identity
    from pyscf import gto, scf

    t0 = time.perf_counter()
    atom = Atom(10, charge=9, config="1s1", corrections="none").solve()
    energy = float(atom.total_energy)
    assert -60 < energy < -40
    save("ne_x_ground.json", {
        "system": "Ne X, Ne9+, one electron", "Z": 10, "charge": 9,
        "config": "1s1", "engine": "shooting", "corrections": "none",
        "total_energy_hartree": energy,
        "interpretation": "Finite-grid Dirac ground-state API demonstration; no transition, QED, convergence extrapolation or rigorous accuracy bound.",
        "elapsed_seconds_uncontrolled": time.perf_counter() - t0,
    })

    t0 = time.perf_counter()
    mol = gto.M(atom="C 0 0 0; O 0 0 1.128", unit="Angstrom", basis="cc-pvdz", charge=0, spin=0, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.max_cycle = 100
    mf.kernel()
    assert mf.converged
    payload = from_scf(mf, metadata={"case_id": "S05", "role": "ground_state_contract_demo"})
    payload.validate()
    count = float(payload.electron_count())
    residual = float(payload.orthonormality_residual())
    assert abs(count - 14) < 1e-8 and residual < 1e-8
    save("co_provider.json", {
        "geometry_angstrom": [["C", 0, 0, 0], ["O", 0, 0, 1.128]],
        "geometry_role": "declared demonstration input, not newly optimized geometry",
        "basis": "cc-pvdz", "method": "nonrelativistic RHF", "scf_converged": bool(mf.converged),
        "total_energy_hartree": float(mf.e_tot), "electron_count": count,
        "MO_metric_residual": residual,
        "interpretation": "Provider and metric checks only; no core-hole or XAS calculation.",
        "elapsed_seconds_uncontrolled": time.perf_counter() - t0,
    })

    t0 = time.perf_counter()
    model = rocksalt_embedding("Mg", "O", 2.0, -2.0, 2.106, quantum_shells=0, mm_shells=1, evjen=True)
    embedded = build_embedded_scf(model, "sto-3g", conv_tol=1e-10, max_cycle=100)
    identity = core_orbital_identity(embedded, 0, "1s", environment=model.label)
    assert identity.dominance > 0.8
    save("mgo_embedding.json", {
        "model": model.label, "quantum_shells": 0, "mm_shells": 1,
        "nearest_neighbor_distance_angstrom": 2.106, "formal_point_charges_e": {"Mg": 2, "O": -2},
        "n_mm_charges": int(model.n_mm_charges), "basis": "sto-3g", "method": "point-charge embedded RHF",
        "scf_converged": bool(embedded.converged), "total_energy_hartree": float(embedded.e_tot),
        "core_label": identity.label, "core_MO_index": int(identity.mo_index),
        "core_dominance": float(identity.dominance), "core_runner_up": float(identity.runner_up),
        "core_MO_energy_hartree": float(identity.energy_hartree), "energy_zero": identity.energy_zero,
        "interpretation": "Finite qm0 embedding and identity demonstration; STO-3G is not a quantitative core spectroscopy basis. MO energy is not Delta-SCF binding energy or a crystal XAS peak.",
        "elapsed_seconds_uncontrolled": time.perf_counter() - t0,
    })
    files = ["src/atomx/atom.py", "src/atomx/atom_inputs.py", "src/atomx/providers/pyscf_adapter.py", "src/atomx/providers/pyscf_embedding.py", "src/atomx/providers/contract.py", "src/atomx/scattering/autoionization/molecular.py", "src/atomx/scattering/xray.py"]
    save("source_identity.json", {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "source_status": "working tree contains pre-existing uncommitted work; HEAD alone is not source identity",
        "audited_entrypoint_sha256": {f: hashlib.sha256((REPO/f).read_bytes()).hexdigest() for f in files},
        "hash_scope": "audited entry points only; not a complete transitive source snapshot",
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pyscf")},
        "BLAS_threads": 1,
    })
    save("checks.json", {"passed": True, "scope": "three software demonstrations", "new_physical_spectrum_certified": False})
    print(json.dumps({"passed": True, "output": str(OUT), "scope": "three software demonstrations"}, ensure_ascii=False))


if __name__ == "__main__":
    run()
