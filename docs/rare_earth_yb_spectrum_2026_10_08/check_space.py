"""Cheap CSF/determinant preflight; does not allocate a Hamiltonian."""
from collections import defaultdict
import json
from pathlib import Path
from atomx.config.csf import enumerate_jj_csfs
from atomx.angular.csf_hamiltonian import csf_signature, pattern_m_counts
from run_yb import CORE, CONFIGS

configs = CONFIGS + [CORE + " 4f13 6s2", CORE + " 4f13 5d1 6s1"]
csfs = [c for cfg in configs for c in enumerate_jj_csfs(cfg)]
patterns = defaultdict(set)
counts = defaultdict(int)
for c in csfs:
    block = (c.total_J2, c.parity)
    patterns[block].add(csf_signature(c).pattern)
    counts[block] += 1
blocks = []
for (j2, parity), pats in sorted(patterns.items()):
    dets = sum(dict(pattern_m_counts(p)).get(j2, 0) for p in pats)
    blocks.append({"two_J": j2, "parity": parity, "n_CSFS": counts[j2, parity],
        "raw_determinants": dets, "n_occupation_patterns": len(pats)})
output = {"configurations": configs, "n_CSFS": len(csfs), "blocks": blocks,
    "hypothesis": "Adding complete 4f13 6s2 and 4f13 5d6s manifolds at fixed common orbitals shifts/mixes the odd 6p levels; even 6s/5d energies must remain unchanged because parity separates the Hamiltonian.",
    "fixed_conditions": "same converged 24 radial orbitals, same Fermi nucleus and Dirac-Coulomb Hamiltonian; no orbital reoptimization and no selected-CSF truncation",
    "matching": "same J/parity and largest overlap with the embedded baseline eigenvector; require reference overlap squared >=0.5, otherwise mark ambiguous",
    "acceptance": {"CI_eigenpair_absolute_residual_Ha": 1e-8, "even_state_energy_drift_Ha": 1e-9},
    "resource_estimate": "46 CSFs; raw determinant dimensions explicitly listed. Estimate <3 GiB RSS; single CPU/BLAS thread; dense CI only, no all-electron SD expansion.",
    "stopping_rule": "900 seconds; exception, raw determinant cap 2000, eigenpair residual or even-state drift failure blocks a validated diagnosis; failed data retained",
    "interpretation_limit": "Finite fixed-orbital configuration mixing probe; no claim of correlation-complete or extended-space orbital stationarity"}
out = Path(__file__).resolve().parent / "extension_contract.json"
out.write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n")
print(json.dumps(output,ensure_ascii=False,indent=2))
