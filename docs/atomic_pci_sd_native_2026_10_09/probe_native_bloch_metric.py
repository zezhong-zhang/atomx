"""Full native finite Be+ Bloch metric, signs, core occupation and left/right."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from atomx.config.csf import enumerate_jj_orbital_space
from atomx.methods.ci_effective import build_finite_ci_effective_hamiltonian
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.mbpt import build_virtual_basis

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]
out = OWNER / "native_bloch_metric001"
out.mkdir(exist_ok=False)
contract = dict(
    hypothesis="Native exact full CI may yield non-Hermitian Bloch H, while wave-operator metric, full reconstructed states and physical signed properties remain consistent",
    fixed="Be+ Z4 DC/ns30/ks7, same closed1s2 DHF projector, two s and first p1/2,p3/2 virtual roots; full3electron J1/2 even space including all core holes",
    selected_roots="full lowest two eigenvalues, fixed before calculation; reject singular P projection or split degeneracy",
    p_space="exactly all closed1s2 core configurations within declared J/parity sector",
    gates_Ha=dict(intertwining=1e-10, metric_adjoint=1e-10, eigenenergy=1e-11),
    full_signed_property_gate=1e-11, dimension_cap=128, memory_bytes=256*1024**2,
    property="1s number operator in complete native CSF order; physical Q contributions retained",
    scope="finite native Bloch/metric anchor; does not infer a physical metric for fixed-energy SD Sigma1/2 or EXP-T output",
    stop="any failed complete signed reconstruction/metric/property rejects reduction; no averaging or root replacement",
    load=os.getloadavg(),
)
(out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
hashes = {}
sources = sorted((ROOT / "src/atomx").rglob("*.py")) + [Path(__file__).resolve()]
for path in sources:
    target = out / "source" / path.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
(out / "source_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
core = DFSolver(Z=4, subshells=[Subshell(1,-1,2)], ns=30, ks=7).solve()
orbitals = {(1,-1): core.orbitals_by_subshell()[1,-1][:2]}
for k in (-1,1,-2):
    _, states = build_virtual_basis(core, k, e_max=50, potential_mode="VN", orthogonal_to_occupied=True)
    for index, state in enumerate(states[:2 if k==-1 else 1]):
        orbitals[index+2,k] = state[:2]
all_csfs = enumerate_jj_orbital_space(orbitals, 3, magnetic_state_cap=256)
csfs = [c for c in all_csfs if (c.total_J2, c.parity)==(1,1)]
p = tuple(i for i,c in enumerate(csfs)
          if any(sh.n==1 and sh.kappa==-1 and sh.occupation==2 for sh in c.subshells))
if len(p)!=2:
    raise AssertionError("complete native closed-core P dimension differs")
model = build_finite_ci_effective_hamiltonian(
    csfs, p, orbitals, core._bsp_data, 4, 30,
    reference_identity="Be+:native-complete-finite-Bloch:2P",
    nuclear_potential=core._nuclear_potential, dimension_cap=128)
reduction = model.bloch_reduction((0,1), dimension_cap=128, memory_bytes=contract["memory_bytes"])
occupation = np.diag([sum(sh.occupation for sh in c.subshells if (sh.n,sh.kappa)==(1,-1)) for c in csfs])
energy, full = np.linalg.eigh(model.hamiltonian_Ha)
psi = full[:,:2]
expected_property = psi.T @ occupation @ psi
transformed = reduction.effective_operator(occupation, representation="bloch")
actual_property = reduction.left_states @ transformed @ reduction.right_states
root_defect = float(np.max(abs(reduction.wave_operator @ reduction.right_states-psi)))
property_defect = float(np.max(abs(expected_property-actual_property)))
energy_defect = float(np.max(abs(np.linalg.eigvalsh(reduction.orthonormal_Ha)-energy[:2])))
arrays = dict(full_Ha=model.hamiltonian_Ha, p_indices=np.asarray(p), energies_Ha=energy,
              full_states=full, wave_operator=reduction.wave_operator, metric=reduction.metric,
              directed_Ha=reduction.directed_Ha, orthonormal_Ha=reduction.orthonormal_Ha,
              isometry=reduction.orthonormal_wave_operator, right=reduction.right_states,
              left=reduction.left_states, full_core_occupation=occupation,
              effective_core_occupation=transformed, expected_property=expected_property,
              actual_property=actual_property)
for key,(a,b) in orbitals.items():
    arrays[f"P_{key[0]}_{key[1]}"],arrays[f"Q_{key[0]}_{key[1]}"] = a,b
for key,value in core._bsp_data.items():
    a=np.asarray(value)
    if a.dtype.kind in "biufc":
        arrays[f"workspace_{key}"]=a
np.savez_compressed(out / "complete_native_arrays.npz", **arrays)
record = dict(status="pass", full_csf_dimension=len(csfs), p_dimension=len(p),
              signed_state_defect=root_defect, full_property_defect=property_defect,
              eigenenergy_defect_Ha=energy_defect,
              directed_adjoint_defect_Ha=float(np.max(abs(reduction.directed_Ha-reduction.directed_Ha.T))),
              physical_metric_adjoint_defect_Ha=reduction.metric_adjoint_residual_Ha,
              intertwining_residual_Ha=reduction.intertwining_residual_Ha,
              projection_condition=reduction.projection_condition,
              bare_P_core_occupation=np.diag(occupation)[list(p)].tolist(),
              correlated_core_occupation=np.diag(expected_property).tolist(),
              csf_identities=[repr(x) for x in model.csf_identities], radial_binding=repr(model.radial_binding),
              space_converged=False, experiment_validated=False)
if max(root_defect,property_defect,energy_defect) > 1e-11:
    record["status"]="fail"
(out / "comparison.json").write_text(json.dumps(record, indent=2)+"\n")
print(json.dumps({k:v for k,v in record.items() if k not in ("csf_identities","radial_binding")}), flush=True)
if record["status"]!="pass":
    raise AssertionError("complete native finite Bloch signed gates failed")
