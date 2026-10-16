"""Complete finite Be+ electric lines: two metrics, signed magnetic oracle."""
import hashlib
import json
import os
import runpy
import shutil
from pathlib import Path

import numpy as np

from atomx.config.csf import enumerate_jj_orbital_space
from atomx.methods.ci_effective import build_finite_ci_effective_hamiltonian
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.mbpt import build_virtual_basis
from atomx.properties.transitions.ci_effective import finite_ci_electric_line

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]
OUT = OWNER / "native_bloch_transitions004"
OUT.mkdir(exist_ok=False)
contract = dict(
    hypothesis="Two endpoint wave operators and their physical metrics reproduce complete signed native electric transitions without bare POP or Euclidean averaging",
    fixed="Be+ Z4/ns30/ks7 closed1s2 DC DHF; complete three-electron CSFs over 1s,2s,3s,2p1/2,2p3/2 native VN metric-orthogonal spinors; infinite nuclear mass; same native nuclear potential",
    sectors=[[1,1],[1,-1],[3,-1]],
    roots="lowest two complete full-CI roots in J1/2 even; lowest root in each odd sector, selected before calculation",
    lines="2p1/2,2p3/2 -> 2s and 3s -> 2p1/2,2p3/2; line-specific many-electron photon difference in every orbital pair",
    operators="E1 length/velocity x finite_wavelength/explicit long_wavelength",
    independent_oracle="lowest-M CSF vectors, explicit fermionic creation/annihilation and one-electron Wigner-Eckart components; no one_body_tensor_terms/effective transform; radial photon kernel is common",
    gates=dict(full_signed_operator_au=1e-11, normalized_amplitude_au=1e-11,
               reconstructed_state=1e-11, invariant_residual_Ha=1e-10),
    caps=dict(dimension_cap=128,max_determinants=2000,angular_work_cap=10_000_000,
              memory_bytes=64*1024**2),
    stop="Any complete array, signed state, amplitude or metric failure rejects this increment; no root substitution, amplitude screening, fitted shifts or operator averaging",
    scope="Exact finite CI consumer only. No metric inferred for production SD, no Yb spectrum/space/experiment convergence, no performance claim",
    load=os.getloadavg(),
)
(OUT / "contract.json").write_text(json.dumps(contract,indent=2)+"\n")
oracle_path = ROOT / "tests/test_ci_effective_transitions.py"
sources = sorted((ROOT / "src/atomx").rglob("*.py")) + [Path(__file__).resolve(),oracle_path]
hashes = {}
for path in sources:
    key = str(path.relative_to(ROOT))
    target = OUT / "source" / key
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path,target)
    hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
(OUT / "source_sha256.json").write_text(json.dumps(hashes,indent=2)+"\n")
oracle = runpy.run_path(str(oracle_path))["determinant_e1_matrix"]
core = DFSolver(Z=4,subshells=[Subshell(1,-1,2)],ns=30,ks=7).solve()
orbitals = {(1,-1):core.orbitals_by_subshell()[1,-1][:2]}
for k in (-1,1,-2):
    _,states = build_virtual_basis(core,k,e_max=50,potential_mode="VN",orthogonal_to_occupied=True)
    for i,state in enumerate(states[:2 if k == -1 else 1]):
        orbitals[i+2,k] = state[:2]
all_csfs = enumerate_jj_orbital_space(orbitals,3,magnetic_state_cap=256)
arrays, panel, sectors = {},{},[]
for j2,parity in ((1,1),(1,-1),(3,-1)):
    csfs = tuple(c for c in all_csfs if (c.total_J2,c.parity) == (j2,parity))
    p = tuple(i for i,c in enumerate(csfs)
              if any((s.n,s.kappa,s.occupation) == (1,-1,2) for s in c.subshells))
    model = build_finite_ci_effective_hamiltonian(
        csfs,p,orbitals,core._bsp_data,4,30,reference_identity=f"Be+:E1:complete:{j2}:{parity}",
        nuclear_potential=core._nuclear_potential,dimension_cap=128)
    reduction = model.bloch_reduction(tuple(range(len(p))),dimension_cap=128,
                                     memory_bytes=contract["caps"]["memory_bytes"])
    energies, states = np.linalg.eigh(model.hamiltonian_Ha)
    reconstructed = reduction.wave_operator @ reduction.right_states
    defect = float(np.max(abs(reconstructed-states[:,:len(p)])))
    name = f"sector_{j2}_{parity}"
    for key,value in dict(H=model.hamiltonian_Ha,p_indices=p,energies=energies,full_states=states,
                          Omega=reduction.wave_operator,G=reduction.metric,HB=reduction.directed_Ha,
                          W=reduction.orthonormal_wave_operator,Horth=reduction.orthonormal_Ha,
                          left=reduction.left_states,right=reduction.right_states,
                          reconstructed_states=reconstructed).items():
        arrays[f"{name}_{key}"] = np.asarray(value)
    sectors.append(dict(J2=j2,parity=parity,full_csf_dimension=len(csfs),P_dimension=len(p),
                         Q_dimension=len(csfs)-len(p),signed_state_defect=defect,
                         H_Omega_residual_Ha=reduction.intertwining_residual_Ha,
                         metric_adjoint_residual_Ha=reduction.metric_adjoint_residual_Ha,
                         projection_condition=reduction.projection_condition,
                         csf_identities=[repr(i) for i in model.csf_identities],
                         radial_binding=repr(model.radial_binding)))
    panel[j2,parity] = (csfs,reduction,states[:,:len(p)])
lines = []
for j2 in (1,3):
    for excitation in ("2p_to_2s","3s_to_2p"):
        upper_key,lower_key = ((j2,-1),(1,1)) if excitation == "2p_to_2s" else ((1,1),(j2,-1))
        upper_state = 0 if excitation == "2p_to_2s" else 1
        uc,u,pu = panel[upper_key]
        lc,l,pl = panel[lower_key]
        for gauge in ("length","velocity"):
            for approximation in ("finite_wavelength","long_wavelength"):
                line = finite_ci_electric_line(
                    u,l,uc,lc,orbitals,core._bsp_data,4,30,upper_state=upper_state,
                    representation="bloch",gauge=gauge,photon_approximation=approximation,
                    nuclear_potential=core._nuclear_potential,**contract["caps"])
                direct = oracle(uc,lc,orbitals,core._bsp_data,line.omega_Ha,gauge,approximation)
                expected = pu.T @ direct @ pl
                bloch = u.effective_transition(line.full_operator,l,representation="bloch")
                orth = u.effective_transition(line.full_operator,l,representation="orthonormal")
                ab = u.transition_amplitudes(line.full_operator,l,representation="bloch")
                ao = u.transition_amplitudes(line.full_operator,l,representation="orthonormal")
                bare = direct[np.ix_(u.parent.p_indices,l.parent.p_indices)]
                physical_p_only = u.right_states.T @ bare @ l.right_states
                bare_bloch = u.left_states @ bare @ l.right_states
                op_defect = float(np.max(abs(line.full_operator.matrix-direct)))
                b_defect = float(np.max(abs(ab-expected)))
                o_defect = float(np.max(abs(ao-expected)))
                name = f"line_{j2}_{excitation}_{gauge}_{approximation}"
                for key,value in dict(full_operator=line.full_operator.matrix,
                                      determinant_operator=direct,effective_bloch=bloch,
                                      effective_orthonormal=orth,expected_amplitudes=expected,
                                      bloch_amplitudes=ab,orthonormal_amplitudes=ao,
                                      physical_P_only_amplitudes=physical_p_only,
                                      bare_bloch_amplitudes=bare_bloch).items():
                    arrays[f"{name}_{key}"] = value
                lines.append(dict(J2=j2,transition=excitation,gauge=gauge,photon_approximation=approximation,
                                  operator_identity=line.full_operator.operator_identity,
                                  upper_root_index=line.upper_root_index,lower_root_index=line.lower_root_index,
                                  omega_Ha=line.omega_Ha,reduced_matrix_element_au=line.reduced_matrix_element_au,
                                  S=line.line_strength,A_per_s=line.A_rate_per_s,f=line.f_value,gf=line.gf_value,
                                  signed_operator_defect=op_defect,signed_bloch_amplitude_defect=b_defect,
                                  signed_orthonormal_amplitude_defect=o_defect,
                                  physical_P_only_defect=float(np.max(abs(expected-physical_p_only))),
                                  bare_bloch_consumer_defect=float(np.max(abs(expected-bare_bloch)))))
for key,(p,q) in orbitals.items():
    arrays[f"P_{key[0]}_{key[1]}"],arrays[f"Q_{key[0]}_{key[1]}"] = p,q
for key,value in core._bsp_data.items():
    a = np.asarray(value)
    if a.dtype.kind in "biufc":
        arrays[f"workspace_{key}"] = a
np.savez_compressed(OUT / "complete_native_arrays.npz",**arrays)
max_op = max(line["signed_operator_defect"] for line in lines)
max_amp = max(max(line["signed_bloch_amplitude_defect"],line["signed_orthonormal_amplitude_defect"])
              for line in lines)
source_unchanged = all(hashlib.sha256((ROOT / key).read_bytes()).hexdigest() == digest
                       for key,digest in hashes.items())
passed = (max_op <= 1e-11 and max_amp <= 1e-11 and source_unchanged
          and all(s["signed_state_defect"] <= 1e-11 and s["H_Omega_residual_Ha"] <= 1e-10
                  and s["metric_adjoint_residual_Ha"] <= 1e-10 for s in sectors))
record = dict(status="pass" if passed else "fail",sector_count=len(sectors),line_count=len(lines),
              max_complete_signed_operator_defect=max_op,max_normalized_signed_amplitude_defect=max_amp,
              source_unchanged=source_unchanged,sectors=sectors,lines=lines,
              finite_CI_consumer_verified=passed,production_SD_metric_verified=False,
              basis_converged=False,experiment_validated=False,performance_claim=False)
(OUT / "comparison.json").write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps({k:v for k,v in record.items() if k not in ("sectors","lines")}),flush=True)
if not passed:
    raise AssertionError("complete native finite transition gates failed")
