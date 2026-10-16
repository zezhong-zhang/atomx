"""Replay ten sealed Yb matrix snapshots into native exact CI, no new integrals."""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np

OUT = Path(__file__).resolve().parent
OLD = OUT.parent/'yb_matrix_self_energy_2026_10_08'
sys.path.insert(0, str(OUT.parent/'yb_diagnosis_repair_2026_10_08'))
from common import CORE, KEYS, load_reference, save, sha, source_identity
from atomx.config.csf import enumerate_jj_csfs_many, get_shared_orbital_basis
from atomx.methods.ci_mbpt import ci_mbpt_solve
from atomx.methods.mbpt_matrix import MatrixSelfEnergy
from atomx.physics.csf.effective_one_body import CIReferenceBinding
from atomx.physics.csf.matrix import build_ci_matrix
from atomx.physics.csf.radial_snapshot import CIRadialSnapshot


def timeout(*_):
    raise TimeoutError('180-second declared native replay cap per state')


if __name__ == '__main__':
    signal.signal(signal.SIGALRM, timeout)
    contract = dict(hypothesis='Full signed Sigma1 is consumed exactly by native CI in the same frozen core',
                    fixed='Saved ten canonical 3-function matrices; DC Fermi nucleus; same core/spinor phases/metric/zero',
                    gates=dict(bare_Fock_matrix_Ha=1e-8, CI_Dyson_energy_Ha=1e-8,
                               added_operator_matrix_Ha=1e-10, eigen_residual_Ha=1e-8,
                               wave_projector=1e-7, seconds_per_state=180, RSS_MiB=6144),
                    scope='consumer/representation regression, no new Yb accuracy or all-order physics')
    contract['legacy_representation'] = ('Explicit Hermitian roundoff projection (R+R.T)/2 only after '
                                         'the existing 64-eps symmetry/PSD residue gate; record changes. '
                                         'No magnitude truncation, pole merging or experimental fit.')
    save(OUT/'native_replay_contract.json', contract)
    initial_source = source_identity()
    save(OUT/'native_replay_source_sha256.json', initial_source)
    records = []
    started = time.monotonic()
    try:
        for case in json.loads((OLD/'preflight/results.json').read_text())['cases']:
            info = case['case']
            core, ref, _, identity = load_reference(info['reference'])
            core_od = {key: value[:2] for key, value in core.orbitals_by_subshell().items()}
            ledger = {(s.n, s.kappa): s.occupation for s in core.subshells}
            workspace = core._bsp_data
            ns = workspace['ns']
            potential = core._nuclear_potential
            cc = enumerate_jj_csfs_many([CORE])
            zero = float(build_ci_matrix(cc, core_od, get_shared_orbital_basis(cc), workspace, 70, ns,
                                          nuclear_potential=potential)[0, 0])
            z = np.load(OLD/'preflight'/(info['name']+'_virtuals.npz'))
            for label, (n0, kappa) in KEYS.items():
                signal.alarm(180)
                tick = time.monotonic()
                targets = [(P.copy(), Q.copy(), kappa, float(e)) for P, Q, e in zip(
                    z[f'external_P_{kappa}'], z[f'external_Q_{kappa}'], z[f'external_energy_{kappa}'])]
                keys = tuple((n0+i, kappa) for i in range(3))
                od = dict(core_od)
                od.update({key: target[:2] for key, target in zip(keys, targets)})
                letter = 's' if kappa == -1 else ('p' if kappa in (1, -2) else 'd')
                # Select the one-electron fine-structure component by exact J.
                csfs = [c for c in enumerate_jj_csfs_many([CORE+f' {n0+i}{letter}1' for i in range(3)])
                        if c.total_J2 == 2*abs(kappa)-1]
                basis = get_shared_orbital_basis(csfs)
                snapshot = CIRadialSnapshot(od, workspace, 70, ns, nuclear_potential=potential)
                bare = build_ci_matrix(csfs, od, basis, workspace, 70, ns,
                                       nuclear_potential=potential, radial_snapshot=snapshot)
                path = OLD/'results'/info['name']/label.replace('/', '_')/'matrix_poles.npz'
                binding = CIReferenceBinding.capture(targets, core_od, workspace, 70, ns, potential)
                with np.load(path) as saved:
                    # Legacy captures did not persist CI binding. Explicitly
                    # rebind only their archived canonical producer spinors/core;
                    # record the source and saved-input hashes below.
                    raw = MatrixSelfEnergy(saved['eps_Ha'], saved['poles_Ha'], saved['residues_Ha2'],
                                           str(saved['reference_identity']), ci_binding=binding)
                # The archived orthogonal rotation introduces last-bit
                # transpose asymmetry. The native operator remains strict;
                # this explicit, measured projection belongs to legacy replay.
                residue_asymmetry = float(np.max(abs(raw.residues_Ha2-raw.residues_Ha2.swapaxes(1,2))))
                model = MatrixSelfEnergy(raw.reference_energies_Ha, raw.poles_Ha,
                                         .5*(raw.residues_Ha2+raw.residues_Ha2.swapaxes(1,2)),
                                         raw.reference_identity, ci_binding=binding)
                eps = float(model.reference_energies_Ha[0])
                root = model.solve_dyson((eps-.1, eps+.02))
                sigma_projection_error = float(np.max(abs(model.evaluate(root.energy_Ha)-raw.evaluate(root.energy_Ha))))
                del raw
                op = model.as_ci_operator(root.energy_Ha, keys, od, workspace, 70, ns,
                                          correlated_core=ledger, nuclear_potential=potential)
                full = build_ci_matrix(csfs, od, basis, workspace, 70, ns, nuclear_potential=potential,
                                        radial_snapshot=snapshot, one_body_operator=op)
                result = ci_mbpt_solve(csfs, od, basis, workspace, 70, ns, sigma1_operator=op,
                                       nuclear_potential=potential, radial_snapshot=snapshot)
                eigenvector = result.mixing_coefficients[:, 0]
                rec = dict(case=info['name'], state=label, core_zero_Ha=zero,
                           bare_Fock_error_Ha=float(np.max(abs(bare-zero*np.eye(3)-np.diag(model.reference_energies_Ha)))),
                           operator_error_Ha=float(np.max(abs(full-bare-op.matrix_Ha))),
                           CI_Dyson_error_Ha=float(result.ground_energy-zero-root.energy_Ha),
                           eigen_residual_Ha=float(np.linalg.norm(full@eigenvector-result.ground_energy*eigenvector)),
                           wave_projector_error=float(np.max(abs(np.outer(eigenvector,eigenvector)-np.outer(root.coefficients,root.coefficients)))),
                           saved_matrix_sha256=sha(path), producer_identity=model.reference_identity,
                           legacy_residue_asymmetry_Ha2=residue_asymmetry,
                           legacy_sigma_projection_error_Ha=sigma_projection_error,
                           explicit_legacy_binding=op.provenance(), seconds=time.monotonic()-tick,
                           RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
                g = contract['gates']
                rec['passed'] = (rec['bare_Fock_error_Ha'] <= g['bare_Fock_matrix_Ha']
                                 and abs(rec['CI_Dyson_error_Ha']) <= g['CI_Dyson_energy_Ha']
                                 and rec['operator_error_Ha'] <= g['added_operator_matrix_Ha']
                                 and rec['eigen_residual_Ha'] <= g['eigen_residual_Ha']
                                 and rec['wave_projector_error'] <= g['wave_projector']
                                 and rec['RSS_MiB'] <= g['RSS_MiB'])
                rec['passed'] = rec['passed'] and sigma_projection_error <= g['added_operator_matrix_Ha']
                records.append(rec)
                save(OUT/'native_replay_results.json', dict(records=records, passed=False))
                print(info['name'], label, rec['passed'], rec['CI_Dyson_error_Ha'], flush=True)
                if not rec['passed']:
                    raise ValueError('native CI replay scientific gate failed')
                del model, op
                signal.alarm(0)
        if source_identity() != initial_source:
            raise ValueError('source changed during replay')
        save(OUT/'native_replay_results.json', dict(records=records, passed=True, seconds=time.monotonic()-started))
    except Exception as error:
        save(OUT/'native_replay_results.json', dict(records=records, passed=False, error=repr(error), seconds=time.monotonic()-started))
        raise
