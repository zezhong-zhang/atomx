"""Fresh native Yb68 DC reference and Yb+ Table-I DHF comparison.

This is an independently specified B-spline baseline, not matched RCCSDT
reproduction: the paper's GTO/nuclear inputs and DHF Breit ledger are unknown.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import signal
import tarfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'pyproject.toml').is_file())
CORE = '1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 4f14'
THREADS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    def scalar(x):
        return x.tolist()
    Path(path).write_text(json.dumps(obj, indent=2, default=scalar,
                                    allow_nan=False) + '\n')


def source_map():
    return {str(p.relative_to(ROOT)): sha(p)
            for p in sorted((ROOT / 'src/atomx').rglob('*.py'))}


def prepare():
    mapping = source_map()
    fingerprint = hashlib.sha256(json.dumps(mapping, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    save(HERE / 'source003.json', dict(all_atomx_python_sha256=mapping,
                                  source_map_sha256=fingerprint))
    with tarfile.open(HERE / 'source003.tar.gz', 'w:gz') as archive:
        for name in mapping:
            archive.add(ROOT / name, arcname=name, recursive=False)
    parent = HERE.parent / 'paper_contract.json'
    contract = dict(
        hypothesis='A fresh stationary native 68-electron reference reproduces the scale of the published DHF attachment energies before any correlation correction; a fixed two-basis comparison distinguishes radial error from the later correlation gap.',
        paper=dict(version='arXiv:2504.01438v1',
            url='https://arxiv.org/html/2504.01438v1', table='Table1, energy/DHF',
            EA_cm1=[-90790, -66517, -66038],
            state_order=['6s S1/2', '5d D3/2', '5d D5/2'],
            reference_only=True, rounding_halfwidth_cm1=0.5,
            unknowns='40-GTO exponents, nuclear model/parameters, exact DHF DC/Breit separation, correlated partitions and solver controls remain unknown.'),
        fixed=dict(Z=70, closed_core_electrons=68, target_electrons=69,
            core=CORE, nucleus='Fermi empirical radius selected by A171; not an experimental radius and not an asserted paper input',
            nuclear_mass_number=171, r_min='native Fermi first-step default',
            r_max_bohr=500.0, basis_order=9, kinetic_balance=True,
            grid='pure exponential native make_knots, not the paper GTO basis',
            exchange='nonlocal HF', hamiltonian='Dirac-Coulomb only, rest energy subtracted; no recoil/QED/Breit',
            electron_projector='native solve_dirac positive branch, stored-core S-orthogonal complement for attached orbitals',
            comparison_zero='E(Yb69 frozen core)-E(Yb68) = valence Fock eigenvalue',
            optimization='closed68-electron DHF only; no69-electron MCDHF claim',
            radial_sizes=[180, 220], scf_max_iter=160,
            scf_tolerance=1e-9, scf_orbital_tolerance=1e-8,
            valence_keys=[[6, -1], [5, 2], [5, -3]]),
        acceptance=dict(core_converged=True, occupied_S_Gram=1e-10,
            occupied_Fock_tangent_dual_Ha=1e-6,
            valence_projected_dual_Ha=1e-8, valence_S_Gram=1e-10,
            core_CI_minus_DHF_Ha=1e-7,
            full69_CI_minus_core_plus_valence_Ha=1e-7,
            radial_EA_and_lines_drift_cm1=1.0,
            paper='Observed difference from rounded row only; numerical agreement is not exact input/method reproduction or experimental accuracy.'),
        resources=dict(cpu=27, threads=1, AS_bytes=8*1024**3,
            seconds=1200, load1_max=24, available_min_bytes=16*1024**3,
            estimate='19 core and3 attached radial orbitals, seven Fock pencils/arm, three complete electron eigensystems/arm; largest active pencil433 and at most217 electron columns; four singleton CI blocks/arm; <256MiB retained NumPy payload/arm, <2GiB anticipated RSS.',
            fock_calls_per_arm=7, valence_eigensystems_per_arm=3,
            CI_CSFS_per_arm=4, CI_calls_per_arm=2, raw_determinant_cap=32,
            array_output_bytes=256*1024**2),
        stopping='Stop on first identity, convergence, residual or resource failure; preserve partial results. No automatic retry, radial-size scan, relaxed gate, fitted energy, borrowed paper correction or backend execution.',
        source_map_sha256=fingerprint,
        inputs={str(parent.relative_to(ROOT)):sha(parent),
                str(Path(__file__).relative_to(ROOT)):sha(__file__)})
    contract['predecessor'] = dict(contract_sha256=sha(HERE/'contract.json'),
        failure_sha256=sha(HERE/'actual002/result.json'),
        diagnosis_sha256=sha(HERE/'diagnosis002/result.json'),
        scope='Same source, nucleus, radial sizes, physics, scientific gates and caps; repair only explicit whole-branch metric representation before native refinement.')
    contract['metric_repair'] = dict(method='invertible Cholesky metric whitening, ordered QR of the whole positive branch, native leading-block refinement, back transport and native original-pencil Newton; final ordered QR retains every column',
        original_and_white_Gram_max=1e-10, original_Gram_arithmetic='extended precision on the same saved FP64 metric and returned FP64 coefficients; ordinary FP64 Gram is also reported',
        whitening='L L.T = S, Hwhite=L^-1 H L^-T using extended contractions; no physical coordinate/column/root count reduction',
        root_policy='Original selected-root S-dual residual, norm, core orthogonality and one-quarter original nearest gap remain mandatory; all unselected columns retained without a high-root residual certificate.')
    for item in ('contract.json','actual001/result.json','actual001/core_ns180.h5','diagnosis001/result.json','actual002/result.json','actual002/core_ns220.h5','diagnosis002/result.json'):
        contract['inputs'][str((HERE/item).relative_to(ROOT))] = sha(HERE/item)
    save(HERE / 'contract003.json', contract)
    print(json.dumps(dict(status='PREPARED', source_files=len(mapping),
                          source_map_sha256=fingerprint)))


def run():
    contract = json.loads((HERE / 'contract003.json').read_text())
    identity = json.loads((HERE / 'source003.json').read_text())
    assert source_map() == identity['all_atomx_python_sha256']
    assert all(sha(ROOT / p) == h for p, h in contract['inputs'].items())
    assert all(os.environ.get(k) == '1' for k in THREADS)
    limits = contract['resources']
    assert os.getloadavg()[0] <= limits['load1_max']
    available = next(int(line.split()[1])*1024 for line in
        Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    assert available >= limits['available_min_bytes']
    os.sched_setaffinity(0, {limits['cpu']})
    resource.setrlimit(resource.RLIMIT_AS, (limits['AS_bytes'], limits['AS_bytes']))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(
        TimeoutError('Declared1200-second experiment cap')))
    signal.alarm(limits['seconds'])
    out = HERE / 'actual003'
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    report = dict(status='RUNNING', source_map_sha256=identity['source_map_sha256'],
                  contract_sha256=sha(HERE / 'contract003.json'), arms=[], controls=[])
    def gate(name, value, cap=None):
        passed = bool(value) if cap is None else bool(value <= cap)
        report['controls'].append(dict(name=name, measured=value,
                                       cap=cap, passed=passed))
        if not passed:
            raise AssertionError(name + ': ' + str(value))
    try:
        import numpy as np
        from scipy import linalg

        from atomx.config.csf import enumerate_jj_csfs
        from atomx.constants import C_SQ, HARTREE_CM1
        from atomx.methods.ci import ci_solve
        from atomx.methods.dirac_fock import DFSolver, SCFResult
        from atomx.methods.mbpt import _build_closed_shell_fock
        from atomx.numerics.bspline.convol import ConvolCache
        from atomx.numerics.bspline.dirac_assembly import (
            _get_keep_grid,
            _refine_eigenpair,
            refine_generalized_eigenblock,
            solve_dirac,
        )

        a = contract['acceptance']
        f = contract['fixed']
        for ns in f['radial_sizes']:
            arm = dict(ns=ns, stage='SCF', fock_blocks=[], valence=[])
            report['arms'].append(arm)
            print('START_SCF', ns, flush=True)
            solver = DFSolver.from_configuration(70, CORE, ns=ns, ks=9,
                r_max=500.0, kinetic_balance=True, nuclear='fermi',
                nuclear_mass_number=171, starting_potential='tfda')
            gate(f'{ns}:whole_closed_core', len(solver.subshells) == 19 and
                 sum(s.occupation for s in solver.subshells) == 68 and
                 all(s.occupation == 2*abs(s.kappa) for s in solver.subshells))
            if ns in (180, 220):
                checkpoint = HERE / ('actual001/core_ns180.h5' if ns == 180 else 'actual002/core_ns220.h5')
                core = SCFResult.load(checkpoint)
                from atomx.potentials.nuclear import make_nuclear_potential
                gate(f'{ns}:restored_exact_nucleus',
                     core.solver_diagnostics['_nuclear_profile'] == solver.nuclear_profile)
                gate(f'{ns}:restored_exact_knots', np.array_equal(core._bsp_data['t'], solver.t))
                gate(f'{ns}:restored_exact_core', [(x.n,x.kappa,x.occupation) for x in core.subshells] ==
                     [(x.n,x.kappa,x.occupation) for x in solver.subshells])
                core._nuclear_potential = make_nuclear_potential(70, 'fermi', 171)
                arm['SCF_source'] = str(checkpoint.relative_to(HERE)) + '; exact source-bound checkpoint, no second SCF'
            else:
                core = solver.solve(max_iter=f['scf_max_iter'], tol=f['scf_tolerance'],
                                    orbital_tolerance=f['scf_orbital_tolerance'])
                arm['SCF_source'] = 'fresh current native SCF'
            arm.update(iterations=core.iterations, converged=core.converged,
                residual=core.residual, total_energy_Ha=core.total_energy,
                orbital_energies_Ha=core.orbital_energies,
                nuclear_profile=solver.nuclear_profile,
                solver_diagnostics=core.solver_diagnostics,
                calculation_spec=core.calculation_spec.to_dict())
            core.save(str(out / f'core_ns{ns}'))
            save(out / 'partial.json', report)
            gate(f'{ns}:core_converged', core.converged)
            data = core._bsp_data
            cache = ConvolCache(data)
            orbitals = {k: (p.copy(), q.copy())
                        for k, (p, q, occ) in core.orbitals_by_subshell().items()}
            arrays = dict(orbital_keys=np.asarray(list(orbitals)),
                core_P=np.stack([p for p,q in orbitals.values()]),
                core_Q=np.stack([q for p,q in orbitals.values()]))
            keep, grid = _get_keep_grid(ns, data['nsp'], data['nsq'], 1, 2)
            arrays['active_coordinates'] = keep
            total_bytes = 0
            eps = {}
            for kappa in sorted({k for n,k in orbitals}):
                print('FOCK', ns, kappa, flush=True)
                H, S = _build_closed_shell_fock(core, kappa, cache=cache,
                                               potential_mode='VN')
                h = np.tril(H[grid]) + np.tril(H[grid], -1).T
                s = S[grid]
                lower = linalg.cholesky(s, lower=True)
                keys = [key for key in orbitals if key[1] == kappa]
                c = np.column_stack([np.concatenate(orbitals[key])[keep] for key in keys])
                gram = c.T @ s @ c
                lam = c.T @ h @ c
                defect = h @ c - s @ c @ lam
                dual = linalg.solve_triangular(lower, defect, lower=True)
                norm = float(np.linalg.norm(dual, axis=0).max())
                gram_error = float(np.max(abs(gram-np.eye(len(keys)))))
                block = dict(kappa=kappa, occupied_keys=keys,
                    occupied_tangent_dual_Ha=norm, occupied_Gram_error=gram_error)
                arm['fock_blocks'].append(block)
                arrays[f'H_k{kappa}'] = h
                arrays[f'S_k{kappa}'] = s
                arrays[f'occupied_k{kappa}'] = c
                arrays[f'occupied_Fock_k{kappa}'] = lam
                arrays[f'occupied_tangent_k{kappa}'] = defect
                # Save even a failed occupied stationarity measurement.
                np.savez(out / f'blocks_ns{ns}.npz', **arrays)
                gate(f'{ns}:occupied_Gram:{kappa}', gram_error, a['occupied_S_Gram'])
                gate(f'{ns}:occupied_tangent:{kappa}', norm,
                     a['occupied_Fock_tangent_dual_Ha'])
                selected = [tuple(key) for key in f['valence_keys'] if key[1] == kappa]
                if not selected:
                    continue
                gate(f'{ns}:one_valence_per_kappa:{kappa}', len(selected) == 1)
                key = selected[0]
                # Explicit occupied complement preserves the actual core bank.
                normals = s @ c
                basis, triangular = linalg.qr(normals, mode='full')
                gate(f'{ns}:independent_core_constraints:{kappa}',
                     bool(np.min(abs(np.diag(triangular[:len(keys)]))) > 1e-13))
                complement = basis[:, len(keys):]
                hp = complement.T @ h @ complement
                sp = complement.T @ s @ complement
                # eigh uses the lower triangle; materialize that same metric
                # for the explicit exactly symmetric refinement contract.
                block['complement_metric_asymmetry'] = float(np.max(abs(sp-sp.T)))
                sp = np.tril(sp) + np.tril(sp, -1).T
                raw = solve_dirac(H, S, ns, kappa, nsp=data['nsp'],
                    nsq=data['nsq'], orthogonal_to=np.column_stack(
                        [np.concatenate(orbitals[k]) for k in keys]))
                gate(f'{ns}:complete_electron_branch:{kappa}',
                     raw.coefficients.shape[1] == len(raw.eigenvalues))
                v = complement.T @ raw.coefficients[keep]
                g = v.T @ sp @ v
                block['raw_complement_Gram'] = float(np.max(abs(g-np.eye(len(g)))))
                metric_lower = linalg.cholesky(sp, lower=True)
                inverse_lower = linalg.solve_triangular(metric_lower, np.eye(len(sp)), lower=True)
                hp = np.tril(hp) + np.tril(hp, -1).T
                wide = np.longdouble
                white_h = np.asarray(inverse_lower.astype(wide) @ hp.astype(wide)
                                     @ inverse_lower.T.astype(wide), dtype=float)
                white_h = np.tril(white_h) + np.tril(white_h, -1).T
                white_v, qr = linalg.qr(metric_lower.T @ v, mode='economic')
                signs = np.sign(np.diag(qr))
                gate(f'{ns}:whole_white_QR_rank:{kappa}', bool(np.all(signs != 0)))
                white_v *= signs
                e, white_v, info = refine_generalized_eigenblock(white_h,
                    np.eye(len(sp)), raw.eigenvalues, white_v, leading_count=1,
                    dimension_cap=512, column_cap=256, memory_cap_bytes=256*1024**2)
                v = linalg.solve_triangular(metric_lower.T, white_v, lower=False)
                original_gap = float(min(abs(raw.eigenvalues[1:]-raw.eigenvalues[0])))
                e[0], v[:,0] = _refine_eigenpair(hp, sp, e[0], v[:,0], original_gap,
                    residual_norm=lambda r, lower=metric_lower: np.linalg.norm(linalg.solve_triangular(lower,r,lower=True)))
                # Ordered QR preserves the selected leading vector's direction.
                white_v, qr = linalg.qr(metric_lower.T @ v, mode='economic')
                white_v *= np.sign(np.diag(qr))
                v = linalg.solve_triangular(metric_lower.T, white_v, lower=False)
                block['whole_original_Gram_float'] = float(np.max(abs(v.T@sp@v-np.eye(len(g)))))
                block['whole_original_Gram_wide'] = float(np.max(abs(v.astype(wide).T@sp.astype(wide)@v.astype(wide)-np.eye(len(g)))))
                block['whole_white_Gram'] = float(np.max(abs(white_v.T@white_v-np.eye(len(g)))))
                block['selected_root_shift_Ha'] = float(e[0]-raw.eigenvalues[0])
                arrays[f'raw_complement_Gram_k{kappa}'] = g
                arrays[f'white_H_k{kappa}'] = white_h
                arrays[f'white_C_k{kappa}'] = white_v
                arrays[f'metric_Cholesky_k{kappa}'] = metric_lower
                gate(f'{ns}:whole_original_Gram:{kappa}', block['whole_original_Gram_wide'], a['valence_S_Gram'])
                gate(f'{ns}:whole_white_Gram:{kappa}', block['whole_white_Gram'], a['valence_S_Gram'])
                gate(f'{ns}:selected_original_root:{kappa}', abs(block['selected_root_shift_Ha'])/original_gap, .25)
                full = np.zeros((2*ns, v.shape[1]))
                full[keep] = complement @ v
                p, q = full[:ns, 0].copy(), full[ns:, 0].copy()
                energy = float(e[0])
                r = hp @ v[:,0] - energy*(sp @ v[:,0])
                projected_dual = float(np.linalg.norm(linalg.solve_triangular(
                    linalg.cholesky(sp, lower=True), r, lower=True)))
                root_gram = abs(float(np.r_[p,q] @ S @ np.r_[p,q])-1)
                core_overlap = float(np.max(abs(c.T @ s @ full[keep,0])))
                row = dict(key=key, EA_Ha=energy, EA_cm1=energy*HARTREE_CM1,
                    complete_electron_columns=len(e), full_coordinate_dimension=len(keep),
                    complement_dimension=len(sp), projected_dual_Ha=projected_dual,
                    norm_defect=root_gram, occupied_overlap=core_overlap,
                    eigenblock=info)
                arm['valence'].append(row)
                arrays[f'complement_k{kappa}'] = complement
                arrays[f'positive_E_k{kappa}'] = e
                arrays[f'positive_C_k{kappa}'] = full
                arrays[f'raw_positive_E_k{kappa}'] = raw.eigenvalues
                arrays[f'raw_positive_C_k{kappa}'] = raw.coefficients
                np.savez(out / f'blocks_ns{ns}.npz', **arrays)
                gate(f'{ns}:bound_lowest_virtual:{kappa}', -C_SQ < energy < 0)
                gate(f'{ns}:valence_dual:{kappa}', projected_dual,
                     a['valence_projected_dual_Ha'])
                gate(f'{ns}:valence_metric:{kappa}', max(root_gram, core_overlap),
                     a['valence_S_Gram'])
                eps[key] = energy
                orbitals[key] = p, q
                print('EA', ns, key, row['EA_cm1'], projected_dual, flush=True)
            gate(f'{ns}:whole_fock_census', len(arm['fock_blocks']) == 7)
            gate(f'{ns}:whole_valence_census', len(eps) == 3)
            arm['stage'] = 'CI'
            # Distinct fixed-N sectors retain their own complete CSF/H ledger.
            sectors = [enumerate_jj_csfs(CORE),
                enumerate_jj_csfs(CORE+' 6s1')+enumerate_jj_csfs(CORE+' 5d1')]
            gate(f'{ns}:whole_CSF_census', [len(cs) for cs in sectors] == [1,3])
            rows = []
            provenance = {}
            for csfs in sectors:
                electrons = csfs[0].n_electrons
                gate(f'{ns}:fixed_N_sector:{electrons}',
                     all(c.n_electrons == electrons for c in csfs))
                ci = ci_solve(csfs, orbitals, sorted(orbitals), data, 70, ns,
                    solver='dense', nuclear_potential=core._nuclear_potential,
                    max_determinants=limits['raw_determinant_cap'], retain_hamiltonian=True)
                for i, c in enumerate(csfs):
                    root = int(np.argmax(abs(ci.mixing_coefficients[i])))
                    energy = float(ci.state_energies[root])
                    valence = [(s.n,s.kappa) for s in c.subshells if s.occupation == 1]
                    reference = core.total_energy + (eps[valence[0]] if valence else 0)
                    error = abs(energy-reference)
                    rows.append(dict(CSF=c.label, root=root, electrons=electrons,
                        energy_Ha=energy, expected_Ha=reference, discrepancy_Ha=error))
                    gate(f'{ns}:CI_energy_ledger:{electrons}:{i}', error,
                        a['full69_CI_minus_core_plus_valence_Ha'] if valence else
                        a['core_CI_minus_DHF_Ha'])
                arrays[f'CI_H_N{electrons}'] = ci.hamiltonian_snapshot.matrix_Ha
                provenance[electrons] = ci.provenance
                arrays[f'CI_E_N{electrons}'] = ci.state_energies
                arrays[f'CI_C_N{electrons}'] = ci.mixing_coefficients
            arm['CI'] = dict(rows=rows, provenance=provenance)
            arrays['all_orbital_keys'] = np.asarray(list(orbitals))
            arrays['all_P'] = np.stack([p for p,q in orbitals.values()])
            arrays['all_Q'] = np.stack([q for p,q in orbitals.values()])
            total_bytes = sum(x.nbytes for x in arrays.values())
            gate(f'{ns}:output_array_bytes', total_bytes, limits['array_output_bytes'])
            np.savez(out / f'blocks_ns{ns}.npz', **arrays)
            ordered = [eps[tuple(k)]*HARTREE_CM1 for k in f['valence_keys']]
            arm['EA_cm1'] = ordered
            arm['lines_cm1'] = [ordered[1]-ordered[0], ordered[2]-ordered[0]]
            arm['minus_paper_DHF_cm1'] = [x-y for x,y in zip(ordered,
                contract['paper']['EA_cm1'], strict=True)]
            arm['stage'] = 'COMPLETE'
            arm['array_bytes'] = total_bytes
            save(out / 'partial.json', report)
            print('ARM_COMPLETE', ns, arm['EA_cm1'], arm['lines_cm1'], flush=True)
        report['radial_EA_drift_cm1'] = np.subtract(
            report['arms'][1]['EA_cm1'], report['arms'][0]['EA_cm1']).tolist()
        report['radial_line_drift_cm1'] = np.subtract(
            report['arms'][1]['lines_cm1'], report['arms'][0]['lines_cm1']).tolist()
        gate('predeclared_two_basis_drift', max(abs(x) for x in
            report['radial_EA_drift_cm1'] + report['radial_line_drift_cm1']),
            a['radial_EA_and_lines_drift_cm1'])
        gate('source_unchanged', source_map() == identity['all_atomx_python_sha256'])
        report['status'] = 'PASS_NATIVE_DHF_BASELINE_ONLY'
    except Exception:
        report['status'] = 'FAILED'
        report['exception'] = traceback.format_exc()
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds'] = time.monotonic()-start
        report['peak_RSS_MiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        save(out / 'result.json', report)
        print('END', report['status'], report['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run()
