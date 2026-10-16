"""Same literal native31 DC/static/zero RCI; default is SOURCE/JSON only."""
import argparse
import hashlib
import inspect
import json
import math
import os
import resource
import signal
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'AGENTS.md').is_file())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, 'dtype') and getattr(value, 'shape', None) == ():
        return int(value) if value.dtype.kind in 'iu' else bool(value) if value.dtype.kind == 'b' else float(value)
    if hasattr(value, 'tolist'):
        return plain(value.tolist())
    return value


def write(path, value):
    temp = path.with_name(path.name + '.part')
    temp.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def metadata(*, array_hashes=False):
    contract = read(HERE / 'contract.json')
    roles = {k: ROOT / v['path'] for k, v in contract['input_roles'].items()}
    for name, row in contract['input_roles'].items():
        if (roles[name].suffix != '.npz' or array_hashes) and sha(roles[name]) != row['sha256']:
            raise ValueError('Frozen input changed: ' + name)
    source = ROOT / contract['source_archive']
    identity = read(roles['source_identity'])
    if identity['source_map_sha256'] != contract['source_map_sha256'] or len(identity['all_source_sha256']) != 365:
        raise ValueError('Wrong complete365 source epoch')
    for name, expected in identity['all_source_sha256'].items():
        if sha(source / name.removeprefix('src/')) != expected:
            raise ValueError('Frozen source changed: ' + name)
    for name in ('primary_receipt', 'integration_receipt', 'matched_receipt'):
        receipt = read(roles[name])
        if receipt['status'] != 'foreground_closed' or receipt['exit_code'] != 0:
            raise ValueError('Required independent closed control failed: ' + name)
    matched = read(roles['matched_result'])
    if not matched['same_epoch_H_and_signed_gradient_bitwise'] or not matched['DC_schema1_header_equal']:
        raise ValueError('Matched owner baseline is not qualified')
    if matched['original_mixed_suite_passed'] or matched['optimizer_authorized']:
        raise ValueError('Original failure boundary changed')
    if read(roles['mixed_failed_receipt'])['exit_code'] != 1:
        raise ValueError('Original mixed failure must remain archived')
    original = read(roles['actual/result.json'])
    if not original['all_declared_controls_passed'] or original['expanded_MCDHF_optimized']:
        raise ValueError('Literal recovered31 prerequisite changed')
    return contract, roles, source


def authorize(binding_path):
    contract, roles, source = metadata()
    binding = read(binding_path)
    expected = dict(execution_authorized_by_root=True, contract_sha256=sha(HERE / 'contract.json'),
                    driver_sha256=sha(__file__), preparation_sha256=sha(HERE / 'preparation.json'),
                    source_map_sha256=contract['source_map_sha256'],
                    input_roles=contract['input_roles'], resources=contract['resources'])
    if any(binding.get(k) != v for k, v in expected.items()):
        raise ValueError('Separate exact root execution binding required')
    prep = read(HERE / 'preparation.json')
    if not prep['all_SOURCE_JSON_checks_passed'] or prep['driver_sha256'] != sha(__file__) or prep['contract_sha256'] != sha(HERE / 'contract.json'):
        raise ValueError('Stale SOURCE-only preparation')
    return contract, roles, source


def execute(binding_path):
    contract, roles, source = authorize(binding_path)
    out = HERE / 'actual'
    out.mkdir(exist_ok=False)
    caps, fixed = contract['resources'], contract['fixed']
    checks, points, archives = [], {}, []
    counters = dict.fromkeys(caps['calls'], 0)
    returned = dict(counters)
    hS = dict(requests=0, hits=0, misses=0, by_model={}, carry_events=[])
    model, raw = [None], {}
    started = time.monotonic()

    def progress(phase):
        write(out / 'progress.json', dict(phase=phase, entered=counters, returned=returned,
              checks=len(checks), last_check=checks[-1] if checks else None,
              model=model[0], completed_models=list(points), hS=hS,
              elapsed_uncontrolled_s=time.monotonic()-started))

    try:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Declared1200s cap')))
        signal.alarm(caps['inner_seconds'])
        if sys.executable != caps['interpreter_literal']:
            raise ValueError('Use the bound .venv interpreter')
        threads = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                   'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
        if any(os.environ.get(k) != '1' for k in threads):
            raise ValueError('All thread variables must be1 before numerical imports')
        affinity = sorted(os.sched_getaffinity(0))
        memory = next(int(r.split()[1]) for r in Path('/proc/meminfo').read_text().splitlines()
                      if r.startswith('MemAvailable:')) / 1024**2
        admission = dict(original_affinity=affinity, CPU=caps['CPU'], load=os.getloadavg(),
                         available_GiB=memory, controlled_timing=False)
        write(out / 'resource_admission.json', admission)
        if caps['CPU'] not in affinity or admission['load'][0] > caps['maximum_system_load1'] or memory < caps['minimum_host_available_GiB']:
            raise RuntimeError('Host resource admission failed')
        os.sched_setaffinity(0, {caps['CPU']})
        maximum = int(caps['AS_GiB']*2**30)
        resource.setrlimit(resource.RLIMIT_AS, (maximum, maximum))
        metadata(array_hashes=True)
        if any(k == 'atomx' or k.startswith('atomx.') for k in sys.modules):
            raise ValueError('A fresh immutable-source process is required')
        (out / 'numba_cache').mkdir()
        os.environ['NUMBA_CACHE_DIR'] = str(out / 'numba_cache')
        sys.path.insert(0, str(source))
        import numpy as np
        from scipy import linalg

        from atomx.angular.csf_hamiltonian import csf_signature
        from atomx.config.csf import enumerate_jj_csfs, get_shared_orbital_basis
        from atomx.constants import HARTREE_CM1
        from atomx.methods import ci as ci_module
        from atomx.methods.csf.variational import CSFOrbitalFunctional
        from atomx.numerics.bspline.convol import ConvolCache
        from atomx.numerics.bspline.dirac_assembly import (
            build_dirac_matrices,
            solve_dirac,
        )
        from atomx.numerics.bspline.orbital_optimization import (
            _polar_retraction,
            _residual_metric_sha256,
            optimize_orbitals,
        )
        from atomx.numerics.helmholtz import HelmholtzKernel
        from atomx.physics.csf import matrix, tracking
        from atomx.physics.csf.hamiltonian_snapshot import (
            CIHamiltonianSnapshot,
            build_ci_hamiltonian_snapshot,
        )
        from atomx.physics.csf.levels import ci_level_blocks
        from atomx.physics.csf.state_average import CIStateAverage
        from atomx.potentials.nuclear import make_nuclear_potential

        def check(name, error, limit=0.):
            row = dict(name=name, error=float(error), limit=limit,
                       passed=bool(np.isfinite(error) and error <= limit))
            checks.append(row)
            progress(name)
            if not row['passed']:
                raise ValueError('Original gate failed: ' + name)

        def save(name, **arrays):
            path = out / name
            if path.exists():
                raise FileExistsError(name)
            np.savez_compressed(path, **arrays)
            archives.append(dict(path=str(path.relative_to(HERE)), sha256=sha(path)))

        def source_guard():
            for name, module in tuple(sys.modules.items()):
                if name == 'atomx' or name.startswith('atomx.'):
                    filename = getattr(module, '__file__', None)
                    if filename and source not in Path(filename).resolve().parents:
                        raise ValueError('Source escaped archive: ' + name)

        # Pure observation of original function bodies; no function replacement.
        funcs = dict(H=matrix.build_ci_matrix, CI=ci_module.ci_solve,
            owner_build=build_ci_hamiltonian_snapshot, owner_matrix_read=CIHamiltonianSnapshot.matrix_for,
            CI_internal_hS=build_dirac_matrices, tracker_resolve=tracking.CIBasisTracking.resolve,
            tracker_overlap=tracking.ci_subspace_overlaps, tracking_SVD=np.linalg.svd._implementation,
            nucleus_factory=make_nuclear_potential, convol_cache=ConvolCache.__init__,
            transverse_factory=matrix._breit_radial_evaluator, zero_Helmholtz_kernels=HelmholtzKernel.__init__,
            Dirac=solve_dirac, gradient=CSFOrbitalFunctional.gradient_from_weights,
            optimizer=optimize_orbitals, polar=_polar_retraction)
        codes = {inspect.unwrap(f).__code__: key for key, f in funcs.items()}
        for f in (linalg.eigh, np.linalg.eigh._implementation):
            codes[inspect.unwrap(f).__code__] = 'complete_eigh'
        active, getters, resets, transverse = {}, {}, {}, {}
        coverage = {name: set() for name in contract['operators']}

        def profile(frame, event, value):
            token, code = id(frame), frame.f_code
            key = codes.get(code)
            if event == 'call':
                if frame.f_globals.get('__name__') == 'atomx.methods.dirac_fock' and code.co_name in ('solve', '_build_hf_matrix_dbsr', '_build_hf_matrix'):
                    key = 'Fock'
                if key is not None:
                    counters[key] += 1
                    if counters[key] > caps['calls'][key]:
                        raise RuntimeError('Primitive cap: ' + key)
                    active[token] = key
                    if key == 'zero_Helmholtz_kernels' and (model[0] != 'zero' or frame.f_locals['wave_number'] != 0. or not 0 <= frame.f_locals['order'] <= 8):
                        raise ValueError('Only declared zero kernels/ranks are admitted')
                if code is matrix._dirac_blocks.__code__:
                    hS['requests'] += 1
                    if hS['requests'] > caps['maximum_hS_getter_requests']:
                        raise RuntimeError('h/S getter cap')
                    k = int(frame.f_locals['kappa'])
                    coverage[model[0]].add(k)
                    hit = ('h0', 123, k, 70) in matrix._SLATER_MEMO and matrix._MEMO_CONTEXT is frame.f_locals['bsp_data']
                    getters[token] = (k, hit)
                if code is matrix._reset_memo_for_ci_build.__code__:
                    resets[token] = ({k: v for k, v in matrix._SLATER_MEMO.items()
                                      if isinstance(k, tuple) and k and k[0] == 'h0'},
                                     matrix._MEMO_CONTEXT is frame.f_locals['bsp_data'],
                                     matrix._SLATER_MEMO.get('_nuclear_potential') is frame.f_locals['nuclear_potential'])
            elif event == 'return':
                if token in active:
                    returned[active.pop(token)] += 1
                    if key == 'transverse_factory' and value is not None:
                        codes[inspect.unwrap(value).__code__] = 'transverse_uncached_evaluations'
                        transverse[model[0]] = value
                    if key == 'CI_internal_hS' and value is not None:
                        raw[int(frame.f_locals['kappa'])] = tuple(a.copy() for a in value)
                if token in getters and value is not None:
                    _, hit = getters.pop(token)
                    hS['hits' if hit else 'misses'] += 1
                if token in resets:
                    previous, same_workspace, same_nucleus = resets.pop(token)
                    carried = {k: v for k, v in matrix._SLATER_MEMO.items()
                               if isinstance(k, tuple) and k and k[0] == 'h0'}
                    hS['carry_events'].append(dict(model=model[0], same_workspace=same_workspace,
                        same_nucleus=same_nucleus, previous=len(previous), carried=len(carried),
                        same_tuple_objects=all(k in previous and v is previous[k] for k, v in carried.items()),
                        nuclear_geometry_fingerprint=matrix._MEMO_GEOMETRY.hex()))

        sys.setprofile(profile)

        def archive(role):
            counters['NPZ_load'] += 1
            if counters['NPZ_load'] > caps['calls']['NPZ_load']:
                raise RuntimeError('NPZ load cap')
            with np.load(roles[role], allow_pickle=False) as a:
                result = {k: a[k].copy() for k in a.files}
            returned['NPZ_load'] += 1
            check('finite_saved_' + role, not all(np.isfinite(v).all() and not np.iscomplexobj(v) for v in result.values()))
            return result

        point, old, metric_archive, radial, old_hS, reference = [archive(role) for role in
            ('literal31_recovered_point', 'literal31_DC_H_E_C', 'original31_fullmetric',
             'geometry_workspace', 'screened_geometry', 'tracking_ci')]
        keys = tuple(map(tuple, fixed['keys31']))
        waves = {k: (p.copy(), q.copy()) for k, p, q in zip(keys, point['P'], point['Q'], strict=True)}
        check('point_keys', tuple(map(tuple, point['keys'])) != keys)
        check('oldCI_same_literal_keys', not np.array_equal(old['keys'], point['keys']))
        check('oldCI_same_literal_PQ', not (np.array_equal(old['P'], point['P']) and np.array_equal(old['Q'], point['Q'])))
        data = {k[6:]: v.item() if v.ndim == 0 else v.copy() for k, v in radial.items() if k.startswith('basis_')}
        layouts = read(roles['layout'])
        original_layout = [(tuple(map(tuple, r['keys'])), point[f'projector_{i}'], tuple(r['shape']), slice(*r['slice']))
                           for i, r in enumerate(layouts)]
        inverse = [tuple(point[f'inverse_blocks_{i}']) for i in range(7)]
        check('bound3689_metric', _residual_metric_sha256(original_layout, point['inverse_diagonal'], inverse)
              != contract['bound3689_metric_sha256'])
        check('bound3689_coordinates', not (np.array_equal(point['x'], metric_archive['x']) and point['x'].shape == (3689,)))
        metrics = {k: old_hS[f's_{k}'] for k in fixed['kappas']}
        for i, (group, projector, shape, where) in enumerate(original_layout):
            s = metrics[group[0][1]]
            check('projector_bytes_' + str(i), not np.array_equal(projector, metric_archive[f'projector_{i}']))
            check('projector_S_Gram_' + str(i), np.max(abs(projector.T @ s @ projector-np.eye(shape[0]))), 1e-10)
            C = np.column_stack([np.concatenate(waves[k]) for k in group])
            check('samek_radial_Gram_' + str(i), np.max(abs(C.T @ s @ C-np.eye(len(group)))), 1e-10)
            check('literal_originalP_x_' + str(i), np.max(abs(C-projector @ point['x'][where].reshape(shape))), 0.)
            leak = C-projector @ (projector.T @ s @ C)
            check('originalP_leakage_squared_' + str(i), np.max(np.sum(leak*(s @ leak), axis=0)), 1e-10)

        models = read(roles['models'])
        sectors = set(map(tuple, models['selected_sectors']))
        full = [c for row in models['models']['complete5f']['configs'] for c in enumerate_jj_csfs(row['config'])]
        csfs = [c for c in full if (c.total_J2, c.parity) in sectors]
        check('full_parent3789', len(full) != 3789)
        check('complete_target1595', len(csfs) != 1595)
        check('five_complete_sectors', Counter((c.total_J2, c.parity) for c in csfs)
              != {(r['twoJ'], r['parity']): r['CSFs'] for r in fixed['five_complete_target_blocks']})
        check('all_parent_charge69', {c.n_electrons for c in full} != {69})
        signatures = [json.dumps(plain((csf_signature(c).subshells, csf_signature(c).coupling, c.total_J2, c.parity)), separators=(',', ':')) for c in csfs]
        check('whole_signed_basis_order', signatures != read(roles['canonical_basis'])['complete5f']['retained_signatures'])
        check('public_all31_basis', tuple(sorted(get_shared_orbital_basis(csfs))) != keys)
        reference_csfs = [c for row in read(roles['reference_models'])['models']['baseline']['configs']
                          for c in enumerate_jj_csfs(row['config'])]
        check('reference531', len(reference_csfs) != 531)
        reference_waves = {tuple(map(int, k)): (p, q) for k, p, q in zip(radial['orbital_keys'], radial['P'], radial['Q'], strict=True)}
        targets = read(roles['tracking_binding'])['targets']
        average = CIStateAverage(tuple((t['twoJ'], t['parity'], t['models']['DC']['local_root'], 1.) for t in targets))
        tracker = tracking.CIBasisTracking(average, reference_csfs,
                  ci_module.CIResult(reference['energies_Ha'], reference['mixing']), reference_waves,
                  metrics, max_determinant_pairs=30000)
        nucleus = make_nuclear_potential(70, 'fermi', A=175)
        cache = ConvolCache(data)
        check('cache_same_grid', not cache.same_grid(data))
        original_PQ = tuple(tuple(a.tobytes() for a in waves[k]) for k in keys)
        nominal = {r['label']: r['NIST_nominal_cm1'] for r in read(roles['NIST_nominal_source'])['rows']}
        expected = {tuple(uid): label for label, uid in fixed['target_UIDs']}
        check('original_five_target_contract', set(expected) != {(t['twoJ'], t['parity'], t['models']['DC']['local_root']) for t in targets})
        old_assignment = {tuple(r['reference_levels'][0]): tuple(r['selected_levels'][0])
                          for r in read(roles['actual/recovered_tracking.json'])['tracked']}
        old_ci = ci_module.CIResult(old['energies'], old['mixing'])
        _, old_blocks = ci_level_blocks(csfs, old_ci, require_full=True)
        all_spectra = {}
        scalar_calls = {}

        for name in ('DC', 'static', 'zero'):
            model[0] = name
            before = {k: hS[k] for k in ('requests', 'hits', 'misses')}
            include = name != 'DC'
            omega = {k: 0.0 for k in keys} if name == 'zero' else None
            result = ci_module.ci_solve(csfs, waves, keys, data, 70, 123, nuclear_potential=nucleus,
                     convol_cache=cache, solver='dense', n_states=None, max_determinants=2000,
                     retain_hamiltonian=True, include_breit=include, breit_orbital_energies=omega)
            owner = result.hamiltonian_snapshot
            H = result.hamiltonian_for(csfs, waves, keys, data, 70, 123,
                 nuclear_potential=nucleus, max_determinants=2000,
                 include_breit=include, breit_orbital_energies=omega)
            save(name+'_whole_CI.npz', H=H, energies=result.state_energies, mixing=result.mixing_coefficients,
                 keys=point['keys'], P=point['P'], Q=point['Q'])
            write(out / (name+'_owner.json'), owner.identity_payload())
            hS['by_model'][name] = {k: hS[k]-v for k, v in before.items()}
            check(name+'/all_seven_hS_coverage', coverage[name] != set(fixed['kappas']))
            check(name+'/whole_H_shape', H.shape != (1595, 1595))
            check(name+'/whole_H_symmetry', not np.array_equal(H, H.T))
            residual = np.linalg.norm(H @ result.mixing_coefficients-result.mixing_coefficients*result.state_energies, axis=0)
            gram = float(np.max(abs(result.mixing_coefficients.T @ result.mixing_coefficients-np.eye(1595))))
            save(name+'_full_spectrum_controls.npz', residuals_Ha=residual, Gram=np.array(gram))
            check(name+'/all1595_residuals_Ha', residual.max(), 1e-8)
            check(name+'/all1595_CI_Gram', gram, 1e-10)
            for sector in sectors:
                inside = [i for i, c in enumerate(csfs) if (c.total_J2, c.parity) == sector]
                outside = [i for i, c in enumerate(csfs) if (c.total_J2, c.parity) != sector]
                check(name+'/cross_sector_'+str(sector), np.max(abs(H[np.ix_(inside, outside)])), 0.)
            if name == 'DC':
                check('fresh_DC_exact_whole_saved_H', not (np.array_equal(H, old['H']) and H.tobytes() == old['H'].tobytes()))
                check('fresh_DC_exact_schema1_header', plain(owner.identity_payload()) != read(roles['actual/recovered_owner.json']))
                check('saved_old_all1595_residuals', np.max(np.linalg.norm(H @ old['mixing']-old['mixing']*old['energies'], axis=0)), 1e-8)
                check('saved_old_all1595_Gram', np.max(abs(old['mixing'].T @ old['mixing']-np.eye(1595))), 1e-10)
            check(name+'/literal_all31_PQ_bytes', tuple(tuple(a.tobytes() for a in waves[k]) for k in keys) != original_PQ)
            for k in fixed['kappas']:
                h, s = raw[k]
                check(name+'/logical_h_'+str(k), not np.array_equal(np.tril(h)+np.tril(h, -1).T, old_hS[f'h_{k}']))
                check(name+'/original_S_'+str(k), not np.array_equal(s, metrics[k]))
            weights, tracked = tracker.resolve(csfs, result, waves, metrics)
            check(name+'/five_equal_weights', not (np.count_nonzero(weights) == 5 and np.array_equal(weights[weights > 0], np.repeat(.2, 5))))
            assigned = {}
            for r in tracked:
                if len(r['reference_levels']) != 1 or len(r['selected_levels']) != 1:
                    raise ValueError('Complete target degeneracy is outside this five-singleton contract')
                uid, selected = tuple(r['reference_levels'][0]), tuple(r['selected_levels'][0])
                if uid in assigned or uid not in expected or uid[:2] != selected[:2]:
                    raise ValueError('Changed physical UID assignment')
                assigned[uid] = selected
            check(name+'/original_UID_assignment', assigned != old_assignment)
            _, blocks = ci_level_blocks(csfs, result, require_full=True)
            rows = []
            for label, uid in fixed['target_UIDs']:
                j, p, rank = assigned[tuple(uid)]
                column = int(blocks[j, p][rank])
                c = result.mixing_coefficients[:, column].astype(np.longdouble)
                q = float((c @ H.astype(np.longdouble) @ c)/(c @ c))
                rows.append(dict(label=label, UID=uid, selected_level=(j, p, rank), column=column,
                                 weight=float(weights[column]), energy_Ha=float(result.state_energies[column]),
                                 normalized_Rayleigh_Ha=q, reported_minus_Rayleigh_Ha=float(result.state_energies[column])-q))
            lower = next(r for r in rows if r['label'] == '6s_1/2')
            for r in rows:
                r.update(excitation_cm1=(r['energy_Ha']-lower['energy_Ha'])*HARTREE_CM1,
                         Rayleigh_excitation_cm1=(r['normalized_Rayleigh_Ha']-lower['normalized_Rayleigh_Ha'])*HARTREE_CM1,
                         NIST_nominal_cm1=nominal[r['label']], NIST_uncertainty_cm1=None)
                r['nominal_gap_cm1'] = r['excitation_cm1']-r['NIST_nominal_cm1']
                r['nominal_relative_gap'] = r['nominal_gap_cm1']/r['NIST_nominal_cm1'] if r['NIST_nominal_cm1'] else None
            by_label = {r['label']: r for r in rows}
            multiplets = []
            for label, a, b, degeneracies in (('5d', '5d_3/2', '5d_5/2', (4, 6)), ('6p', '6p_1/2', '6p_3/2', (2, 4))):
                centroid = sum(g*by_label[k]['excitation_cm1'] for g, k in zip(degeneracies, (a, b), strict=True))/sum(degeneracies)
                splitting = by_label[b]['excitation_cm1']-by_label[a]['excitation_cm1']
                multiplets.append(dict(label=label, degeneracy_weights=degeneracies, centroid_cm1=centroid, splitting_cm1=splitting))
            points[name] = dict(rows=rows, multiplets=multiplets, EOL_reported_Ha=float(weights @ result.state_energies),
                EOL_normalized_Rayleigh_Ha=math.fsum(.2*r['normalized_Rayleigh_Ha'] for r in rows),
                nominal_four_gap_RMS_cm1=math.sqrt(sum(r['nominal_gap_cm1']**2 for r in rows if r['nominal_relative_gap'] is not None)/4),
                nominal_four_1percent_goal_passed=all(abs(r['nominal_relative_gap'])<=.01 for r in rows if r['nominal_relative_gap'] is not None),
                original_tracker_not_rebased=True, operator_terms=owner.hamiltonian_terms)
            write(out / (name+'_tracking.json'), dict(weights=weights, tracked=tracked))
            all_spectra[name] = (H, result, blocks)
            if name == 'DC':
                cluster_rows, signed_arrays = [], {}
                for sector, old_columns in old_blocks.items():
                    new_columns = blocks[sector]
                    boundaries = [0]+[i for i in range(1, len(old_columns)) if old['energies'][old_columns[i]]-old['energies'][old_columns[i-1]]>1e-8]+[len(old_columns)]
                    for lo, hi in zip(boundaries[:-1], boundaries[1:], strict=True):
                        overlap = old['mixing'][:, old_columns[lo:hi]].T @ result.mixing_coefficients[:, new_columns[lo:hi]]
                        tag = str(sector)+'_'+str(lo)+'_'+str(hi)
                        signed_arrays[tag] = overlap
                        cluster_rows.append(dict(sector=sector, roots=(lo, hi), complete_span_size=hi-lo,
                            signed_overlap_array=tag, projected_Gram_defect=float(np.max(abs(overlap.T @ overlap-np.eye(hi-lo)))),
                            maximum_reported_energy_offset_Ha=float(np.max(abs(result.state_energies[new_columns[lo:hi]]-old['energies'][old_columns[lo:hi]])))))
                save('DC_old_complete_signed_clusters.npz', **signed_arrays)
                write(out / 'DC_old_complete_signed_clusters.json', dict(complete_all1595_spans=cluster_rows, diagnostic_only=True,
                      no_posthoc_vector_rotation=True, no_extra_eigen_or_SVD=True))
            if name == 'zero':
                diagnostics = transverse[name].diagnostics()
                scalar_calls[name] = diagnostics['caches']['values']['hits']+diagnostics['caches']['values']['misses']
                check('zero_actual_scalar_request_cap', scalar_calls[name], caps['maximum_transverse_scalar_requests'])
                write(out / 'zero_transverse_cache_diagnostics.json', diagnostics)
                check('static_zero_full_H_max_Ha', np.max(abs(H-all_spectra['static'][0])), 1e-9)
                np.fill_diagonal(z := H-all_spectra['static'][0], 0.)
                write(out / 'static_zero_whole_H_limit.json', dict(maximum_all_entries_Ha=float(np.max(abs(H-all_spectra['static'][0]))),
                      maximum_offdiagonal_Ha=float(np.max(abs(z))), limit_Ha=1e-9, independently_implemented_zero_limit=True))
            progress('completed_'+name)

        save('actual_raw_hS_same_geometry.npz', **{f'{component}_{k}': array for k, pair in raw.items() for component, array in zip(('raw_h', 'S'), pair, strict=True)})
        for new_name, reference_name in (('static', 'DC'), ('zero', 'DC'), ('zero', 'static')):
            new, base = points[new_name], points[reference_name]
            reference_rows = {r['label']: r for r in base['rows']}
            reference_lower = reference_rows['6s_1/2']
            new_lower = next(r for r in new['rows'] if r['label']=='6s_1/2')
            delta_lower = new_lower['energy_Ha']-reference_lower['energy_Ha']
            rows = [dict(label=r['label'], UID=r['UID'], delta_lower_Ha=delta_lower,
                         delta_upper_Ha=r['energy_Ha']-reference_rows[r['label']]['energy_Ha'],
                         transition_delta_upper_minus_lower_cm1=((r['energy_Ha']-reference_rows[r['label']]['energy_Ha'])-delta_lower)*HARTREE_CM1)
                    for r in new['rows']]
            multiplets = [dict(label=m['label'], delta_centroid_cm1=m['centroid_cm1']-o['centroid_cm1'],
                         delta_splitting_cm1=m['splitting_cm1']-o['splitting_cm1']) for m, o in zip(new['multiplets'], base['multiplets'], strict=True)]
            write(out / (new_name+'_minus_'+reference_name+'.json'), dict(same_literal_radials=True, rows=rows, multiplets=multiplets,
                  EOL_delta_Ha=new['EOL_reported_Ha']-base['EOL_reported_Ha'], no_old_epoch_correction_splicing=True))
        sys.setprofile(None)
        source_guard()
        metadata(array_hashes=True)
        for key, expected_count in contract['expected_exact_completed'].items():
            check('expected_completed_'+key, returned[key] != expected_count)
        check('raw_hS_successful_returns_equal_misses', returned['CI_internal_hS'] != hS['misses'])
        check('all_hS_getters_successful', hS['requests'] != hS['hits']+hS['misses'])
        for key, count in counters.items():
            check('entered_returned_'+key, count != returned[key])
        write(out / 'result.json', dict(status='closed_native31_fixed_RCI_phaseA', all_declared_controls_passed=True,
            checks=checks, entered=counters, returned=returned, hS=hS, points=points, archives=archives,
            source_map_sha256=contract['source_map_sha256'], contract_sha256=sha(HERE/'contract.json'), driver_sha256=sha(__file__),
            execution_binding_sha256=sha(binding_path), literal_recovered31=True, same_point_transverse_budget_measured=True,
            native_MCDHF_optimization_executed=False, original3689_stationarity_passed=False,
            original_mixed_suite_passed=False, finite_frequency_nonzero_executed=False, QED_executed=False,
            current_MCDHF_optimizer_qualified=False, global_spectrum_certified=False,
            elapsed_uncontrolled_s=time.monotonic()-started, peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            controlled_timing=False))
    except Exception as error:
        sys.setprofile(None)
        if raw:
            save('failure_completed_raw_hS.npz', **{f'{component}_{k}': array for k, pair in raw.items() for component, array in zip(('raw_h', 'S'), pair, strict=True)})
        write(out / 'failure.json', dict(type=type(error).__name__, message=str(error), checks=checks, entered=counters, returned=returned,
              hS=hS, completed_models=list(points), archives=archives, elapsed_uncontrolled_s=time.monotonic()-started,
              all_declared_controls_passed=False, current_optimizer_qualified=False, no_retry=True))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--authorize', type=Path)
    parser.add_argument('--execute', type=Path)
    args = parser.parse_args()
    if args.execute:
        execute(args.execute)
    elif args.authorize:
        authorize(args.authorize)
        print('Exact root binding accepted; 0 numerical imports, arrays or calls.')
    else:
        contract, _, _ = metadata()
        print(json.dumps(dict(status='SOURCE_JSON_only', source_map=contract['source_map_sha256'],
              deferred_NPZ_hashes=6, numerical_imports=0, physical_calls=0)))


if __name__ == '__main__':
    main()
