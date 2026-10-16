"""Source-replayable Sn12+ n=4 CI baseline for the published EUV problem.

The fold is the number of adjacent p->d / d->f promotions:
    m = (6 - N_4p) + N_4f.
The {6} paper space has 22 NR configurations, not the unrestricted eight-
electron p/d/f Fock space. Core MBPT, Breit and QED are NOT applied here.
This workflow composes native DHF, signed CSF CI and electric operators;
the resulting bare-DC calculation is a baseline, not a complete paper result.
"""
from collections import Counter, defaultdict
from math import comb, fsum

CORE = '1s2 2s2 2p6 3s2 3p6 3d10 4s2'
REFERENCE = CORE + ' 4p6 4d2'


def _integer(value, name, minimum=1, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, int) or value < minimum
            or (maximum is not None and value > maximum)):
        raise ValueError(f'{name} must be an integer in [{minimum}, {maximum}]')


def plan_tin_n4_ci(max_fold):
    """Count full J/parity spaces without constructing CSFs or Hamiltonians.

    Raw determinant dimensions are the exact fixed-M dimensions, obtained
    from all J multiplicities. Dense bytes describe H only, not total RSS.
    J-allowed E1 pairs are upper bounds, not nonzero calculated line counts.
    """
    from atomx.config.csf import count_j_resolved_csfs

    _integer(max_fold, 'max_fold', 0, 6)
    configs, blocks = [], Counter()
    for fold in range(max_fold + 1):
        for holes in range(7):
            nf, np_ = fold - holes, 6 - holes
            nd = 8 - np_ - nf
            if nf < 0 or nd < 0:
                continue
            config = ' '.join(f'4{s}{q}' for s, q in [('p', np_), ('d', nd), ('f', nf)] if q)
            counts = count_j_resolved_csfs(config)
            magnetic = comb(6, np_) * comb(10, nd) * comb(14, nf)
            if sum((j + 1)*n for j, n in counts.items()) != magnetic:
                raise RuntimeError('J multiplicities do not span the complete magnetic space')
            parity = (-1)**fold
            blocks.update({(j, parity): n for j, n in counts.items()})
            configs.append(dict(configuration=config, fold=fold, parity=parity,
                                occupations=[np_, nd, nf], J2_counts=dict(counts),
                                levels=sum(counts.values()), magnetic_states=magnetic))
    block_rows = [dict(J2=j, parity=p, levels=n,
                       raw_determinants=sum(k for (jj, pp), k in blocks.items() if pp == p and jj >= j))
                  for (j, p), n in sorted(blocks.items())]
    return dict(max_fold=max_fold, configurations=configs, blocks=block_rows,
                configuration_count=len(configs), levels=sum(blocks.values()),
                magnetic_states=sum(c['magnetic_states'] for c in configs),
                largest_J_block=max(blocks.values()),
                largest_raw_determinant_block=max(b['raw_determinants'] for b in block_rows),
                dense_block_H_bytes=8*sum(n*n for n in blocks.values()),
                dense_global_H_bytes=8*sum(blocks.values())**2,
                core_electrons=30, active_electrons=8, total_electrons=38)


def enumerate_tin_n4_csfs(max_fold, *, max_levels=2000, max_determinants=2000):
    """Enumerate the declared complete space, or reject before enumeration."""
    from atomx.config.csf import enumerate_jj_csfs

    _integer(max_levels, 'max_levels')
    _integer(max_determinants, 'max_determinants')
    plan = plan_tin_n4_ci(max_fold)
    if plan['levels'] > max_levels:
        raise ValueError(f"complete space has {plan['levels']} levels; cap={max_levels}")
    if plan['largest_raw_determinant_block'] > max_determinants:
        raise ValueError('complete fixed-M determinant block exceeds declared cap')
    csfs, folds = [], []
    for row in plan['configurations']:
        group = enumerate_jj_csfs(CORE + ' ' + row['configuration'])
        if len(group) != row['levels']:
            raise RuntimeError('enumerated CSFs disagree with independent J counts')
        csfs.extend(group)
        folds.extend([row['fold']] * len(group))
    return tuple(csfs), tuple(folds), plan


def _length_e1_matrix(bra_csfs, ket_csfs, orbitals, workspace, radial_cache, *, engine='batched',
                      max_determinants=5000, memory_bytes=512*1024**2,
                      bra_coefficients=None, ket_coefficients=None, projection_cache=None):
    """Signed reduced long-wavelength CSF operator, reused across its roots."""
    import numpy as np

    from atomx.angular.csf_hamiltonian import csf_signature, one_body_tensor_terms
    from atomx.properties.transitions.angular import angular_eL
    from atomx.properties.transitions.multipole import radial_eL

    if engine == 'batched':
        from atomx.angular.one_body_tensor import one_body_tensor_matrix
        bs = tuple(csf_signature(c) for c in bra_csfs)
        ks = tuple(csf_signature(c) for c in ket_csfs)
        all_keys = {(n, k) for s in bs+ks for n, k, _ in s.pattern}
        elements = {}
        for destination in all_keys:
            for source in all_keys:
                angular = angular_eL(source[1], destination[1], 1)
                if angular == 0:
                    continue
                key = (destination, source)
                if key not in radial_cache:
                    radial_cache[key] = angular*radial_eL(*orbitals[source], *orbitals[destination], workspace, 1)
                elements[key] = radial_cache[key]
        return one_body_tensor_matrix(bs, ks, 1, elements, max_determinants=max_determinants,
                                       memory_bytes=memory_bytes, bra_coefficients=bra_coefficients,
                                       ket_coefficients=ket_coefficients, projection_cache=projection_cache)
    if engine != 'pair':
        raise ValueError('explicit pair or batched tensor engine required')
    matrix = np.zeros((len(bra_csfs), len(ket_csfs)))
    for i, a in enumerate(bra_csfs):
        for j, b in enumerate(ket_csfs):
            for destination, source, coefficient in one_body_tensor_terms(csf_signature(a), csf_signature(b), 1):
                key = (destination, source)
                if key not in radial_cache:
                    radial_cache[key] = angular_eL(source[1], destination[1], 1) * radial_eL(
                        *orbitals[source], *orbitals[destination], workspace, 1)
                matrix[i, j] += coefficient * radial_cache[key]
    if bra_coefficients is not None:
        matrix = bra_coefficients.conjugate().T @ matrix
    if ket_coefficients is not None:
        matrix = matrix @ ket_coefficients
    return matrix


def summarize_tin_lines(levels, lines, *, temperature_eV=32.0):
    """Keep gA centroids distinct from the paper's Boltzmann line weights.

    Families use explicitly reported dominant-fold state assignments. Full
    fold weights must also be inspected when bands mix strongly; this is
    not a nearest-energy match to a published target.
    """
    from math import exp, isfinite

    if not isfinite(temperature_eV) or temperature_eV <= 0:
        raise ValueError('positive finite temperature_eV required')
    by_id = {level['id']: level for level in levels}
    if len(by_id) != len(levels):
        raise ValueError('distinct state identities required')
    groups = defaultdict(list)
    for line in lines:
        upper, lower = by_id[line['upper']], by_id[line['lower']]
        energy, rate = line['energy_eV'], line['A_per_s']
        if not isfinite(energy) or energy <= 0 or not isfinite(rate) or rate < 0:
            raise ValueError('positive photon and finite nonnegative A required')
        gu = upper['J2'] + 1
        family = f"{upper['dominant_fold']}_to_{lower['dominant_fold']}"
        ga = gu * rate
        boltzmann = ga * exp(-upper['excitation_eV']/temperature_eV)
        groups[family].append((energy, ga, boltzmann))
    return {name: dict(lines=len(rows), gA_sum=fsum(r[1] for r in rows),
                       gA_centroid_eV=(fsum(r[0]*r[1] for r in rows)/fsum(r[1] for r in rows)
                                       if fsum(r[1] for r in rows) else None),
                       Boltzmann_gA_sum=fsum(r[2] for r in rows),
                       temperature_eV=temperature_eV)
            for name, rows in sorted(groups.items())}


def run_tin_n4_ci(max_fold=1, *, ns=90, ks=9, r_max=60.0,
                  max_levels=2000, max_determinants=2000, tensor_engine='batched', progress=None,
                  storage='global', emission_max_fold=None, memory_bytes=8*1024**3, line_sink=None,
                  ci_engine='streamed'):
    """Solve all roots and E1 lines of a bounded, explicitly declared DC model.

    Reference: statistical VN DHF [Kr]4d2, 120Sn Fermi nucleus. Both fine-
    structure partners of 4p,4d,4f are retained. Length E1 uses the explicitly
    declared long-wavelength approximation, allowing one CSF matrix per pair
    of J/parity blocks instead of rebuilding an operator for every line.
    """
    if storage == 'blocked':
        from atomx.workflows.tin_euv_blocked import run_tin_n4_ci_blocked
        return run_tin_n4_ci_blocked(max_fold, ns=ns, ks=ks, r_max=r_max,
                    max_levels=max_levels, max_determinants=max_determinants,
                    tensor_engine=tensor_engine, progress=progress,
                    emission_max_fold=emission_max_fold, memory_bytes=memory_bytes,
                    line_sink=line_sink, ci_engine=ci_engine)
    if (storage != 'global' or emission_max_fold is not None or line_sink is not None
            or ci_engine != 'streamed'):
        raise ValueError('explicit supported storage/observable controls required')
    import time

    import numpy as np

    from atomx.config.csf import get_shared_orbital_basis
    from atomx.constants import HARTREE_EV
    from atomx.methods.ci import ci_solve
    from atomx.methods.dirac_fock import DFSolver
    from atomx.methods.mbpt import build_virtual_basis
    from atomx.properties.transitions.rates import einstein_A_eL, oscillator_strength

    csfs, folds, plan = enumerate_tin_n4_csfs(max_fold, max_levels=max_levels,
                                           max_determinants=max_determinants)
    if progress is None:
        def progress(*_):
            pass
    timings, started = {}, time.monotonic()
    progress('DHF_START', plan['levels'])
    solver = DFSolver.from_configuration(50, REFERENCE, ns=ns, ks=ks, r_max=r_max,
                                         kinetic_balance=True, nuclear='fermi',
                                         nuclear_mass_number=120, starting_potential='tfda')
    core = solver.solve(max_iter=100, tol=1e-10, orbital_tolerance=1e-7)
    if not core.converged:
        raise RuntimeError(f'Sn12+ statistical DHF unconverged: {core.residual}')
    timings['DHF_seconds'] = time.monotonic()-started
    orbitals = {k: tuple(v[:2]) for k, v in core.orbitals_by_subshell().items()}
    for key in get_shared_orbital_basis(csfs):
        if key not in orbitals:
            occupied, band = build_virtual_basis(core, key[1], e_max=0, potential_mode='VN',
                                                 orthogonal_to_occupied=True)
            if not band or key[0] != 4 or key[1] not in (3, -4):
                raise RuntimeError(f'no declared bound n4 virtual for {key}')
            orbitals[key] = tuple(band[0][:2])
    progress('CI_START', core.residual)
    stage = time.monotonic()
    keys = get_shared_orbital_basis(csfs)
    ci = ci_solve(csfs, orbitals, keys, core._bsp_data, 50, ns,
                  nuclear_potential=core._nuclear_potential, convol_cache=solver._convol_cache,
                  max_determinants=max_determinants, retain_hamiltonian=True)
    h = ci.hamiltonian_for(csfs, orbitals, keys, core._bsp_data, 50, ns,
                           nuclear_potential=core._nuclear_potential,
                           max_determinants=max_determinants)
    residual = np.max(np.abs(h @ ci.mixing_coefficients-ci.mixing_coefficients*ci.state_energies))
    gram = np.max(np.abs(ci.mixing_coefficients.T @ ci.mixing_coefficients-np.eye(len(csfs))))
    if residual > 1e-8 or gram > 1e-10:
        raise RuntimeError(f'full CI residual/Gram failed: {residual}, {gram}')
    timings['CI_seconds'] = time.monotonic()-stage
    groups = defaultdict(list)
    for i, csf in enumerate(csfs):
        groups[csf.total_J2, csf.parity].append(i)
    root_groups = defaultdict(list)
    levels = []
    folds = np.asarray(folds)
    for root, energy in enumerate(ci.state_energies):
        vector = ci.mixing_coefficients[:, root]
        anchor = int(np.argmax(np.abs(vector)))
        sector = csfs[anchor].total_J2, csfs[anchor].parity
        if abs(np.vdot(vector[groups[sector]], vector[groups[sector]])-1) > 1e-10:
            raise RuntimeError('CI root does not belong to one exact J/parity sector')
        weights = [float(np.sum(np.abs(vector[folds == m])**2)) for m in range(max_fold+1)]
        state_id = f'J2={sector[0]};p={sector[1]};root={len(root_groups[sector])}'
        root_groups[sector].append(root)
        levels.append(dict(id=state_id, J2=sector[0], parity=sector[1], energy_Ha=float(energy),
                           excitation_eV=float((energy-ci.ground_energy)*HARTREE_EV),
                           dominant_fold=int(np.argmax(weights)), fold_weights=weights,
                           leading_csf=anchor, leading_csf_weight=float(abs(vector[anchor])**2)))
    progress('E1_START', len(levels))
    stage = time.monotonic()
    lines, operators, radial_cache = [], {}, {}
    sectors = sorted(groups)
    for ia, a in enumerate(sectors):
        for b in sectors[ia+1:]:
            if a[1] == b[1] or abs(a[0]-b[0]) > 2 or a[0]+b[0] < 2:
                continue
            ga, gb = groups[a], groups[b]
            op = _length_e1_matrix([csfs[i] for i in gb], [csfs[i] for i in ga],
                                   orbitals, core._bsp_data, radial_cache, engine=tensor_engine)
            ra, rb = root_groups[a], root_groups[b]
            amplitude = ci.mixing_coefficients[np.ix_(gb, rb)].T @ op @ ci.mixing_coefficients[np.ix_(ga, ra)]
            operators[f'bra_J2_{b[0]}_p_{b[1]}_ket_J2_{a[0]}_p_{a[1]}'] = op
            for ib, j in enumerate(rb):
                for ii, i in enumerate(ra):
                    upper, lower = (i, j) if ci.state_energies[i] > ci.state_energies[j] else (j, i)
                    omega = float(ci.state_energies[upper]-ci.state_energies[lower])
                    signed = float(amplitude[ib, ii])
                    if upper == j:
                        signed *= (-1)**((a[0]-b[0])//2)
                    strength = signed*signed
                    if omega == 0 or strength == 0:
                        continue
                    f_value, gf_value = oscillator_strength(omega, strength, 1,
                                                           levels[lower]['J2'], levels[upper]['J2'])
                    lines.append(dict(upper=levels[upper]['id'], lower=levels[lower]['id'],
                                      energy_eV=omega*HARTREE_EV,
                                      reduced_lower_upper_au=signed, line_strength_au=strength,
                                      A_per_s=float(einstein_A_eL(omega, strength, 1, levels[upper]['J2'])),
                                      f_absorption=float(f_value), gf=float(gf_value)))
    timings['E1_seconds'] = time.monotonic()-stage
    timings['total_seconds'] = time.monotonic()-started
    summary = summarize_tin_lines(levels, lines)
    return dict(plan=plan, levels=levels, lines=lines, summary=summary,
                diagnostics=dict(DHF_converged=core.converged, DHF_iterations=core.iterations,
                                 DHF_residual_Ha=float(core.residual), DHF_energy_Ha=float(core.total_energy),
                                 CI_residual_Ha=float(residual), CI_Gram_defect=float(gram),
                                 timings=timings, operator='Dirac-Coulomb',
                                 core_correlation_applied=False, Breit_applied=False, QED_applied=False,
                                 E1='length, long_wavelength',
                                 tensor_engine=tensor_engine,
                                 state_assignment='maximum reported fold weight, not energy matching'),
                arrays=dict(H_Ha=h, energies_Ha=ci.state_energies,
                            coefficients=ci.mixing_coefficients, csf_folds=folds,
                            **{f'E1_{k}': v for k, v in operators.items()}),
                orbitals=orbitals, scf=core, csfs=csfs)
