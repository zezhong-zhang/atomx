"""Complete CI mixing with bounded block storage and explicit emission window."""
import time
from collections import defaultdict
from math import exp, fsum

import numpy as np

from atomx.angular.one_body_tensor import TensorProjectionCache
from atomx.config.csf import get_shared_orbital_basis
from atomx.constants import HARTREE_EV
from atomx.methods.ci_blocks import solve_ci_blocks
from atomx.methods.dirac_fock import DFSolver
from atomx.methods.mbpt import build_virtual_basis
from atomx.properties.transitions.rates import einstein_A_eL, oscillator_strength
from atomx.workflows.tin_euv import (
    REFERENCE,
    _integer,
    _length_e1_matrix,
    enumerate_tin_n4_csfs,
)


def run_tin_n4_ci_blocked(max_fold, *, ns, ks, r_max, max_levels, max_determinants,
                          tensor_engine, progress, emission_max_fold, memory_bytes, line_sink,
                          ci_engine):
    """Keep every CI root; restrict emitted endpoints only by declared fold.

    The paper analyzes folds0..3 even when CI contains higher configurations.
    State assignment uses reported dominant configuration weights, never
    target energies. Ambiguous assignments are retained and counted.
    """
    if emission_max_fold is None:
        emission_max_fold = min(3, max_fold)
    _integer(emission_max_fold, 'emission_max_fold', 0, max_fold)
    if tensor_engine not in ('pair', 'batched'):
        raise ValueError('explicit tensor engine required')
    if progress is None:
        def progress(*_):
            pass
    csfs, folds, plan = enumerate_tin_n4_csfs(max_fold, max_levels=max_levels,
                                            max_determinants=max_determinants)
    started, timings = time.monotonic(), {}
    progress('DHF_START', plan['levels'])
    solver = DFSolver.from_configuration(50, REFERENCE, ns=ns, ks=ks, r_max=r_max,
                     kinetic_balance=True, nuclear='fermi', nuclear_mass_number=120,
                     starting_potential='tfda')
    core = solver.solve(max_iter=100, tol=1e-10, orbital_tolerance=1e-7)
    if not core.converged:
        raise RuntimeError(f'Sn12+ statistical DHF unconverged: {core.residual}')
    orbitals = {k: tuple(v[:2]) for k, v in core.orbitals_by_subshell().items()}
    for key in get_shared_orbital_basis(csfs):
        if key not in orbitals:
            _, band = build_virtual_basis(core, key[1], e_max=0., potential_mode='VN',
                                          orthogonal_to_occupied=True)
            if not band or key[0] != 4 or key[1] not in (3, -4):
                raise RuntimeError(f'no declared bound n4 virtual for {key}')
            orbitals[key] = tuple(band[0][:2])
    timings['DHF_seconds'] = time.monotonic()-started
    stage = time.monotonic()
    spectrum = solve_ci_blocks(csfs, orbitals, core._bsp_data, 50, ns,
                               nuclear_potential=core._nuclear_potential,
                               max_determinants=max_determinants, memory_bytes=memory_bytes,
                               progress=progress, engine=ci_engine,
                               spinor_reference_keys=((4,-2),(4,1)) if ci_engine == 'spinor' else None)
    timings['CI_seconds'] = time.monotonic()-stage
    blocks = {b.sector: b for b in spectrum.blocks}
    ground = min(float(b.energies_Ha[0]) for b in spectrum.blocks)
    levels, level_by_local, selected = [], {}, {}
    arrays = {'csf_folds': np.asarray(folds)}
    for b in spectrum.blocks:
        name = f'J2_{b.sector[0]}_p_{b.sector[1]}'
        arrays.update({f'{name}_H_Ha': b.hamiltonian_Ha,
                       f'{name}_coefficients': b.coefficients,
                       f'{name}_energies_Ha': b.energies_Ha,
                       f'{name}_csf_indices': b.indices})
        local_folds = np.asarray(folds)[b.indices]
        chosen = []
        for j, energy in enumerate(b.energies_Ha):
            v = b.coefficients[:, j]
            weights = [float(np.sum(v[local_folds == m]**2)) for m in range(max_fold+1)]
            anchor = int(np.argmax(np.abs(v)))
            level = dict(id=f'J2={b.sector[0]};p={b.sector[1]};root={j}',
                         J2=b.sector[0], parity=b.sector[1], energy_Ha=float(energy),
                         excitation_eV=(float(energy)-ground)*HARTREE_EV,
                         dominant_fold=int(np.argmax(weights)), fold_weights=weights,
                         leading_csf=int(b.indices[anchor]), leading_csf_weight=float(v[anchor]**2))
            level_by_local[b.sector, j] = level
            levels.append(level)
            if level['dominant_fold'] <= emission_max_fold:
                chosen.append(j)
        selected[b.sector] = chosen
        arrays[f'{name}_emission_roots'] = np.asarray(chosen, dtype=int)
    levels.sort(key=lambda r: r['energy_Ha'])
    progress('E1_START', len(levels), sum(map(len, selected.values())))
    stage = time.monotonic()
    lines = [] if line_sink is None else None
    total_lines, radial_cache, moments = 0, {}, defaultdict(list)
    projection_cache = TensorProjectionCache(min(memory_bytes//8, 2*1024**3))
    sectors = sorted(blocks)
    for ia, a in enumerate(sectors):
        for b in sectors[ia+1:]:
            ra, rb = selected[a], selected[b]
            if not ra or not rb or a[1] == b[1] or abs(a[0]-b[0]) > 2 or a[0]+b[0] < 2:
                continue
            ba, bb = blocks[a], blocks[b]
            progress('E1_BLOCK_START', a, b, len(ra), len(rb))
            projected = max_fold >= 4
            op = _length_e1_matrix([csfs[i] for i in bb.indices], [csfs[i] for i in ba.indices],
                                   orbitals, core._bsp_data, radial_cache, engine=tensor_engine,
                                   max_determinants=max_determinants,
                                   memory_bytes=memory_bytes//4,
                                   bra_coefficients=bb.coefficients[:, rb] if projected else None,
                                   ket_coefficients=ba.coefficients[:, ra] if projected else None,
                                   projection_cache=projection_cache)
            amplitude = op if projected else bb.coefficients[:, rb].T @ op @ ba.coefficients[:, ra]
            tag = f'bra_J2_{b[0]}_p_{b[1]}_ket_J2_{a[0]}_p_{a[1]}'
            arrays[f'line_amplitudes_{tag}'] = amplitude
            if max_fold <= 3:
                arrays[f'E1_{tag}'] = op
            block_moments = defaultdict(lambda: [[], [], []])
            for ib, j in enumerate(rb):
                lb = level_by_local[b, j]
                for ii, i in enumerate(ra):
                    la = level_by_local[a, i]
                    upper, lower = (la, lb) if la['energy_Ha'] > lb['energy_Ha'] else (lb, la)
                    omega = upper['energy_Ha']-lower['energy_Ha']
                    signed = float(amplitude[ib, ii])
                    if upper is lb:
                        signed *= (-1)**((a[0]-b[0])//2)
                    strength = signed*signed
                    if omega == 0 or strength == 0:
                        continue
                    rate = float(einstein_A_eL(omega, strength, 1, upper['J2']))
                    f_value, gf = oscillator_strength(omega, strength, 1, lower['J2'], upper['J2'])
                    energy = omega*HARTREE_EV
                    line = dict(upper=upper['id'], lower=lower['id'], energy_eV=energy,
                                reduced_lower_upper_au=signed, line_strength_au=strength,
                                A_per_s=rate, f_absorption=float(f_value), gf=float(gf))
                    if line_sink is None:
                        lines.append(line)
                    else:
                        line_sink(line)
                    total_lines += 1
                    family = f"{upper['dominant_fold']}_to_{lower['dominant_fold']}"
                    ga = (upper['J2']+1)*rate
                    group = block_moments[family]
                    group[0].append(ga)
                    group[1].append(ga*energy)
                    group[2].append(ga*exp(-upper['excitation_eV']/32.))
            for family, values in block_moments.items():
                moments[family].append((len(values[0]), *[fsum(v) for v in values]))
    timings['E1_seconds'] = time.monotonic()-stage
    timings['total_seconds'] = time.monotonic()-started
    summary = {f: dict(lines=sum(v[0] for v in parts), gA_sum=fsum(v[1] for v in parts),
                       gA_centroid_eV=fsum(v[2] for v in parts)/fsum(v[1] for v in parts),
                       Boltzmann_gA_sum=fsum(v[3] for v in parts), temperature_eV=32.)
               for f, parts in sorted(moments.items())}
    diagnostics = dict(DHF_converged=core.converged, DHF_iterations=core.iterations,
                       DHF_residual_Ha=float(core.residual), DHF_energy_Ha=float(core.total_energy),
                       CI_residual_Ha=max(b.residual_Ha for b in spectrum.blocks),
                       CI_Gram_defect=max(b.gram_defect for b in spectrum.blocks),
                       CI_storage='complete J/parity blocks; all roots retained',
                       CI_engine=ci_engine,
                       CI_resource_plan=spectrum.resource_plan, radial_snapshot=spectrum.radial_diagnostics,
                       timings=timings, operator='Dirac-Coulomb', core_correlation_applied=False,
                       Breit_applied=False, QED_applied=False, E1='length, long_wavelength',
                       tensor_engine=tensor_engine, emission_max_fold=emission_max_fold,
                       E1_early_root_projection=max_fold >= 4,
                       E1_projection_cache=projection_cache.diagnostics(),
                       emission_endpoint_count=sum(map(len, selected.values())),
                       state_assignment='maximum reported fold weight, not energy matching',
                       ambiguous_fold_states=sum(max(r['fold_weights']) < .5 for r in levels))
    return dict(plan=plan, levels=levels, lines=lines, line_count=total_lines, summary=summary,
                diagnostics=diagnostics, arrays=arrays, orbitals=orbitals, scf=core, csfs=csfs)
