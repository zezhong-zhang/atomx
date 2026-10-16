"""Native complete4d2/4d/closed-core Sn partial PI/RR pilot, never full Sn CE."""

import argparse
import hashlib
import json
import resource
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from atomx.config.csf import enumerate_jj_csfs
from atomx.methods.ci_blocks import solve_ci_blocks
from atomx.methods.dirac_fock import DFSolver
from atomx.scattering.models import ContinuumModel
from atomx.scattering.photoionization.ci import ExplicitCIPhotoionization


def run(output, *, ns=90, nodes=4000, orders=(16, 32), temperatures=(20., 24., 32.)):
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    core_config = '1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6'
    solver = DFSolver.from_configuration(50, core_config+' 4d2', ns=ns, ks=9, r_max=60.,
                                        kinetic_balance=True, nuclear='fermi', nuclear_mass_number=120,
                                        starting_potential='tfda')
    core = solver.solve(max_iter=100, tol=1e-10, orbital_tolerance=1e-7)
    if not core.converged:
        raise RuntimeError('Sn common statistical VN reference unconverged')
    core.save(str(output/'scf.h5'))
    orbitals = {key: tuple(value[:2]) for key, value in core.orbitals_by_subshell().items()}
    sectors, arrays = {}, {}
    for valence in (0, 1, 2):
        csfs = tuple(enumerate_jj_csfs(core_config+(f' 4d{valence}' if valence else '')))
        spectrum = solve_ci_blocks(csfs, orbitals, core._bsp_data, 50, ns,
                                   nuclear_potential=core._nuclear_potential,
                                   memory_bytes=1024**3)
        sectors[36+valence] = csfs, spectrum
        for block in spectrum.blocks:
            token = f'N{36+valence}_J2{block.sector[0]}_p{block.sector[1]}'
            for name, value in (('H', block.hamiltonian_Ha), ('C', block.coefficients),
                                ('E', block.energies_Ha), ('indices', block.indices)):
                arrays[token+'_'+name] = value
    provider = ExplicitCIPhotoionization(sectors, orbitals, core._bsp_data, 50,
                nuclear_potential=core._nuclear_potential,
                continuum_model=ContinuumModel.shooting(n_grid=nodes, r_max=80., orbital_removal='none'),
                coverage='only complete [Kr]4d2, [Kr]4d1, [Kr]4d0 fine multiplets; all other daughter channels absent')
    print(json.dumps(dict(stage='CI_complete', levels=len(provider.levels), plan=provider.plan)), flush=True)
    pairs = [(i, f) for i, a in enumerate(provider.levels) for f, b in enumerate(provider.levels)
             if a.electrons == b.electrons+1]
    # Match only the declared absolute CI energies; no NIST replacement.
    for i, f in pairs:
        provider.threshold_eV(i, f)
    rows = []
    for T in temperatures:
        for order in orders:
            for i, f in pairs:
                moments = provider.recombination_moments(i, f, T, order=order)
                rows.append(dict(initial=i, final=f, T_eV=T, threshold_eV=provider.threshold_eV(i, f), **moments))
            print(json.dumps(dict(stage='RR', T_eV=T, order=order, cached_waves=len(provider._waves))), flush=True)
    # A small explicit PI excess-energy grid, retained per level pair.
    excess = np.array([.1, 1., 5., 10., 20., 40., 80., 160., 320.])
    for i, f in pairs:
        arrays[f'PI_i{i}_f{f}_photon_eV'] = provider.threshold_eV(i, f)+excess
        arrays[f'PI_i{i}_f{f}_sigma_cm2'] = provider.cross_section_cm2(i, f, provider.threshold_eV(i, f)+excess)
    ground = {n: min((i for i, level in enumerate(provider.levels) if level.electrons == n),
                     key=lambda i: provider.levels[i].energy_Ha) for n in sectors}
    comparison = []
    for n, reference in ((38, 258.), (37, 282.)):
        i, f = ground[n], ground[n-1]
        actual = provider.threshold_eV(i, f)
        comparison.append(dict(charge=50-n, native_IP_eV=actual, NIST_IP_eV=reference,
                               NIST_uncertainty_eV=3., NIST_origin='theoretical, parenthesized ASD value',
                               difference_eV=actual-reference, relative_difference=(actual-reference)/reference))
    relative = []
    if len(orders) > 1:
        for T in temperatures:
            for i, f in pairs:
                lo = next(row for row in rows if (row['initial'], row['final'], row['T_eV'], row['order']) == (i, f, T, orders[0]))
                hi = next(row for row in rows if (row['initial'], row['final'], row['T_eV'], row['order']) == (i, f, T, orders[-1]))
                for moment in ('alpha_cm3_s', 'photon_eV_cm3_s'):
                    if hi[moment] != 0:
                        relative.append(dict(T_eV=T, initial=i, final=f, moment=moment,
                                             relative_change=abs(hi[moment]-lo[moment])/hi[moment]))
    result = dict(model='native frozen-common-orbital complete4d CI partial E1 PI/RR',
                  source_id=provider.source_id, conditions=dict(ns=ns, ks=9, R_bohr=60., shooting_nodes=nodes,
                         shooting_R_bohr=80., isotope=120, Z=50, temperatures_eV=temperatures, orders=orders),
                  plan=provider.plan, coverage=provider.coverage,
                  SCF=dict(converged=core.converged, residual_Ha=float(core.residual), iterations=core.iterations,
                           energy_Ha=float(core.total_energy)),
                  CI={str(n): dict(levels=len(csfs), maximum_residual_Ha=max(b.residual_Ha for b in spec.blocks),
                                   maximum_gram_defect=max(b.gram_defect for b in spec.blocks))
                      for n, (csfs, spec) in sectors.items()},
                  levels=[asdict(level) for level in provider.levels], pairs=pairs, RR=rows,
                  quadrature_changes=relative, NIST_comparison=comparison,
                  maximum_Pauli_overlap=max(provider.projection_diagnostics, default=0.),
                  cached_waves=len(provider._waves), observed_wall_s=time.perf_counter()-start,
                  observed_peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
                  missing=['excited4p/4f configurations', 'core-hole daughter channels', 'orbital relaxation/correlation convergence',
                           'nonlocal continuum exchange and interchannel coupling', 'EII/DR/Auger/excitation data',
                           'population/transport/hydrodynamic calculation'],
                  full_Sn_kinetics_accepted=False, experimental_CE_accepted=False)
    np.savez_compressed(output/'native_arrays.npz', **arrays)
    result['native_arrays_sha256'] = hashlib.sha256((output/'native_arrays.npz').read_bytes()).hexdigest()
    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ns', type=int, default=90)
    parser.add_argument('--nodes', type=int, default=4000)
    parser.add_argument('--orders', type=int, nargs='+', default=[16, 32])
    parser.add_argument('--temperatures', type=float, nargs='+', default=[20., 24., 32.])
    args = parser.parse_args()
    result = run(args.output, ns=args.ns, nodes=args.nodes, orders=tuple(args.orders), temperatures=tuple(args.temperatures))
    print(json.dumps(dict(NIST=result['NIST_comparison'], wall_s=result['observed_wall_s'],
                          max_quad=max((r['relative_change'] for r in result['quadrature_changes']), default=0.))), flush=True)
