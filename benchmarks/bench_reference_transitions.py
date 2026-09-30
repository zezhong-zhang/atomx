"""Live one-electron multipole audit with retained FAC inputs and outputs.

FAC mode 0 is compared with finite-frequency Babushkin/Coulomb operators.
Mode 1 is compared with explicit static operators. Both use the same Dirac
orbitals. The default finite-frequency electric rates are also compared to
their static limits to quantify the effect of photon retardation.
State matching uses n, l, J and parity, never nearest transition energy.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import time
import numpy as np

from benchmarks.bench_reference_structure import run_reference, sha256
from benchmarks.fac_runner import parse_levels, radial_grid_command, orbital_accuracy_commands
from atomx.constants import ALPHA, AU_RATE_PER_S, HARTREE_EV, chemical_symbols
from atomx.config.subshell import Subshell
from atomx.io.fac_writer import read_tra
from atomx.numerics.shooting.local import ShootingSolver
from atomx.physics.dirac import hydrogenic_dirac_energy
from atomx.properties.transitions.angular import angular_eL, angular_mL
from atomx.properties.transitions.multipole import radial_eL, radial_eL_velocity, radial_mL
from atomx.properties.transitions.rates import einstein_A_eL, oscillator_strength

GROUPS = [('g', '1s1'), ('s', '2s1')] + [
    (name+branch, f'{n}[{ell}{sign}]1')
    for name, n, ell in [('p', 2, 'p'), ('d', 3, 'd'), ('f', 4, 'f'), ('q', 5, 'g')]
    for branch, sign in [('minus', '-'), ('plus', '+')]]


def script(z, points, *, fac_grid_density=1., fac_energy_tolerance=None):
    lines = orbital_accuracy_commands(fac_energy_tolerance) + [f"SetAtom('{chemical_symbols[z]}',{z},-1,0)",
        'SetBreit(0)', 'SetSE(0)', 'SetVP(0)', 'SetMS(0,0)',
        radial_grid_command(points, density_scale=fac_grid_density), 'SetTransitionCut(0)']
    lines += [f"Config('{name}','{config}')" for name, config in GROUPS]
    lines += ["OptimizeRadial(['g'])", f"Structure('levels.b',{[g for g, _ in GROUPS]!r})",
        "MemENTable('levels.b')", "PrintTable('levels.b','levels.txt',1)"]
    for mode in (1, 0):
        lines.append(f'SetTransitionOptions(2,{mode})')
        for rank in range(1, 5):
            for sign, groups in ((-1, 'pdfq'), (1, 'spdf')):
                # FAC always evaluates M1 with its relativistic operator
                # and writes MODE=0, even when MODE=1 was requested.
                if sign == 1 and rank == 1 and mode == 1:
                    continue
                name = groups[rank-1]
                # One resolved upper state per call: otherwise SaveTransition
                # can evaluate close fine-structure partners at the midpoint
                # photon frequency, including angularly forbidden partners.
                upper_groups = ([name+'minus', name+'plus'] if sign < 0
                                else [name if rank == 1 else name+'plus'])
                for upper in upper_groups:
                    lines.append(f"TransitionTable('lines.b',['g'],['{upper}'],{sign*rank})")
    lines.append("PrintTable('lines.b','lines.txt',1)")
    return '\n'.join(lines)+'\n'


def identity(level):
    n, ell = divmod(int(level['VNL']), 100)
    twoj = int(level['TWOJ'])
    if n <= ell or twoj not in (2*ell-1, 2*ell+1) or twoj <= 0:
        raise ValueError('invalid one-electron FAC state')
    if level['P'] != ell % 2:
        raise ValueError('FAC parity differs from the orbital label')
    return n, -(ell+1) if twoj == 2*ell+1 else ell


def atomx_orbitals(z, keys, points):
    result = {}
    for n, kappa in keys:
        state = ShootingSolver(z, [Subshell(n, kappa, 1)], nuclear='point',
            n_grid=points, r_max=125./z).solve(tol=1e-11)
        if not state.converged:
            raise RuntimeError(f'unconverged Atomx orbital Z={z}, {(n,kappa)}')
        result[n, kappa] = state
    return result


def compare(z, levels, rows, orbitals):
    ids = {level['ILEV']: identity(level) for level in levels}
    if len(set(ids.values())) != len(ids) or len(ids) != 10:
        raise ValueError('unexpected one-electron level coverage')
    output = []
    for row in rows:
        key_u, key_l = ids[row['upper']], ids[row['lower']]
        up, lo = orbitals[key_u], orbitals[key_l]
        if not np.array_equal(up._grid, lo._grid):
            raise ValueError('transition spinors need the same quadrature')
        workspace = lo.radial_state.correction_workspace()
        rank, electric = abs(row['multipole_code']), row['multipole_code'] < 0
        radial = radial_eL if electric else radial_mL
        angular = angular_eL if electric else angular_mL
        omega = up.total_energy-lo.total_energy
        waves = (lo._pa[0,1:], lo._qa[0,1:], up._pa[0,1:], up._qa[0,1:])
        opts = dict(kappa_i=key_l[1], kappa_f=key_u[1]) if electric else {}
        rme = radial(*waves, workspace, rank,
                     omega=omega if row['mode']==0 else None, **opts)
        gauge_ratio = None
        retardation_rate_ratio = None
        if electric and row['mode']==0:
            velocity = radial_eL_velocity(*waves, workspace, rank, omega=omega, **opts)
            gauge_ratio = (velocity/rme)**2
            static = radial_eL(*waves, workspace, rank)
            retardation_rate_ratio = (rme/static)**2
        strength = (rme*angular(key_u[1], key_l[1], rank))**2
        rate = einstein_A_eL(omega, strength, rank, row['twoj_upper'])
        _, gf = oscillator_strength(omega, strength, rank, row['twoj_lower'], row['twoj_upper'])
        ref_omega = row['DELTA_E_eV']/HARTREE_EV
        ref_gf_from_rate = ((row['twoj_upper']+1)*row['A_per_s'] /
                           (2*ALPHA**3*ref_omega**2*AU_RATE_PER_S))
        exact_omega = (hydrogenic_dirac_energy(*key_u,z)-hydrogenic_dirac_energy(*key_l,z))
        output.append(dict(upper=key_u, lower=key_l, multipole=row['multipole'],
            fac_mode=row['mode'], comparison=('same_long_wavelength_operator'
                if row['mode']==1 else 'same_finite_frequency_operator'),
            gauge_rate_ratio=gauge_ratio, finite_to_static_rate_ratio=retardation_rate_ratio,
            fac=row, atomx=dict(energy_eV=omega*HARTREE_EV, line_strength=strength,
                A_per_s=rate, gf=gf), energy_difference_eV=omega*HARTREE_EV-row['DELTA_E_eV'],
            atomx_energy_error_vs_analytic_Ha=omega-exact_omega,
            A_ratio=rate/row['A_per_s'] if row['A_per_s'] else None,
            gf_ratio=gf/row['gf'] if row['gf'] else None,
            fac_gf_rate_identity_relative=ref_gf_from_rate/row['gf']-1 if row['gf'] else None))
    expected = {(kind+str(rank),mode) for kind in 'EM' for rank in range(1,5) for mode in (0,1)}
    expected.remove(('M1',1))
    present = {(r['multipole'],r['fac_mode']) for r in output}
    if present != expected or len(output) != 23:
        raise ValueError(f'incomplete transition panel: {expected-present}, {len(output)} rows')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fac', type=Path, default=Path.home()/'code/fac/bin/sfac')
    parser.add_argument('--z', nargs='+', type=int, default=[1,10,26,54,82])
    parser.add_argument('--points', type=int, default=2000)
    parser.add_argument('--fac-grid-density', type=float, default=1.)
    parser.add_argument('--fac-energy-tolerance', type=float)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    report = dict(scope='one-electron point-nucleus, infinite mass, no corrections; E1-E4/M1-M4',
        python=platform.python_version(), points=args.points, fac_grid_density=args.fac_grid_density,
        fac_energy_tolerance=args.fac_energy_tolerance,
        fac_photon_grid='one exact line frequency per resolved upper-state call',
        constants=dict(alpha_atomx=ALPHA, fac_constants='unmodified supplied build'),
        timing='FAC process includes output; Atomx orbitals include construction and solve, no imports; repeated solve reuses JIT caches',
        source_sha256={str(p.relative_to(root)):sha256(p) for p in (root/'src').rglob('*.py')},
        reference=dict(path=str(args.fac.resolve()), sha256=sha256(args.fac)),
        threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')}, cases=[])
    for z in args.z:
        folder = args.out/f'Z{z:03d}'
        folder.mkdir()
        (folder/'run.sf').write_text(script(z, args.points, fac_grid_density=args.fac_grid_density,
                                         fac_energy_tolerance=args.fac_energy_tolerance))
        seconds = run_reference([str(args.fac),'run.sf'], folder)
        levels, rows = parse_levels(folder/'levels.txt'), read_tra(folder/'lines.txt')
        times = []
        for _ in range(2):
            start = time.perf_counter()
            orbitals = atomx_orbitals(z, [identity(l) for l in levels], args.points)
            times.append(time.perf_counter()-start)
        start = time.perf_counter()
        comparison = compare(z,levels,rows,orbitals)
        report['cases'].append(dict(z=z, reference_s=seconds, atomx_orbitals_s=times,
            comparison_s=time.perf_counter()-start, rows=comparison))
        (args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        print(z, len(comparison), 'lines', 'FAC s', seconds, 'Atomx s', times, flush=True)


if __name__ == '__main__':
    main()
