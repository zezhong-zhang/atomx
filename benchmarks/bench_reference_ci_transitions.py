"""Live Ne-like frozen-configuration CI levels and E1 rates against FAC.

Both codes mix all jj states within each 2p5 3s / 2p5 3d configuration,
separately. They optimize their own local ground-state potentials: this
compares physical models, not identical radial inputs or full MCDHF.
Point nuclei, infinite mass and Dirac-Coulomb energies isolate that difference.
Static and finite-wavelength photons are compared independently. Every FAC
transition call contains a single upper level, avoiding frequency averaging.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
import time

from atomx import Atom, Basis, Sampling
from atomx.constants import HARTREE_EV, chemical_symbols
from atomx.properties.transitions.ci_lines import ci_line_strengths
from benchmarks.bench_reference_structure import run_reference, sha256
from benchmarks.fac_runner import (
    orbital_accuracy_commands, parse_levels, parse_transitions, radial_grid_command)


CONFIGS = ('2p5 3s1', '2p5 3d1')
MODES = {'long_wavelength': 1, 'finite_wavelength': 0}


def fac_script(z, points, density, *, configuration_shifts=False):
    lines = [f"SetAtom('{chemical_symbols[z]}',{z},-1,0)",
             'SetSE(0)', 'SetVP(0)', 'SetMS(0,0)', 'SetBreit(0)',
             radial_grid_command(points, density_scale=density),
             *orbital_accuracy_commands(1e-12),
             # This is a SCALE on FAC's compile-time convergence constants,
             # not an absolute energy tolerance: see OptimizeILoop.
             'SetOptimizeTolerance(1e-4)', 'SetOptimizeMaxIter(150)', 'SetOptimizePrint(1)',
             "Closed('1s')", "Config('gnd','2s2 2p6')",
             "Config('up3s','2s2 2p5 3s1')", "Config('up3d','2s2 2p5 3d1')"]
    if configuration_shifts:
        lines += ['ConfigEnergy(0)', "OptimizeRadial(['gnd'])", 'ConfigEnergy(1)']
    else:
        lines += ['SetConfigEnergyMode(-1)', "OptimizeRadial(['gnd'])"]
    for group in ('gnd', 'up3s', 'up3d'):
        lines.append(f"Structure('levels.b',['{group}'])")
    lines += ["MemENTable('levels.b')", "PrintTable('levels.b','levels.txt',1)",
              'SetTransitionCut(0)']
    for name, mode in MODES.items():
        lines.append(f'SetTransitionOptions(2,{mode})')
        # Separate Structure calls give ground 0, four 3s and twelve 3d
        # levels. Verify that inventory from the actual output before use.
        for upper in range(1, 17):
            lines.append(f"TRTable('lines.b',[0],[{upper}],-1)")
    lines.append("PrintTable('lines.b','lines.txt',1)")
    return '\n'.join(lines)+'\n'


def reference(folder):
    levels = parse_levels(folder/'levels.txt')
    if len(levels) != 17 or [lv['ILEV'] for lv in levels] != list(range(17)):
        raise ValueError('FAC level inventory differs from the requested configuration space')
    grouped = {}
    for config, vnl, count in zip(CONFIGS, (300, 302), (4, 12), strict=True):
        rows = [lv for lv in levels if int(lv['VNL']) == vnl]
        if len(rows) != count or any(lv['P'] != 1 for lv in rows):
            raise ValueError(f'FAC configuration/parity inventory mismatch: {config}')
        grouped[config] = rows
    modes = {}
    all_lines = parse_transitions(folder/'lines.txt')
    if len(all_lines) != 10:
        raise ValueError('FAC must return five lines for each of the two photon operators')
    for name, mode in MODES.items():
        lines = [row for row in all_lines if row['mode'] == mode]
        if len(lines) != 5 or any(
            row['mode'] != mode or row['gauge'] != 'length' or row['mult'] != -1
            or row['lower'] != 0 or row['twoj_upper'] != 2 or row['twoj_lower'] != 0
            for row in lines
        ):
            raise ValueError(f'FAC transition inventory/operator mismatch: {name}')
        modes[name] = lines
    return grouped, modes


def compare_levels(fac, calculated, ground):
    """Compare eigenvalue multisets within J/parity, without nearest-energy fitting.

    Energy rank within a symmetry block is not a configuration-mixing
    identity through an avoided crossing. Retain labels for later overlap audits.
    """
    rows = []
    symmetries = {(row['TWOJ'], (-1)**row['P']) for row in fac}
    if symmetries != {(row.two_J, row.parity) for row in calculated}:
        raise ValueError('CI symmetries differ between codes')
    for twoj, parity in sorted(symmetries):
        left = sorted((row for row in fac if row['TWOJ'] == twoj), key=lambda r: r['ENERGY_eV'])
        right = sorted((row for row in calculated if row.two_J == twoj and row.parity == parity),
                       key=lambda r: r.energy_au)
        for rank, (fr, ax) in enumerate(zip(left, right, strict=True)):
            energy = (ax.energy_au-ground)*HARTREE_EV
            rows.append(dict(two_J=twoj, parity=parity, energy_rank=rank,
                fac_level=fr['ILEV'], atomx_level=ax.index,
                fac_label=fr['config'], atomx_label=ax.label,
                fac_eV=fr['ENERGY_eV'], atomx_eV=energy,
                difference_eV=energy-fr['ENERGY_eV']))
    return rows


def run_case(z, folder, args):
    folder.mkdir()
    (folder/'run.sf').write_text(fac_script(z, args.fac_points, args.fac_density,
        configuration_shifts=args.fac_configuration_shifts))
    fac_seconds = run_reference([str(args.fac.resolve()), 'run.sf'], folder)
    output = (folder/'stdout.txt').read_text()
    if 'Maximum iteration reached' in output or 'maxiter reached' in output.lower():
        raise RuntimeError('FAC reported an optimization iteration limit')
    iterations = [line.split(':', 1)[1].split() for line in output.splitlines()
                  if line.startswith('optimize loop:')]
    if not iterations or max(int(row[0]) for row in iterations) >= 150:
        raise RuntimeError('FAC optimization convergence is not established')
    fac_levels, fac_lines = reference(folder)
    timings = []
    for repeat in range(args.repeats+1):
        start = time.perf_counter()
        atom = Atom(z, charge=z-10, config='1s2 2s2 2p6', engine='shooting',
                    exchange='gu', nuclear='point', corrections='none',
                    basis=Basis(sampling=Sampling(n_grid=args.atomx_points)))
        atom.solve(tol=1e-10, max_iter=150)
        if not atom.converged:
            raise RuntimeError(f'Atomx local ground solve did not converge for Z={z}')
        ground_seconds = time.perf_counter()-start
        line_sets = {name: {cfg: ci_line_strengths(atom.result, cfg, '2p6',
                        photon_approximation=name) for cfg in CONFIGS} for name in MODES}
        timings.append(dict(total_s=time.perf_counter()-start, ground_s=ground_seconds))
    levels, lines, photon_shifts = {}, {}, {}
    for cfg in CONFIGS:
        ls = line_sets['finite_wavelength'][cfg]
        levels[cfg] = compare_levels(fac_levels[cfg], ls.levels_i, ls.levels_f[0].energy_au)
    for name in MODES:
        rows = []
        for cfg in CONFIGS:
            ax_by_id = {ln.upper.index: ln for ln in line_sets[name][cfg].lines}
            fac_by_id = {ln['upper']: ln for ln in fac_lines[name]}
            for match in levels[cfg]:
                if match['two_J'] != 2:
                    continue
                ax, fr = ax_by_id[match['atomx_level']], fac_by_id[match['fac_level']]
                # FAC's signed E1 amplitude is sqrt(2) times the conventional
                # reduced dipole, in both MODE=0 and MODE=1 (OscillatorStrength).
                strength = fr['reduced_matrix_element']**2/2
                rows.append(dict(config=cfg, energy_rank=match['energy_rank'],
                    fac=fr, atomx=asdict(ax), S_ratio=ax.S/strength,
                    A_ratio=ax.A_per_s/fr['A_per_s'],
                    energy_difference_eV=ax.omega_au*HARTREE_EV-fr['DELTA_E_eV']))
        lines[name] = rows
    for finite, static in zip(lines['finite_wavelength'], lines['long_wavelength'], strict=True):
        key = f"{finite['config']}:{finite['energy_rank']}"
        photon_shifts[key] = dict(
            fac_relative_A=finite['fac']['A_per_s']/static['fac']['A_per_s']-1,
            atomx_relative_A=finite['atomx']['A_per_s']/static['atomx']['A_per_s']-1)
    return dict(z=z, levels=levels, lines=lines, photon_shifts=photon_shifts,
                atomx_converged=True, atomx_timings=timings, fac_process_s=fac_seconds,
                fac_last_optimization_iteration=iterations[-1],
                atomx_warm_median_s=statistics.median(t['total_s'] for t in timings[1:]),
                raw_sha256={p.name: sha256(p) for p in folder.iterdir() if p.is_file()})


def write_fixture(report, destination):
    """Export only external values, with the raw files and executable identified."""
    fixture = dict(scope=__doc__, controls=report['controls'],
        fac_sha256=report['fac_sha256'], runner_sha256=report['runner_sha256'],
        amplitude_convention='E1 S = reduced_matrix_element**2 / 2; printed rate is A, not gA',
        cases=[])
    for case in report['cases']:
        folder = Path(report['controls']['out'])/f"Z{case['z']:03d}"
        levels, lines = reference(folder)
        fixture['cases'].append(dict(z=case['z'], levels=levels, lines=lines,
                                    raw_sha256=case['raw_sha256']))
    destination.write_text(json.dumps(fixture, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fac', type=Path, required=True)
    parser.add_argument('--z', type=int, nargs='+', default=[18, 26, 54, 82])
    parser.add_argument('--fac-points', type=int, default=8000)
    parser.add_argument('--fac-density', type=float, default=4.)
    parser.add_argument('--atomx-points', type=int, default=2000)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--fac-configuration-shifts', action='store_true',
        help='diagnostic only: add FAC configuration relaxation shifts absent in Atomx frozen CI')
    parser.add_argument('--fixture', type=Path, help='export external values only after a successful panel')
    args = parser.parse_args()
    if args.repeats < 1 or any(z < 10 or z > 118 for z in args.z):
        parser.error('need at least one warm repeat and Ne-like Z in [10,118]')
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(scope=__doc__, controls={k: str(v) if isinstance(v, Path) else v
                  for k, v in vars(args).items()},
                  fac_sha256=sha256(args.fac), runner_sha256=sha256(__file__),
                  threads={k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')},
                  cases=[])
    root = Path(__file__).resolve().parents[1]
    report['source_sha256'] = {str(p.relative_to(root)): sha256(p)
                              for p in sorted((root/'src/atomx').rglob('*.py'))}
    for z in args.z:
        case = run_case(z, args.out/f'Z{z:03d}', args)
        report['cases'].append(case)
        (args.out/'results.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(z=z, fac_s=case['fac_process_s'],
            atomx_warm_s=case['atomx_warm_median_s'],
            max_level_difference_eV=max(abs(r['difference_eV']) for rs in case['levels'].values() for r in rs),
            max_finite_A_relative_difference=max(abs(r['A_ratio']-1) for r in case['lines']['finite_wavelength']))), flush=True)
    if args.fixture is not None:
        write_fixture(report, args.fixture)


if __name__ == '__main__':
    main()
