"""Live FAC transverse exchange on independent point-Coulomb spinors.

Compare the exchange energy between two completely occupied subshells.
FAC contracts its reduced Breit integrals with Wigner 3j coefficients;
Atomx independently sums uncoupled magnetic-substate matrix elements.
This is an operator test on fixed spinors, not a many-electron SCF result.
Each FAC mode runs in a fresh process because its Breit cache omits mode.
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from atomx.constants import chemical_symbols
from atomx.physics.csf.breit_ci import breit_terms
from atomx.physics.csf.matrix import GridCIWorkspace
from atomx.physics.csf.transverse import radial_evaluator
from benchmarks.bench_reference_structure import run_reference, sha256
from benchmarks.bench_reference_transitions import atomx_orbitals
from benchmarks.fac_runner import orbital_accuracy_commands, radial_grid_command

PAIRS = [((1, -1), (2, -1)), ((1, -1), (2, 1)),
         ((1, -1), (2, -2)), ((2, 1), (2, -2)),
         ((2, -2), (3, 2)), ((2, -2), (3, -3))]
ELEMENTS = [((1, -1, 1), (1, -1, -1), (2, -1, 1), (3, -1, -1)),
            ((1, -1, 1), (2, -1, -1), (2, 1, 1), (3, 1, -1)),
            ((1, -1, 1), (2, -2, -1), (2, -1, 1), (3, -2, -1))]
ELEMENTS += [(c, d, a, b) for a, b, c, d in ELEMENTS.copy()]


def pair_terms(a, b):
    """Closed-subshell intershell exchange; direct current trace vanishes."""
    terms = {}
    for ma in range(-2*abs(a[1])+1, 2*abs(a[1]), 2):
        for mb in range(-2*abs(b[1])+1, 2*abs(b[1]), 2):
            labels = [(*a, ma), (*b, mb)]
            for key, value in breit_terms(0, 1, 1, 0, -1., labels):
                terms[key] = terms.get(key, 0.) + value
    return {k: float(np.real(v)) for k, v in terms.items() if v}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--fac', type=Path,
                   help='Diagnostic FAC executable; defaults to the common-spinor '
                        'frontend with --common-spinors, otherwise the transverse frontend')
    p.add_argument('--points', type=int, default=2000)
    p.add_argument('--common-spinors', action='store_true',
                   help='Export FAC spinors at full precision and use them in Atomx too')
    p.add_argument('--fac-grid-density', type=float, default=1.,
                   help='Refine FAC log spacing and asymptotic spacing by this factor')
    p.add_argument('--fac-breit-small-x', type=float, default=.01,
                   help='FAC small-argument Breit approximation threshold')
    p.add_argument('--fac-energy-tolerance', type=float)
    p.add_argument('--z', type=int, nargs='+', default=[1, 10, 26, 54, 82])
    args = p.parse_args()
    if args.fac is None:
        args.fac = Path('/tmp/atomx-fac-common-spinors/sfac' if args.common_spinors
                        else '/tmp/atomx-fac-transverse/sfac')
    if not np.isfinite(args.fac_grid_density) or args.fac_grid_density <= 0:
        p.error('--fac-grid-density must be positive and finite')
    if not np.isfinite(args.fac_breit_small_x) or args.fac_breit_small_x <= 0:
        p.error('--fac-breit-small-x must be positive and finite')
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    report = dict(scope=__doc__, points=args.points,
                  spinor_source='fac' if args.common_spinors else 'independent',
                  fac_grid_density=args.fac_grid_density,
                  fac_breit_small_x=args.fac_breit_small_x,
                  fac_energy_tolerance=args.fac_energy_tolerance,
                  threads={k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')},
                  reference=dict(executable=str(args.fac), sha256=sha256(args.fac),
                                 library_sha256=sha256(Path.home()/'code/fac/libfac.a')),
                  source_sha256={str(f.relative_to(root)): sha256(f) for f in (root/'src').rglob('*.py')},
                  cases=[])
    keys = sorted({a for pair in PAIRS for a in pair}
                  | {s[:2] for element in ELEMENTS for s in element})
    terms = [pair_terms(*pair) for pair in PAIRS]
    for element in ELEMENTS:
        row = {}
        for key, value in breit_terms(0, 1, 2, 3, 1., element):
            row[key] = row.get(key, 0.) + float(np.real(value))
        terms.append(row)
    for z in args.z:
        case = dict(z=z, rows=[], off_diagonal=[], fac_process_s={})
        references, elements = {}, {}
        for mode in (2, 1):
            folder = args.out/f'Z{z:03d}-mode{mode}'
            folder.mkdir()
            lines = orbital_accuracy_commands(args.fac_energy_tolerance) + [f"SetAtom('{chemical_symbols[z]}',{z},-1,0)",
                     f'SetBreit(0,-1,-1,{args.fac_breit_small_x:.17g})',
                     'SetSE(0)', 'SetVP(0)', 'SetMS(0,0)',
                     radial_grid_command(args.points, density_scale=args.fac_grid_density),
                     "Config('g','1s1')", "OptimizeRadial(['g'])"]
            lines += [f"DumpBreitPair('pairs.txt',{a[0]},{a[1]},{b[0]},{b[1]},{mode})"
                      for a, b in PAIRS]
            lines += ["DumpBreitElement('elements.txt'," + ','.join(
                      str(v) for s in element for v in s) + f",{mode})"
                      for element in ELEMENTS]
            if args.common_spinors:
                lines += [f"DumpSpinor('wave-{n}-{k}.txt',{n},{k})" for n,k in keys]
            (folder/'run.sf').write_text('\n'.join(lines)+'\n')
            case['fac_process_s'][mode] = run_reference([str(args.fac), 'run.sf'], folder)
            references[mode] = np.loadtxt(folder/'pairs.txt')
            elements[mode] = np.loadtxt(folder/'elements.txt')
        orbital_times = []
        if args.common_spinors:
            orbitals, energies = {}, {}
            grid = None
            for n,k in keys:
                f = args.out/f'Z{z:03d}-mode1'/f'wave-{n}-{k}.txt'
                static = args.out/f'Z{z:03d}-mode2'/f.name
                data = np.loadtxt(f)
                np.testing.assert_array_equal(data, np.loadtxt(static))
                r = np.r_[0., data[:,0]]
                if grid is not None:
                    np.testing.assert_array_equal(r, grid)
                grid = r
                orbitals[n,k] = np.r_[0.,data[:,1]], np.r_[0.,data[:,2]]
                energies[n,k] = float(re.search(r'energy_Ha=(\S+)',f.read_text()).group(1))
            # Only grid geometry is used by the transverse operator.
            ws = GridCIWorkspace(grid, np.ones_like(grid), np.full_like(grid,-z))
        else:
            for _ in range(2):
                t = time.perf_counter()
                states = atomx_orbitals(z, keys, args.points)
                orbital_times.append(time.perf_counter()-t)
            state = states[keys[0]]
            ws = GridCIWorkspace(state._grid, state._difr, np.full_like(state._grid, -z))
            orbitals = {a: (s._pa[0], s._qa[0]) for a, s in states.items()}
            energies = {a: s.total_energy for a, s in states.items()}
        values = {}
        case['atomx_orbitals_s'] = orbital_times
        case['atomx_operator_s'] = {}
        for mode in (2, 1):
            timings = []
            for _ in range(3):
                t = time.perf_counter()
                evaluate = radial_evaluator(orbitals, ws, energies if mode == 1 else {a: 0. for a in keys})
                values[mode] = [sum(c*evaluate(k) for k, c in row.items()) for row in terms]
                timings.append(time.perf_counter()-t)
            case['atomx_operator_s'][mode] = timings
        for i, (a, b) in enumerate(PAIRS):
            fac0, facw = references[2][i, -1], references[1][i, -1]
            ax0, axw = values[2][i], values[1][i]
            case['rows'].append(dict(a=a, b=b, fac_static_Ha=fac0, fac_finite_Ha=facw,
                atomx_static_Ha=ax0, atomx_finite_Ha=axw,
                static_relative_error=ax0/fac0-1, finite_relative_error=axw/facw-1,
                fac_delta_Ha=facw-fac0, atomx_delta_Ha=axw-ax0,
                delta_error_Ha=(axw-ax0)-(facw-fac0)))
        for i, element in enumerate(ELEMENTS):
            ax0, axw = values[2][len(PAIRS)+i], values[1][len(PAIRS)+i]
            fac0, facw = elements[2][i, -1], elements[1][i, -1]
            case['off_diagonal'].append(dict(spinors=element,
                fac_static_Ha=fac0, fac_finite_Ha=facw,
                atomx_static_Ha=ax0, atomx_finite_Ha=axw,
                static_magnitude_error=abs(ax0/fac0)-1,
                finite_magnitude_error=abs(axw/facw)-1,
                finite_signed_error=(axw/facw-1 if args.common_spinors else None),
                fac_finite_to_static_ratio=facw/fac0,
                atomx_finite_to_static_ratio=axw/ax0))
        report['cases'].append(case)
        (args.out/'results.json').write_text(json.dumps(report, indent=2)+'\n')
        print(z, 'static', max(abs(r['static_relative_error']) for r in case['rows']),
              'finite', max(abs(r['finite_relative_error']) for r in case['rows']),
              'off_diagonal', max(abs(r['finite_magnitude_error']) for r in case['off_diagonal']),
              'delta_error_Ha', max(abs(r['delta_error_Ha']) for r in case['rows']), flush=True)


if __name__ == '__main__':
    main()
