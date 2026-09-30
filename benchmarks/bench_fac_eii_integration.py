"""Replay a fixed FAC EII deck while refining one integration control."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from benchmarks.bench_reference_structure import run_reference
from benchmarks.fac_runner import parse_ci


def refine_deck(deck, *, full_relativistic=False, dense_partial_waves=False,
                energy_index=None):
    """Change explicit numerical controls without guessing SFAC defaults."""
    if full_relativistic or dense_partial_waves:
        pattern = r'SetCIPWOptions\(([^\n()]*)\)'
        matches = list(re.finditer(pattern, deck))
        if len(matches) != 1:
            raise ValueError('expected one explicit SetCIPWOptions call')
        match = matches[0]
        values = list(ast.literal_eval('('+match.group(1)+')'))
        if len(values) not in (4, 5):
            raise ValueError('reference must specify tolerance, incident/ejected cutoffs and qr')
        if full_relativistic:
            # SFAC order differs from its C API: tol, max_l, eject_l, qr, cb_l.
            # The C implementation turns negative qr into a disabled threshold.
            values[3] = -1
            deck = deck[:match.start()]+'SetCIPWOptions('+','.join(map(str, values))+')'+deck[match.end():]
        if dense_partial_waves:
            if 'SetCIPWGrid(' in deck:
                raise ValueError('reference already contains partial-wave grid controls')
            maximum = values[1]
            if int(maximum) != maximum or maximum < 1:
                raise ValueError('invalid incident partial-wave cutoff')
            position = "CITable('ci.b'"
            if deck.count(position) != 1:
                raise ValueError('expected one CI table call')
            deck = deck.replace(position, f'SetCIPWGrid([{int(maximum)}],[1])\n'+position)
    if energy_index is not None:
        if energy_index < 0:
            raise ValueError('energy index must be nonnegative')
        # Select output only. FAC interpolates on its internal energy grid;
        # collapsing that grid to one point makes UVIP3P stop with ND=1.
        # Keeping it unchanged also preserves the reference calculation.
        for name in ['SetUsrCIEGrid']:
            pattern = rf'{name}\((\[[^\n]*\])\)'
            matches = list(re.finditer(pattern, deck))
            if len(matches) != 1:
                raise ValueError(f'expected one explicit {name} array')
            match = matches[0]
            values = ast.literal_eval(match.group(1))
            if energy_index >= len(values):
                raise ValueError('energy index exceeds the reference grid')
            deck = deck[:match.start()]+f'{name}({[values[energy_index]]!r})'+deck[match.end():]
    return deck


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fac', type=Path, required=True)
    parser.add_argument('--dense-partial-waves', action='store_true')
    parser.add_argument('--full-relativistic-waves', action='store_true',
                        help='Disable FAC high-l kappa-branch substitution (SFAC qr=-1)')
    parser.add_argument('--energy-index', type=int,
                        help='Select one user output energy; preserve the internal energy grid')
    parser.add_argument('--radial-grid', type=float, nargs=3,
                        metavar=('POINTS', 'RATIO', 'ASYMPTOTIC_DENSITY'),
                        help='Explicit FAC radial-grid controls; r_min stays automatic')
    parser.add_argument('--timeout', type=float, default=600,
                        help='Reference process limit in seconds (default 600)')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    deck = (args.reference/'run.sf').read_text()
    if args.radial_grid is not None:
        points, ratio, asym = args.radial_grid
        if points != int(points) or points < 100 or ratio <= 1 or asym <= 0:
            parser.error('radial-grid needs integer points >= 100, ratio > 1, density > 0')
        if 'SetRadialGrid(' in deck:
            raise ValueError('reference deck already contains radial-grid controls')
        position = "OptimizeRadial("
        assert deck.count(position) == 1
        deck = deck.replace(position,
                            f'SetRadialGrid({int(points)},{ratio:.17g},{asym:.17g},-1)\n'+position)
    deck = refine_deck(deck, full_relativistic=args.full_relativistic_waves,
                       dense_partial_waves=args.dense_partial_waves,
                       energy_index=args.energy_index)
    (args.out/'run.sf').write_text(deck)
    seconds = run_reference([str(args.fac.resolve()), 'run.sf'], args.out,
                            timeout=args.timeout)
    ref, result = parse_ci(args.reference/'ci.txt'), parse_ci(args.out/'ci.txt')
    assert len(ref) == len(result) == 1
    selection = slice(None) if args.energy_index is None else [args.energy_index]
    np.testing.assert_array_equal(result[0]['E_eV'], np.asarray(ref[0]['E_eV'])[selection])
    sigma = np.asarray(result[0]['sigma_1e20cm2'])*1e-20
    baseline = np.asarray(ref[0]['sigma_1e20cm2'])[selection]*1e-20
    report = dict(reference=str(args.reference), seconds=seconds,
        dense_partial_waves=args.dense_partial_waves,
        full_relativistic_waves=args.full_relativistic_waves,
        energy_index=args.energy_index,
        radial_grid=args.radial_grid,
        executable_sha256=hashlib.sha256(args.fac.read_bytes()).hexdigest(),
        sigma_cm2=sigma.tolist(), ratios_reference=(sigma/baseline).tolist(),
        raw_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in args.out.iterdir() if p.is_file()})
    (args.out/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(report, flush=True)


if __name__ == '__main__':
    main()
