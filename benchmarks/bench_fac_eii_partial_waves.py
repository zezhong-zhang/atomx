"""Refine FAC's ejected partial waves on the retained Coulomb-panel decks."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmarks.bench_reference_structure import run_reference
from benchmarks.fac_runner import parse_ci


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fac', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--z', type=int, nargs='+', default=[2, 82])
    parser.add_argument('--eject', type=int, nargs='+', default=[4, 8])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = json.loads((args.reference/'results.json').read_text())
    report = dict(fac_sha256=hashlib.sha256(args.fac.read_bytes()).hexdigest(),
                  reference=str(args.reference), cases=[])
    for z in args.z:
        original = next(c for c in source['cases'] if c['z'] == z)
        deck = (args.reference/f'Z{z:03d}'/'run.sf').read_text()
        # New baseline decks state the legacy rank-3 limit explicitly.
        # Remove it before inserting this study's rank, so a later command
        # cannot silently override the refinement.
        deck = '\n'.join(line for line in deck.splitlines()
                         if not line.startswith("SetOption('ionization:maxk',"))+'\n'
        for eject in args.eject:
            folder = args.out/f'Z{z:03d}-eject{eject}'
            folder.mkdir()
            old = 'SetCIPWOptions(1e-3,36,4,36)'
            assert deck.count(old) == 1
            new = (f'SetCIPWOptions(1e-3,36,{eject},36)\n'
                   f"SetOption('ionization:maxk',{2*eject+2})")
            (folder/'run.sf').write_text(deck.replace(old, new))
            seconds = run_reference([str(args.fac.resolve()), 'run.sf'], folder, timeout=300)
            rows = parse_ci(folder/'ci.txt')
            if len(rows) != 1:
                raise RuntimeError('expected one hydrogen-like EII transition')
            sigma = np.asarray(rows[0]['sigma_1e20cm2'])*1e-20
            case = dict(z=z, eject=eject, incident_lmax=36, seconds=seconds,
                        sigma_cm2=sigma.tolist(),
                        ratios_original=(sigma/original['fac_sigma_cm2']).tolist(),
                        raw_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in folder.iterdir() if p.is_file()})
            report['cases'].append(case)
            (args.out/'results.json').write_text(json.dumps(report, indent=2)+'\n')
            print(case, flush=True)


if __name__ == '__main__':
    main()
