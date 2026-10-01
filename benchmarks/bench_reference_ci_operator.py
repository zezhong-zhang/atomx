"""Independent FAC CI spectra on common radial integrals and common spinors.

First feed FAC's exported one-body/Slater integrals to Atomx's angular CI
assembly. Then recompute those integrals from the exported spinors with the
production shooting-grid kernels. No orbital optimization is substituted.
Compare complete eigenvalue multisets per J/parity block; CSF phase conventions
need not agree. The diagnostic frontend calls the unmodified FAC library.
"""
import argparse
import json
import os
import re
import time
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
from scipy.interpolate import make_interp_spline

from atomx.angular.csf_hamiltonian import csf_signature, exact_block_matrix
from atomx.config.csf import enumerate_jj_csfs_many
from atomx.constants import ALPHA, SPEED_OF_LIGHT, chemical_symbols
from atomx.numerics.shooting.grid import sgrid
from atomx.numerics.shooting.operators import _ds_derivative
from atomx.numerics.shooting.quadrature import slag6
from atomx.physics.csf.matrix import (
    GridCIWorkspace,
    _group_csf_indices,
    _radial_evaluator,
    _reset_memo,
)
from benchmarks.bench_reference_structure import run_reference, sha256
from benchmarks.fac_runner import (
    orbital_accuracy_commands,
    parse_hamiltonian_binary,
    radial_grid_command,
)


def configurations(sequence):
    if sequence == 'helium_like':
        shells = ['1s', '2s', '2p', '3s', '3p', '3d']
        return [a+'2' if a == b else a+'1 '+b+'1'
                for a, b in combinations_with_replacement(shells, 2)]
    if sequence == 'carbon_like':
        return ['1s2 2s2 2p2', '1s2 2s1 2p3', '1s2 2p4']
    if sequence == 'neon_like':
        return ['1s2 2s2 2p6'] + [f'1s2 2s2 2p5 3{ell}1' for ell in 'spd'] + [
            f'1s2 2s1 2p6 3{ell}1' for ell in 'spd']
    raise ValueError(f'unknown isoelectronic sequence {sequence}')


def angular_blocks(configs):
    csfs = enumerate_jj_csfs_many(configs)
    blocks = {sym: [csf_signature(csfs[i]) for i in indices]
              for sym, indices in _group_csf_indices(csfs)}
    keys = set()
    def collect(key):
        keys.add(key)
        return 0.
    for signatures in blocks.values():
        exact_block_matrix(signatures, collect)
    orbitals = sorted({(sh.n, sh.kappa) for csf in csfs for sh in csf.subshells})
    return blocks, sorted(keys), orbitals


def freeze(value):
    return tuple(map(freeze, value)) if isinstance(value, list) else value


def read_spinors(folder, orbitals):
    sources = {}
    for n, kappa in orbitals:
        path = folder/f'wave-{n}-{kappa}.txt'
        header = path.open().readline()
        end = int(re.search(r'ilast=(\d+)', header).group(1))+1
        # Zero padding outside ilast must not enter the interpolant.
        sources[n, kappa] = np.loadtxt(path)[:end, :3]
    return sources


def common_grid(sources, z, points, *, alpha=ALPHA):
    """Interpolate supported FAC spinors to a genuine uniform-index SGRID.

    FAC's positive first radius cannot simply be prefixed with r=0 while
    retaining its old Jacobian. A new grid keeps derivative and integral
    metrics consistent. The tiny unresolved interval to the origin and FAC
    interpolation error are included in the limitations of the comparison.
    Point-Coulomb spinors behave as r**gamma, gamma=sqrt(kappa**2-(Z*alpha)**2).
    Interpolating that nonanalytic cusp as an ordinary polynomial through an
    appended origin can create large derivative errors as source spacing shrinks.
    Interpolate the regular factor instead and restore the known cusp.
    """
    rn = max(data[-1, 0] for data in sources.values())
    r, difr, count = sgrid(rn, 1e-8, 2*rn/points, points)
    orbitals = {}
    for key, data in sources.items():
        supported = r <= data[-1, 0]
        gamma = np.sqrt(key[1]**2-(z*alpha)**2)
        inside = supported & (r >= data[0, 0])
        below = r < data[0, 0]
        components = []
        for j in (1, 2):
            value = np.zeros_like(r)
            regular = data[:, j]/data[:, 0]**gamma
            value[inside] = make_interp_spline(data[:, 0], regular, k=5)(r[inside])
            # Continue the regular factor linearly across the unresolved
            # origin interval. Avoid high-order extrapolation across many
            # tiny source cells; the physical r**gamma factor sets P(0)=Q(0)=0.
            slope = (regular[1]-regular[0])/(data[1, 0]-data[0, 0])
            value[below] = regular[0]+slope*(r[below]-data[0, 0])
            value[supported] *= r[supported]**gamma
            components.append(value)
        orbitals[key] = tuple(components)
    return orbitals, GridCIWorkspace(r, difr, np.full_like(r, -float(z)))


def spectrum(blocks, radial):
    return {sym: np.linalg.eigvalsh(exact_block_matrix(signatures, radial.__getitem__))
            for sym, signatures in blocks.items()}


def one_body_constant_shift(waves, workspace, key, target_c):
    """Exact change of the Dirac expectation with c on these fixed spinors.

    H(c)=c*K-2*c**2*Q_overlap+V. Report this separately from the native
    calculation; it aligns the reference constant without fitting any value
    or altering the source spinors. Coulomb integrals have no explicit c.
    """
    if key[0] != 'h':
        return 0.
    _, na, kappa, nc = key
    pa, qa = waves[na, kappa]
    pc, qc = waves[nc, kappa]
    kin = .5*((qa*_ds_derivative(pc)+qc*_ds_derivative(pa))
              -(pa*_ds_derivative(qc)+pc*_ds_derivative(qa)))
    jac = workspace.difr
    linear = slag6(1., kin+kappa*(pa*qc+qa*pc)*workspace.inv_r*jac, workspace.n)[-1]
    q_overlap = slag6(1., qa*qc*jac, workspace.n)[-1]
    norm = slag6(1., (pc*pc+qc*qc)*jac, workspace.n)[-1]
    dc = target_c-SPEED_OF_LIGHT
    return float((dc*linear-2*dc*(target_c+SPEED_OF_LIGHT)*q_overlap)/norm)


def compare_spectra(calculated, reference):
    if calculated.keys() != reference.keys():
        raise ValueError('FAC and Atomx CI symmetries differ')
    rows = []
    for sym, eigenvalues in calculated.items():
        ref = reference[sym]
        if eigenvalues.shape != ref.shape:
            raise ValueError(f'FAC and Atomx CI dimensions differ for {sym}')
        rows.append(dict(two_J=sym[0], parity=sym[1],
                         difference_Ha=(eigenvalues-ref).tolist()))
    return rows


def script(z, configs, keys, orbitals, points, density, *, disable_yk_cache=False):
    lines = [f"SetAtom('{chemical_symbols[z]}',{z},-1,0)", 'SetSE(0)', 'SetVP(0)',
        'SetBreit(0)', 'SetMS(0,0)', 'SetConfigEnergyMode(-1)',
        radial_grid_command(points, density_scale=density), *orbital_accuracy_commands(1e-12),
        'SetOptimizeTolerance(1e-4)', 'SetOptimizeMaxIter(150)', 'SetOptimizePrint(1)']
    if disable_yk_cache:
        # GetYk's cache stores float potentials and a fitted asymptotic tail.
        # Keep full double-precision Slater caching; bypass only that compression.
        lines.append('LimitArray(0,0)')
    lines += [f"Config('c{i}','{cfg}')" for i, cfg in enumerate(configs)]
    lines += ["OptimizeRadial(['c0'])",
        "Structure('levels.b','hamilton.b',"+repr([f'c{i}' for i in range(len(configs))])+")",
        "PrintTable('levels.b','levels.txt',1)"]
    for index, key in enumerate(keys):
        if key[0] == 'h':
            _, na, kappa, nc = key
            lines.append(f"DumpCI1E('integrals.txt',{index},{na},{kappa},{nc},{kappa})")
        else:
            _, rank, a, c, b, d = key
            values = ','.join(str(v) for pair in (a, b, c, d) for v in pair)
            lines.append(f"DumpCIRK('integrals.txt',{index},{values},{rank})")
    lines += [f"DumpSpinor('wave-{n}-{k}.txt',{n},{k})" for n, k in orbitals]
    return '\n'.join(lines)+'\n'


def run_case(z, sequence, folder, args):
    folder.mkdir()
    configs = configurations(sequence)
    blocks, keys, orbitals = angular_blocks(configs)
    (folder/'run.sf').write_text(script(z, configs, keys, orbitals, args.fac_points, args.fac_density,
        disable_yk_cache=args.disable_fac_yk_cache))
    (folder/'radial-keys.json').write_text(json.dumps(keys, indent=2)+'\n')
    elapsed = run_reference([str(args.fac.resolve()), 'run.sf'], folder)
    output = (folder/'stdout.txt').read_text()
    iterations = [int(line.split(':', 1)[1].split()[0]) for line in output.splitlines()
                  if line.startswith('optimize loop:')]
    if not iterations or max(iterations) >= 150 or 'Maximum iteration reached' in output:
        raise RuntimeError('FAC orbital convergence is not established')
    radial = {keys[int(fields[0])]: float(fields[1])
              for line in (folder/'integrals.txt').read_text().splitlines() if (fields := line.split())}
    if len(radial) != len(keys):
        raise ValueError('FAC radial integral inventory is incomplete')
    hamiltonian = parse_hamiltonian_binary(folder/'hamilton.b')
    ref = {(block['symmetry']//2, (-1)**(block['symmetry']%2)):
           np.linalg.eigvalsh(block['hamiltonian']) for block in hamiltonian['blocks']}
    start = time.perf_counter()
    common = compare_spectra(spectrum(blocks, radial), ref)
    common_seconds = time.perf_counter()-start
    sources = read_spinors(folder, orbitals)
    fac_alpha = float(re.search(r'alpha=([^\s]+)',
        (folder/'wave-1--1.txt').open().readline()).group(1))
    refinements = []
    for points in args.atomx_points:
        waves, workspace = common_grid(sources, z, points, alpha=fac_alpha)
        _reset_memo()
        evaluate = _radial_evaluator(waves, workspace, z, workspace.n)
        start = time.perf_counter()
        evaluated = {key: evaluate(key) for key in keys}
        seconds = time.perf_counter()-start
        constant_shifts = {key: one_body_constant_shift(waves, workspace, key, 1/fac_alpha)
                           for key in keys}
        aligned = {key: value+constant_shifts[key] for key, value in evaluated.items()}
        norms = {str(key): float(slag6(1., (p*p+q*q)*workspace.difr, workspace.n)[-1])
                 for key, (p, q) in waves.items()}
        refinements.append(dict(points=points, radial_seconds=seconds, norms=norms,
            radial=[dict(key=key, atomx=value, fac=radial[key], difference_Ha=value-radial[key],
                         constant_shift_Ha=constant_shifts[key], matched_c_difference_Ha=aligned[key]-radial[key])
                    for key, value in evaluated.items()],
            spectra=compare_spectra(spectrum(blocks, evaluated), ref),
            matched_c_spectra=compare_spectra(spectrum(blocks, aligned), ref)))
    return dict(z=z, sequence=sequence, configurations=configs, fac_process_s=elapsed,
        fac_alpha=fac_alpha, interpolation='point-Coulomb Frobenius factor, quintic regular part',
        n_integrals=len(keys), n_levels=sum(len(v) for v in ref.values()),
        common_integrals=common, common_assembly_s=common_seconds, refinements=refinements,
        raw_sha256={p.name: sha256(p) for p in folder.iterdir() if p.is_file()})


def write_fixture(report, destination):
    """Retain FAC integrals, matrices and raw supported spinors only."""
    destination.mkdir(parents=True, exist_ok=True)
    arrays = {}
    fixture = {key: report[key] for key in ('scope', 'controls', 'fac_sha256', 'runner_sha256')}
    fixture['cases'] = []
    for case in report['cases']:
        label = f"{case['sequence']}-Z{case['z']:03d}"
        folder = Path(report['controls']['out'])/label
        keys = json.loads((folder/'radial-keys.json').read_text())
        radial = [dict(key=keys[int(fields[0])], value_Ha=float(fields[1]))
                  for line in (folder/'integrals.txt').read_text().splitlines() if (fields := line.split())]
        _, _, orbitals = angular_blocks(case['configurations'])
        waves = []
        for (n, kappa), values in read_spinors(folder, orbitals).items():
            name = f'{label}-{n}-{kappa}'
            arrays[name] = values
            waves.append(dict(n=n, kappa=kappa, array=name))
        matrices = [dict(two_J=b['symmetry']//2, parity=(-1)**(b['symmetry']%2),
                         fac_basis_indices=b['basis'], matrix_Ha=b['hamiltonian'].tolist())
                    for b in parse_hamiltonian_binary(folder/'hamilton.b')['blocks']]
        fixture['cases'].append(dict(z=case['z'], sequence=case['sequence'],
            configurations=case['configurations'], fac_alpha=case['fac_alpha'],
            radial=radial, blocks=matrices, orbitals=waves, raw_sha256=case['raw_sha256']))
    path = destination/'fac_ci_spinors.npz'
    np.savez_compressed(path, **arrays)
    fixture['spinors_sha256'] = sha256(path)
    (destination/'fac_ci_operator.json').write_text(json.dumps(fixture, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fac', type=Path, required=True, help='frontend with DumpCI1E/DumpCIRK/DumpSpinor')
    parser.add_argument('--cases', nargs='+', default=['helium_like:10', 'carbon_like:6', 'neon_like:26', 'neon_like:82'])
    parser.add_argument('--fac-points', type=int, default=8000)
    parser.add_argument('--fac-density', type=float, default=4.)
    parser.add_argument('--atomx-points', type=int, nargs='+', default=[2000, 4000])
    parser.add_argument('--disable-fac-yk-cache', action='store_true',
                        help='use FAC LimitArray(0,0) to bypass float Coulomb-potential caching')
    parser.add_argument('--fixture-dir', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    report = dict(scope=__doc__, controls={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        fac_sha256=sha256(args.fac), runner_sha256=sha256(__file__),
        source_sha256={str(p.relative_to(root)): sha256(p) for p in sorted((root/'src/atomx').rglob('*.py'))},
        threads={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}, cases=[])
    for case in args.cases:
        sequence, z = case.split(':')
        result = run_case(int(z), sequence, args.out/f'{sequence}-Z{int(z):03d}', args)
        report['cases'].append(result)
        (args.out/'results.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(case=case, levels=result['n_levels'], integrals=result['n_integrals'],
            angular_max_Ha=max(abs(x) for b in result['common_integrals'] for x in b['difference_Ha']),
            spinor_max_Ha=max(abs(x) for b in result['refinements'][-1]['spectra'] for x in b['difference_Ha']),
            matched_c_spinor_max_Ha=max(abs(x) for b in result['refinements'][-1]['matched_c_spectra'] for x in b['difference_Ha']))), flush=True)
    if args.fixture_dir:
        write_fixture(report, args.fixture_dir)


if __name__ == '__main__':
    main()
