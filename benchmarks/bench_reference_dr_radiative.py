"""Matched FAC/Atomx frozen-CI radiative stabilization of Ne-like dr33 resonances.

Four separately mixed hole/parity spaces, all n=3 spectator final states,
point nuclei, infinite mass, Dirac-Coulomb. Each code optimizes its own
local ground potential. Every FAC photon call has one upper and one lower
level, so no transition-frequency averaging enters the operator comparison.
This panel tests radiative stabilization; it does not validate Auger continua.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import re
from types import SimpleNamespace

import numpy as np
from atomx import Atom, Basis, Sampling
from atomx.config.csf import enumerate_jj_csfs_many
from atomx.constants import HARTREE_EV, chemical_symbols
from atomx.scattering.autoionization.mscheme_ci import manifold_from_configs
from benchmarks.bench_reference_structure import run_reference, sha256
from benchmarks.fac_runner import (parse_levels, parse_transitions, parse_ai,
    radial_grid_command, orbital_accuracy_commands)

GROUPS = {
    'p_even': ['2p5 3s1 3p1', '2p5 3p1 3d1'],
    'p_odd': ['2p5 3s2', '2p5 3p2', '2p5 3d2', '2p5 3s1 3d1'],
    's_even': ['2s1 3s2', '2s1 3p2', '2s1 3d2', '2s1 3s1 3d1'],
    's_odd': ['2s1 3s1 3p1', '2s1 3p1 3d1'],
}
MODES = {'finite_wavelength': 0, 'long_wavelength': 1}
FINAL_KAPPAS = (-1, 1, -2, 2, -3)
ORBITALS = [(1,-1),(2,-1),(2,1),(2,-2)]+[(3,k) for k in FINAL_KAPPAS]
ONE_BODY_KEYS = [('h',a[0],a[1],b[0]) for a in ORBITALS for b in ORBITALS if a[1]==b[1]]


def full_configs(name):
    return [('1s2 2s2 ' if name.startswith('p') else '1s2 2p6 ')+cfg
            for cfg in GROUPS[name]]


def inventories():
    start = 6
    result = {}
    for name in GROUPS:
        csfs = enumerate_jj_csfs_many(full_configs(name))
        result[name] = dict(ids=list(range(start, start+len(csfs))),
                           two_J=sorted(c.total_J2 for c in csfs))
        start += len(csfs)
    return result


def deck(z, points, density, *, common_spinors=False, prune_mixtures=False, include_auger=False):
    counts = inventories()
    lines = [f"SetAtom('{chemical_symbols[z]}',{z},-1,0)",
        'SetSE(0)', 'SetVP(0)', 'SetMS(0,0)', 'SetBreit(0)',
        'SetConfigEnergyMode(-1)', radial_grid_command(points, density_scale=density),
        *orbital_accuracy_commands(1e-12),
        'SetOptimizeTolerance(1e-4)', 'SetOptimizeMaxIter(150)', 'SetOptimizePrint(1)',
        "Closed('1s')", "Config('gnd','2s2 2p6')", "Config('final','2s2 2p6 3*1')"]
    if not prune_mixtures:
        lines += ['SetMixCut(0)', 'SetAngZCut(0)']
    if common_spinors:
        lines.append('LimitArray(0,0)')
    for name in GROUPS:
        for config in full_configs(name):
            lines.append(f"Config('{name}','{config.removeprefix('1s2 ')}')")
    lines += ["OptimizeRadial(['gnd'])", "Structure('levels.b',['gnd'])",
              "Structure('levels.b',['final'])"]
    for name in GROUPS:
        lines.append(f"Structure('levels.b',['{name}'])")
    lines += ["MemENTable('levels.b')", "PrintTable('levels.b','levels.txt',1)",
              'SetTransitionCut(0)']
    for name, mode in MODES.items():
        lines.append(f'SetTransitionOptions(2,{mode})')
        for inventory in counts.values():
            for upper in inventory['ids']:
                for lower in range(1, 6):
                    lines.append(f"TRTable('lines.b',[{lower}],[{upper}],-1)")
    lines.append("PrintTable('lines.b','lines.txt',1)")
    if include_auger:
        lines.append('SetAICut(0)')
        for inventory in counts.values():
            for upper in inventory['ids']:
                lines.append(f"AITable('auger.b',[{upper}],[0])")
        lines.append("PrintTable('auger.b','auger.txt',1)")
    if common_spinors:
        for index, (_, na, kappa, nc) in enumerate(ONE_BODY_KEYS):
            lines.append(f"DumpCI1E('one_body.txt',{index},{na},{kappa},{nc},{kappa})")
        lines += [f"DumpSpinor('wave-{n}-{k}.txt',{n},{k})" for n,k in ORBITALS]
    return '\n'.join(lines)+'\n'


def fac_reference(folder):
    levels = {r['ILEV']: r for r in parse_levels(folder/'levels.txt')}
    expected = 6+sum(len(v['ids']) for v in inventories().values())
    if sorted(levels) != list(range(expected)):
        raise ValueError('FAC level inventory mismatch')
    groups = {}
    for name, inventory in inventories().items():
        group = [levels[i] for i in inventory['ids']]
        if sorted(r['TWOJ'] for r in group) != inventory['two_J']:
            raise ValueError(f'FAC J inventory mismatch: {name}')
        if {r['P'] for r in group} != {int(name.endswith('odd'))}:
            raise ValueError(f'FAC parity inventory mismatch: {name}')
        groups[name] = group
    final = {}
    for i in range(1, 6):
        row = levels[i]
        candidates = [k for k in FINAL_KAPPAS if 2*abs(k)-1 == row['TWOJ']
                      and (k if k > 0 else -k-1) % 2 == row['P']]
        if len(candidates) != 1:
            raise ValueError('ambiguous final state identity')
        final[i] = candidates[0]
    if sorted(final.values()) != sorted(FINAL_KAPPAS):
        raise ValueError('incomplete final-state inventory')
    lines = parse_transitions(folder/'lines.txt')
    if any(r['mult'] != -1 or r['gauge'] != 'length'
           or r['upper'] < 6 or r['lower'] not in final for r in lines):
        raise ValueError('FAC photon convention mismatch')
    return dict(ground=levels[0], groups=groups,
        final=[dict(levels[i], kappa=k) for i,k in final.items()],
        lines={name: [r for r in lines if r['mode'] == mode] for name,mode in MODES.items()},
        auger=parse_ai(folder/'auger.txt') if (folder/'auger.txt').exists() else None)


def common_manifolds(folder, z, points):
    from benchmarks.bench_reference_ci_operator import read_spinors

    alpha = float(re.search(r'alpha=([^\s]+)', (folder/'wave-1--1.txt').open().readline()).group(1))
    sources = read_spinors(folder, ORBITALS)
    bare = np.loadtxt(folder/'one_body.txt')
    if not np.array_equal(bare[:,0], np.arange(len(ONE_BODY_KEYS))):
        raise ValueError('FAC bare-integral inventory mismatch')
    one_body = dict(zip(ONE_BODY_KEYS, bare[:,1], strict=True))
    return common_manifolds_from_data(sources,z,points,alpha,one_body)


def common_manifolds_from_data(sources, z, points, alpha, one_body):
    from benchmarks.bench_reference_ci_operator import common_grid
    from atomx.physics.csf.matrix import _radial_evaluator
    from atomx.physics.csf.frozen_core import closed_core_one_body
    from atomx.numerics.radial_workspace import make_grid_workspace
    from atomx.numerics.shooting.quadrature import slag6_weights
    from atomx.scattering.autoionization.mscheme_ci import ResonanceManifold, _parse_config_tokens

    waves, ws = common_grid(sources,z,points,alpha=alpha)
    native = _radial_evaluator(waves, ws, z, ws.n)
    radial = lambda key: one_body[key] if key[0]=='h' else native(key)
    property_workspace = make_grid_workspace(ws.r[1:], (slag6_weights(ws.n)*ws.difr)[1:])
    bound = {key:(p[1:],q[1:]) for key,(p,q) in waves.items()}
    manifolds = {}
    for name, configs in GROUPS.items():
        hole = [(2,1),(2,-2)] if name.startswith('p') else [(2,-1)]
        hole_l = int(name.startswith('p'))
        groups = [(hole, 5 if hole_l else 1), ([(3,-1)],0),
                  ([(3,1),(3,-2)],0), ([(3,2),(3,-3)],0)]
        active = {k for keys,_ in groups for k in keys}
        core = {k:2*abs(k[1]) for k in ORBITALS if k[0]<3 and k not in active}
        occupations = []
        for config in configs:
            parsed = _parse_config_tokens(config)
            occupations.append((parsed[2,hole_l], *(parsed.get((3,l),0) for l in range(3))))
        manifolds[name] = ResonanceManifold(SimpleNamespace(Z=z),groups,
            r_fine=ws.r[1:],dr_fine=property_workspace['grw'],orbital_map=bound,
            orbital_energies={k:0. for k in active},configs=occupations,
            one_body=closed_core_one_body(active,core,radial),bound_radial=radial,
            bound_orbitals=bound,bound_workspace=property_workspace)
    return manifolds


def calculate(z, points, *, common_folder=None, manifolds=None, include_auger=False):
    start = time.perf_counter()
    if manifolds is not None:
        supplied = manifolds
    elif common_folder is None:
        atom = Atom(z, charge=z-10, config='1s2 2s2 2p6', engine='shooting',
            exchange='gu', corrections='none', nuclear='point',
            basis=Basis(sampling=Sampling(n_grid=points)))
        atom.solve(tol=1e-10, max_iter=150)
        if not atom.converged:
            raise RuntimeError('Atomx ground SCF did not converge')
        supplied = None
    else:
        supplied = common_manifolds(common_folder,z,points)
    timings = dict(orbital_setup_s=time.perf_counter()-start)
    groups = {}
    for name, configs in GROUPS.items():
        start = time.perf_counter()
        if supplied is None:
            man, bsp = manifold_from_configs(atom.result, configs)
        else:
            man = supplied[name]
        ion_energy = man.ion_energy()
        rows = []
        tm = 1
        while dets := man.determinants(tm):
            h = man.hamiltonian(dets)
            energies, vectors = np.linalg.eigh(h)
            j2 = man.j2_matrix(dets)
            for index, energy in enumerate(energies):
                weights = vectors[:, index]
                angular = float(weights @ j2 @ weights)
                twoj = int(round(np.sqrt(1+4*angular)-1))
                if abs(angular-twoj*(twoj+2)/4) > 1e-6:
                    raise ValueError(f'CI state has unresolved J: {name}, {tm}, {energy}')
                if twoj != tm:
                    continue
                channels = {mode: [asdict(c) for c in man.radiative_channels(
                    dets, weights, energy, photon_approximation=mode)] for mode in MODES}
                row=dict(two_J=twoj, energy_eV=float((energy-ion_energy)*HARTREE_EV), channels=channels)
                if include_auger:
                    rates=man.auger_widths(tm,row['energy_eV'],Z=z,bsp_data=bsp,
                        _only_level=index,_precomputed=(dets,h,j2,energies,vectors))
                    if len(rates)!=1:
                        raise ValueError('Auger state inventory mismatch')
                    row['auger_per_s']=rates[0].gamma_auger_per_s
                rows.append(row)
            tm += 2
        groups[name] = sorted(rows, key=lambda r: (r['two_J'],r['energy_eV']))
        timings[name+'_s'] = time.perf_counter()-start
    return dict(groups=groups, timings=timings)


def compare(reference, calculated):
    matches = []
    final_keys = {r['ILEV']: r['kappa'] for r in reference['final']}
    for name, rows in reference['groups'].items():
        fac = sorted(rows, key=lambda r:(r['TWOJ'],r['ENERGY_eV']))
        native = calculated['groups'][name]
        if len(fac) != len(native):
            raise ValueError(f'level dimensions differ: {name}')
        for external, local in zip(fac, native, strict=True):
            if external['TWOJ'] != local['two_J']:
                raise ValueError('symmetry mismatch')
            record = dict(group=name, fac_level=external['ILEV'], two_J=local['two_J'],
                fac_energy_eV=external['ENERGY_eV']-reference['ground']['ENERGY_eV'],
                atomx_energy_eV=local['energy_eV'], modes={})
            record['energy_difference_eV']=record['atomx_energy_eV']-record['fac_energy_eV']
            if reference.get('auger') is not None:
                rate=sum(r['rate_per_s'] for r in reference['auger'] if r['upper']==external['ILEV'])
                record['auger']=dict(fac_per_s=rate,atomx_per_s=local['auger_per_s'],
                                    ratio=local['auger_per_s']/rate if rate else None)
            for mode in MODES:
                fac_lines = {final_keys[r['lower']]:r for r in reference['lines'][mode]
                             if r['upper'] == external['ILEV']}
                ax = {r['final_kappa']:r for r in local['channels'][mode]}
                lines = []
                for k in FINAL_KAPPAS:
                    rate = fac_lines.get(k, {}).get('A_per_s',0.)
                    calc = ax.get(k, {}).get('A_per_s',0.)
                    lines.append(dict(final_kappa=k, fac_A_per_s=rate, atomx_A_per_s=calc,
                        ratio=calc/rate if rate else None,
                        photon_difference_eV=(ax[k]['omega_au']*HARTREE_EV-fac_lines[k]['DELTA_E_eV']
                                              if rate and k in ax else None)))
                fac_width=sum(r['fac_A_per_s'] for r in lines)
                ax_width=sum(r['atomx_A_per_s'] for r in lines)
                record['modes'][mode]=dict(lines=lines, fac_width_per_s=fac_width,
                    atomx_width_per_s=ax_width, ratio=ax_width/fac_width if fac_width else None)
            matches.append(record)
    return matches


def write_fixture(report, destination):
    """Export only independent FAC values; reuse byte-identical existing spinors."""
    from benchmarks.bench_reference_ci_operator import read_spinors
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/'tests/data/fac_ci_operator.json').read_text())
    fixture = dict(scope=__doc__, controls=report['controls'],
        fac_sha256=report['fac_sha256'], runner_sha256=report['runner_sha256'],
        spinors_file='fac_ci_spinors.npz', spinors_sha256=manifest['spinors_sha256'], cases=[])
    with np.load(root/'tests/data/fac_ci_spinors.npz') as saved:
        for case in report['cases']:
            z = case['z']
            previous = next(c for c in manifest['cases'] if c['sequence']=='neon_like' and c['z']==z)
            folder = Path(report['controls']['out'])/f'Z{z:03d}'
            sources = read_spinors(folder,ORBITALS)
            for orbital in previous['orbitals']:
                if not np.array_equal(saved[orbital['array']],sources[orbital['n'],orbital['kappa']]):
                    raise ValueError('existing FAC spinors differ; export a separate native fixture')
            bare = np.loadtxt(folder/'one_body.txt')
            fixture['cases'].append(dict(z=z,reference=case['reference'],
                orbitals=previous['orbitals'],fac_alpha=previous['fac_alpha'],
                bare_one_body=[dict(key=key,value_Ha=float(value))
                    for key,value in zip(ONE_BODY_KEYS,bare[:,1],strict=True)],
                raw_sha256=case['raw_sha256']))
    destination.write_text(json.dumps(fixture,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--fac',type=Path,required=True)
    parser.add_argument('--z',type=int,nargs='+',default=[26,82])
    parser.add_argument('--fac-points',type=int,default=16000)
    parser.add_argument('--fac-density',type=float,default=8.)
    parser.add_argument('--atomx-points',type=int,default=4000)
    parser.add_argument('--common-spinors',action='store_true',help='export FAC spinors/bare h; requires diagnostic FAC frontend')
    parser.add_argument('--include-auger',action='store_true',help='also compare ground-channel autoionization for native orbitals')
    parser.add_argument('--fixture',type=Path,help='export independent common-spinor reference data')
    parser.add_argument('--fac-prune-mixtures',action='store_true',help='diagnostic: retain FAC default small CI/transition coefficient cuts')
    args=parser.parse_args()
    if args.include_auger and args.common_spinors:
        parser.error('common-spinor Auger comparison also needs the reference continuum potential')
    if args.fixture is not None and not args.common_spinors:
        parser.error('--fixture requires --common-spinors')
    args.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    report=dict(scope=__doc__,controls={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        fac_sha256=sha256(args.fac),source_sha256={str(p.relative_to(root)):sha256(p)
            for p in sorted((root/'src/atomx').rglob('*.py'))},runner_sha256=sha256(__file__),cases=[])
    for z in args.z:
        folder=args.out/f'Z{z:03d}';folder.mkdir()
        (folder/'run.sf').write_text(deck(z,args.fac_points,args.fac_density,common_spinors=args.common_spinors,prune_mixtures=args.fac_prune_mixtures,include_auger=args.include_auger))
        elapsed=run_reference([str(args.fac.resolve()),'run.sf'],folder,timeout=600)
        output=(folder/'stdout.txt').read_text()
        loops=[line.split(':',1)[1].split() for line in output.splitlines() if line.startswith('optimize loop:')]
        if not loops or max(int(r[0]) for r in loops)>=150 or 'Maximum iteration reached' in output:
            raise RuntimeError('FAC ground optimization convergence not established')
        ref=fac_reference(folder)
        print(json.dumps(dict(z=z,fac_s=elapsed,fac_lines={k:len(v) for k,v in ref['lines'].items()})),flush=True)
        ax=calculate(z,args.atomx_points,common_folder=folder if args.common_spinors else None,include_auger=args.include_auger)
        rows=compare(ref,ax)
        case=dict(z=z,reference=ref,atomx=ax,comparison=rows,fac_process_s=elapsed,
                  raw_sha256={p.name:sha256(p) for p in folder.iterdir() if p.is_file()})
        report['cases'].append(case)
        (args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        strong=[r['modes']['finite_wavelength']['ratio'] for r in rows
                if r['modes']['finite_wavelength']['fac_width_per_s']>1e10]
        print(json.dumps(dict(z=z,levels=len(rows),max_energy_difference_eV=max(abs(r['energy_difference_eV']) for r in rows),
            strong_width_ratio_range=[min(strong),max(strong)],timings=ax['timings'])),flush=True)

    if args.fixture is not None:
        write_fixture(report,args.fixture)


if __name__=='__main__':main()
