"""Private full public frozen-CI workflow, with complete diagnostics retained."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

START = time.perf_counter()
import numpy as np
from atomx import Atom, Basis, Sampling
from atomx.properties.transitions import ci_lines
from atomx.scattering.autoionization.mscheme_ci import ResonanceManifold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--z', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    stages = {'import_seconds': time.perf_counter()-START}
    arrays = {}
    counts = {'H': 0, 'eig': 0, 'amp': 0}
    original_h = ResonanceManifold.hamiltonian
    original_eig = ci_lines._PairCI.eig_block
    original_amp = ci_lines._PairCI.amplitude
    def capture_h(self, dets):
        h = original_h(self, dets)
        index = counts['H']; counts['H'] += 1
        arrays[f'H_{index:04d}'] = h.copy()
        arrays[f'H_dets_{index:04d}'] = np.array(dets, dtype=np.uint64)
        return h
    def capture_eig(self, tup, tm):
        fresh = (tuple(tup), int(tm)) not in self._eig_cache
        block = original_eig(self, tup, tm)
        if fresh and block is not None:
            index = counts['eig']; counts['eig'] += 1
            for key, value in zip(('dets','eigenvalues','eigenvectors','twoJ'), block, strict=True):
                arrays[f'{key}_{index:04d}'] = np.array(value)
        return block
    def capture_amp(self, *parts):
        value = original_amp(self, *parts)
        index = counts['amp']; counts['amp'] += 1
        arrays[f'signed_amplitude_{index:04d}'] = np.array(value)
        return value
    ResonanceManifold.hamiltonian = capture_h
    ci_lines._PairCI.eig_block = capture_eig
    ci_lines._PairCI.amplitude = capture_amp
    t0 = time.perf_counter()
    atom = Atom(args.z, charge=args.z-10, config='1s2 2s2 2p6',
                engine='shooting', exchange='gu', nuclear='point',
                corrections='none', basis=Basis(sampling=Sampling(n_grid=2000)))
    stages['construct_seconds'] = time.perf_counter()-t0
    t0 = time.perf_counter()
    atom.solve(tol=1e-10, max_iter=150)
    stages['ground_scf_seconds'] = time.perf_counter()-t0
    if not atom.converged:
        raise RuntimeError(f'Atomx ground not converged Z={args.z}')
    modes = {}
    for mode in ('long_wavelength','finite_wavelength'):
        modes[mode] = {}
        for config in ('2p5 3s1','2p5 3d1'):
            t0 = time.perf_counter()
            lines = ci_lines.ci_line_strengths(atom.result, config, '2p6',
                                              photon_approximation=mode)
            stages[f'{mode}_{config}_virtual_CI_E1_seconds'] = time.perf_counter()-t0
            modes[mode][config] = {
                'upper_levels': [asdict(v) for v in lines.levels_i],
                'lower_levels': [asdict(v) for v in lines.levels_f],
                'lines': [asdict(v) for v in lines.lines],
                'S_sum': lines.S_sum, 'trace': lines.trace,
            }
            man = lines.manifold
            prefix = f'{mode}_{config.replace(" ","_")}'
            arrays[prefix+'_r_fine'] = np.array(man.r_fine)
            arrays[prefix+'_dr_fine'] = np.array(man.dr_fine)
            arrays[prefix+'_orbital_energies'] = np.array(man.eps)
            for key, value in sorted(man.bound_radial.items()) if isinstance(man.bound_radial,dict) else []:
                if isinstance(value, np.ndarray):
                    arrays[prefix+'_radial_'+str(key)] = value.copy()
    stages['science_seconds_excluding_import_export'] = sum(v for k,v in stages.items() if k != 'import_seconds')
    t0 = time.perf_counter()
    np.savez(args.out/'complete_arrays.npz', **arrays)
    stages['array_export_seconds'] = time.perf_counter()-t0
    result = {'z': args.z, 'converged': bool(atom.converged), 'modes': modes,
              'counts': counts, 'arrays': len(arrays), 'stages': stages,
              'model': 'independently optimized Gu local point-nucleus DC, frozen configuration CI; 2000 ground grid points'}
    (args.out/'science.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'z':args.z,'stages':stages,'arrays':len(arrays)}), flush=True)

if __name__ == '__main__':
    main()
