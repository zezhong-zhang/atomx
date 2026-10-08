"""Native configuration-resolved Gu rates for small XFEL occupation graphs.

Every integer jj configuration has its own relaxed SCF. Empty orbitals
remain in the basis so a completely empty shell can accept an Auger
refill. PI is occupation weighted once; LS Auger source-pair populations
are partitioned among jj daughters once. This is a configuration-average
local-field model, not a level-resolved relativistic CI decay model.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from atomx.config.subshell import Subshell
from atomx.constants import BOHR_RADIUS_M, HARTREE_EV
from atomx.xfel.cascade import _table_key, config_label
from atomx.xfel.table_mc import CompiledRateTable, _occupation_key


def _ls(shell):
    n, k = shell
    return n, k if k > 0 else -k - 1


def ls_pair_partition(config, source, source2):
    """Fraction of the LS donor population in this unordered jj pair.

    The underlying LS rate already includes Q(Q-1)/[g(g-1)] or QaQb/
    (gagb). These fractions distribute that population, without counting
    it a second time. Same-shell, unlike-j pairs carry both orderings.
    """
    q, q2 = config.get(source, 0), config.get(source2, 0)
    total = sum(v for k, v in config.items() if _ls(k) == _ls(source))
    total2 = sum(v for k, v in config.items() if _ls(k) == _ls(source2))
    if _ls(source) == _ls(source2):
        if total < 2:
            return 0.0
        numerator = q * (q - 1) if source == source2 else 2 * q * q2
        return numerator / (total * (total - 1))
    return q * q2 / (total * total2) if total * total2 else 0.0


def auger_candidates(config, shells):
    """All occupation-legal channels, including initially empty holes.

    Final native total-energy differences, not lexical shell ordering,
    determine whether a candidate is energetically open.
    """
    for hole in shells:
        vacancies = 2 * abs(hole[1]) - config.get(hole, 0)
        if vacancies <= 0:
            continue
        donors = [s for s in shells if s != hole and config.get(s, 0) > 0]
        for a, b in itertools.combinations_with_replacement(donors, 2):
            partition = ls_pair_partition(config, a, b)
            if not partition or (a == b and config[a] < 2):
                continue
            daughter = dict(config)
            for shell, delta in ((hole, 1), (a, -1), (b, -1)):
                daughter[shell] = daughter.get(shell, 0) + delta
            yield hole, a, b, vacancies * partition, {
                s: q for s, q in daughter.items() if q
            }


def _source_identity():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _state_path(directory, config):
    token = hashlib.sha256(_table_key(config).encode()).hexdigest()[:20]
    return Path(directory) / token


def _solve_state(job):
    from atomx.methods.dirac_fock import DFSolver, SCFResult
    config, shells, controls, directory, identity = job
    path = _state_path(directory, config)
    meta_path, result_path = path.with_suffix('.json'), path.with_suffix('.h5')
    if meta_path.exists() and result_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta['identity'] != identity:
            raise ValueError(f'incompatible SCF checkpoint: {meta_path}')
        if meta['h5_sha256'] != hashlib.sha256(result_path.read_bytes()).hexdigest():
            raise ValueError(f'corrupt SCF checkpoint: {result_path}')
        result = SCFResult.load(str(result_path))
    else:
        solver = DFSolver(
            controls['Z'], [Subshell(n, k, config.get((n, k), 0)) for n, k in shells],
            exchange='gu', ns=controls['ns'], ks=controls['ks'],
            r_max=controls['r_max'], nuclear='point', starting_potential='tfda',
        )
        result = solver.solve(max_iter=300, tol=1e-9, damping=.5, mode='orbital')
        if not result.converged or not np.isfinite(result.total_energy):
            raise RuntimeError(f'SCF failed for {_table_key(config)}')
        temp = path.with_suffix('.tmp.h5')
        result.save(str(temp))
        temp.replace(result_path)
        meta = dict(identity=identity, energy_hartree=float(result.total_energy),
                    converged=bool(result.converged), iterations=int(result.iterations),
                    orbital_energies_hartree=result.orbital_energies,
                    h5_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest())
        _write_json(meta_path, meta)
    if not result.converged:
        raise RuntimeError(f'unconverged checkpoint: {result_path}')
    return _table_key(config), meta


def _rate_state(job):
    from atomx.methods.dirac_fock import SCFResult
    from atomx.scattering.autoionization import auger_rate_csf_average
    from atomx.scattering.models import ContinuumModel
    from atomx.scattering.photoionization.subshell import subshell_photoionization
    config, shells, controls, directory, identity, energies = job
    path = _state_path(directory, config)
    row_path = path.with_suffix('.rates.json')
    if row_path.exists():
        cached = json.loads(row_path.read_text())
        if cached['identity'] != identity:
            raise ValueError(f'incompatible rate checkpoint: {row_path}')
        return _table_key(config), cached['row']
    result = SCFResult.load(str(path.with_suffix('.h5')))
    labels = {(s.n, s.kappa): s.label for s in result.subshells}
    # CSF-average decay consumes coefficient pairs, unlike SCFResult's triples.
    bound = result.orbitals_by_subshell()
    decay_result = SimpleNamespace(
        Z=result.Z, bsp_data=result._bsp_data,
        orbitals_dict={key: pair[:2] for key, pair in bound.items()},
        occupations={key: pair[2] for key, pair in bound.items()},
        orbital_energies=result.orbital_energies,
    )
    initial = energies[_table_key(config)]
    thresholds = {}
    for shell, q in config.items():
        if q:
            daughter = dict(config)
            daughter[shell] -= 1
            thresholds[labels[shell]] = (energies[_table_key(daughter)] - initial) * HARTREE_EV
    pi = subshell_photoionization(
        result, [controls['photon_energy_eV']],
        continuum_model=ContinuumModel.fac_frozen_core(
            n_grid=controls['pi_grid'], r_max=controls['r_max']),
        thresholds_eV=thresholds,
    )
    sigmas = {f'{n},{k}': float(pi.sigma_by_orbital_a0_sq[labels[n, k]][0])
              * (BOHR_RADIUS_M * 100)**2 for n, k in config}
    channels = []
    for h, a, b, weight, daughter in auger_candidates(config, shells):
        energy = (initial - energies[_table_key(daughter)]) * HARTREE_EV
        if energy <= 0:
            continue
        decay = auger_rate_csf_average(
            decay_result, hole_label=labels[h], source_label=labels[a], source2_label=labels[b],
            resonance_energy_eV=energy, shooting_n_grid=controls['auger_grid'],
        )
        rate = float(decay['Gamma_per_s']) * weight
        if not np.isfinite(rate) or rate < 0:
            raise RuntimeError(f'invalid Auger rate for {_table_key(config)}')
        if rate:
            channels.append(dict(hole=list(h), s1=list(a), s2=list(b), rate=rate,
                                 energy_eV=energy, ls_pair_and_vacancy_weight=weight))
    row = dict(label=config_label(config),
               occupations={f'{n},{k}': q for (n, k), q in config.items()},
               sigma_cm2=sigmas, auger=channels, thresholds_eV=thresholds,
               energy_hartree=initial, pi_provenance=pi.provenance)
    _write_json(row_path, dict(identity=identity, row=row))
    return _table_key(config), row


def _map_jobs(worker, jobs, workers, progress, phase):
    # Warm one matching signature before fork: avoids competing Numba JITs.
    key, value = worker(jobs[0])
    output = {key: value}
    if progress:
        progress(phase, len(output), len(jobs))
    if workers == 1:
        iterator = map(worker, jobs[1:])
        for key, value in iterator:
            output[key] = value
            if progress:
                progress(phase, len(output), len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(worker, job) for job in jobs[1:]]
            for future in as_completed(futures):
                key, value = future.result()  # failures abort; no empty-row fallback
                output[key] = value
                if progress:
                    progress(phase, len(output), len(jobs))
    return output


def build_atomic_rate_table(Z, base_config, photon_energy_eV, *, checkpoint_dir,
                            ns=180, ks=9, r_max=80, pi_grid=16000,
                            auger_grid=400, workers=1, progress=None):
    """Build a complete relaxed-Gu graph with native Delta-SCF energies.

    Intended for small closed-subshell base graphs (Ne: 134 SCFs). The
    caller should pin BLAS/OpenMP threads before importing atomx when
    using multiple processes. Checkpoints reject changed code/controls.
    Radiation and finite pulse-time decay kinetics are not included.
    """
    _occupation_key(base_config)
    if not base_config or any(q != 2 * abs(k) for (n, k), q in base_config.items()):
        raise ValueError('native graph builder requires a filled-subshell base')
    if sum(base_config.values()) > Z or not np.isfinite(photon_energy_eV) or photon_energy_eV <= 0:
        raise ValueError('invalid nuclear charge or photon energy')
    if workers < 1 or ns < ks or ks < 2 or pi_grid < 20 or auger_grid < 20 or r_max <= 0:
        raise ValueError('invalid solver controls')
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    shells = tuple(sorted(base_config))
    configs = [{s: q for s, q in zip(shells, qs, strict=True) if q}
               for qs in itertools.product(*(range(int(base_config[s]) + 1) for s in shells))
               if any(qs)]
    configs.sort(key=lambda c: (-sum(c.values()), _table_key(c)))
    controls = dict(Z=int(Z), photon_energy_eV=float(photon_energy_eV), ns=ns, ks=ks,
                    r_max=r_max, pi_grid=pi_grid, auger_grid=auger_grid)
    identity = dict(source_sha256=_source_identity(), controls=controls,
                    base_occupations={f'{n},{k}': q for (n, k), q in base_config.items()})
    structure = _map_jobs(_solve_state,
                          [(c, shells, controls, str(directory), identity) for c in configs],
                          workers, progress, 'scf')
    energies = {key: meta['energy_hartree'] for key, meta in structure.items()}
    energies[''] = 0.0
    rate_configs = sorted(configs, key=lambda c: not any(auger_candidates(c, shells)))
    rows = _map_jobs(_rate_state,
                     [(c, shells, controls, str(directory), identity, energies) for c in rate_configs],
                     workers, progress, 'rates')
    table = dict(photon_energy_eV=float(photon_energy_eV), base_nelec=sum(base_config.values()),
                 configs={_table_key(c): rows[_table_key(c)] for c in configs}, failed=[],
                 provenance=dict(identity=identity, model='per-configuration relaxed Gu; LS Auger',
                                 energy_source='native total-energy differences',
                                 pi_field='frozen initial SCF snapshot',
                                 auger_field='final occupations on initial bound orbitals',
                                 scf=structure, radiation=False))
    CompiledRateTable(table, base_config)  # graph closure and all rate/unit gates
    return table
