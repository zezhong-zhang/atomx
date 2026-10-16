"""Full h+V widths, phase covariance, selection and shared inverse capture."""

from dataclasses import replace

import numpy as np
import pytest

from atomx import Atom, Basis
from atomx.config.csf import enumerate_jj_csfs, enumerate_jj_orbital_space
from atomx.constants import CODATA, DEFAULT_CODATA_VERSION
from atomx.methods.ci_blocks import solve_ci_blocks
from atomx.scattering.autoionization.ci import ExplicitCIAutoionization
from atomx.scattering.ci_continuum import TwoElectronCIChannels
from atomx.scattering.models import ContinuumModel
from atomx.scattering.photoionization.ci import ExplicitCIPhotoionization


@pytest.fixture(scope='module')
def native():
    configs = ['1s2', '1s1 2p1', '2s1 2p1']
    atom = Atom(Z=10, config=configs, basis=Basis(ns=40, ks=9, r_max=50), corrections='none')
    atom.solve(max_iter=100, tol=1e-7)
    assert atom.converged
    result = atom.result
    orbitals, data = result.orbitals_dict, result.bsp_data
    parent = tuple(c for cfg in configs for c in enumerate_jj_csfs(cfg))
    daughter = tuple(enumerate_jj_orbital_space(tuple(orbitals), 1, magnetic_state_cap=128))
    sectors = {}
    for n, csfs in ((1, daughter), (2, parent)):
        sectors[n] = csfs, solve_ci_blocks(csfs, orbitals, data, 10, data['ns'], nuclear_potential=result._nuclear_potential)
    model = ContinuumModel.shooting(n_grid=4000, r_max=80, orbital_removal='none')
    kwargs = dict(nuclear_potential=result._nuclear_potential, continuum_model=model, coverage='retained native Ne two-electron control')
    snapshot = ExplicitCIPhotoionization(sectors, orbitals, data, 10, **kwargs)
    return atom, sectors, orbitals, data, kwargs, snapshot


def test_explicit_full_hv_width_matches_independent_two_electron_operator(native):
    atom, _, _, _, kwargs, snapshot = native
    old = TwoElectronCIChannels(atom, continuum_model=kwargs['continuum_model'])
    channels = old.auger()
    new = ExplicitCIAutoionization(snapshot)
    saw_h = saw_v = False
    for i, initial in enumerate(snapshot.levels):
        if initial.electrons != 2:
            continue
        oi = next(k for k, q in old.quantum[2].items() if q == (initial.two_J, initial.parity)
                  and abs(old.sectors[2].state_energies[k]-initial.energy_Ha) < 1e-8)
        for f, final in enumerate(snapshot.levels):
            if final.electrons != 1:
                continue
            of = next(k for k, q in old.quantum[1].items() if q == (final.two_J, final.parity)
                      and abs(old.sectors[1].state_energies[k]-final.energy_Ha) < 1e-8)
            expected = sum(v*v for ch in channels if ch.initial_level == oi and ch.final_level == of
                           for _, v in ch.components)/initial.g
            width = new.partial_width(i, f)
            assert width['partial_rate_per_s'] == pytest.approx(expected, rel=3e-10, abs=1e-12)
            assert width['partial_rate_per_s'] == pytest.approx(
                width['h_only_per_s']+width['V_only_per_s']+width['interference_per_s'], rel=3e-12, abs=1e-12)
            saw_h |= width['h_only_per_s'] > 0
            saw_v |= width['V_only_per_s'] > 0
    assert saw_h and saw_v


def test_inverse_capture_has_one_independent_phase_volume_and_no_branch(native):
    *_, snapshot = native
    op = ExplicitCIAutoionization(snapshot)
    candidates = [(i, f, op.partial_width(i, f)) for i, a in enumerate(snapshot.levels)
                  for f, b in enumerate(snapshot.levels) if a.electrons == b.electrons+1 and a.energy_Ha > b.energy_Ha]
    i, f, width = max(candidates, key=lambda row: row[2]['partial_rate_per_s'])
    C = CODATA[DEFAULT_CODATA_VERSION]
    T = 30.
    wavelength = C['_hplanck']/np.sqrt(2*np.pi*C['_me']*T*C['_e'])
    expected = wavelength**3*1e6*snapshot.levels[i].g/(2*snapshot.levels[f].g)*width['partial_rate_per_s']*np.exp(-width['electron_energy_eV']/T)
    assert op.inverse_capture_cm3_s(i, f, T) == pytest.approx(expected, rel=2e-10)
    assert not width['total_width_certified']


def test_selected_roots_keep_phases_identity_and_preflight(native):
    _, sectors, orbitals, data, kwargs, snapshot = native
    selected = {n: tuple((*b.sector, 0) for b in spec.blocks) for n, (_, spec) in sectors.items()}
    chosen = ExplicitCIPhotoionization(sectors, orbitals, data, 10, **kwargs, state_selection=selected)
    assert len(chosen.levels) == sum(len(v) for v in selected.values())
    assert chosen.source_id != snapshot.source_id
    with pytest.raises(ValueError, match='every supplied sector'):
        ExplicitCIPhotoionization(sectors, orbitals, data, 10, **kwargs, state_selection={1: selected[1]})
    bad = dict(selected)
    bad[2] = ((99, 1, 0),)
    with pytest.raises(ValueError, match='root identities'):
        ExplicitCIPhotoionization(sectors, orbitals, data, 10, **kwargs, state_selection=bad)
    # The root identity remains its original block rank after subset selection.
    parent, spec = sectors[2]
    blocks = tuple(replace(b, coefficients=-b.coefficients) for b in spec.blocks)
    changed = {**sectors, 2: (parent, replace(spec, blocks=blocks))}
    reversed_phase = ExplicitCIPhotoionization(changed, orbitals, data, 10, **kwargs)
    old, new = ExplicitCIAutoionization(snapshot), ExplicitCIAutoionization(reversed_phase)
    for i, a in enumerate(snapshot.levels):
        if a.electrons != 2:
            continue
        for f, b in enumerate(snapshot.levels):
            if b.electrons != 1:
                continue
            left, right = old.partial_width(i, f), new.partial_width(i, f)
            assert left['partial_rate_per_s'] == pytest.approx(right['partial_rate_per_s'], rel=1e-13, abs=0)
            for x, y in zip(left['components'], right['components'], strict=True):
                assert x['h_sqrt_per_s'] == pytest.approx(-y['h_sqrt_per_s'], rel=1e-13, abs=0)
                assert x['V_sqrt_per_s'] == pytest.approx(-y['V_sqrt_per_s'], rel=1e-13, abs=0)
    limited = ExplicitCIAutoionization(snapshot, max_pair_products=1)
    i, f = next((i, f) for i, a in enumerate(snapshot.levels) for f, b in enumerate(snapshot.levels)
                if a.electrons == b.electrons+1 and a.energy_Ha > b.energy_Ha and a.two_J > 0)
    with pytest.raises(MemoryError, match='pair-product'):
        limited.partial_width(i, f)


def test_selection_cannot_split_exact_degeneracy(native):
    _, sectors, orbitals, data, kwargs, _ = native
    csfs, spec = sectors[2]
    b = next(b for b in spec.blocks if len(b.indices) > 1)
    energy = np.full_like(b.energies_Ha, b.energies_Ha[0])
    degenerate = replace(b, hamiltonian_Ha=np.eye(len(energy))*energy[0], energies_Ha=energy, coefficients=np.eye(len(energy)))
    modified = replace(spec, blocks=tuple(degenerate if block is b else block for block in spec.blocks))
    selection = {1: tuple((*x.sector, r) for x in sectors[1][1].blocks for r in range(len(x.indices))), 2: ((*b.sector, 0),)}
    with pytest.raises(ValueError, match='degenerate'):
        ExplicitCIPhotoionization({**sectors, 2: (csfs, modified)}, orbitals, data, 10, **kwargs, state_selection=selection)


def test_highest_weight_streamed_width_is_the_full_unpolarized_scalar(native):
    _, sectors, orbitals, data, kwargs, complete = native
    highest = ExplicitCIPhotoionization(sectors, orbitals, data, 10, **kwargs,
                         magnetic_mode='highest-parent', projection_engine='streamed')
    a, b = ExplicitCIAutoionization(complete), ExplicitCIAutoionization(highest)
    saw_nonzero_J = False
    for i, initial in enumerate(complete.levels):
        if initial.electrons != 2:
            continue
        assert len(highest._vectors[i]) == 1
        for f, final in enumerate(complete.levels):
            if final.electrons != 1:
                continue
            old, new = a.partial_width(i, f), b.partial_width(i, f)
            assert old['partial_rate_per_s'] == pytest.approx(new['partial_rate_per_s'], rel=2e-11, abs=1e-12)
            saw_nonzero_J |= initial.two_J > 0 and old['partial_rate_per_s'] > 0
        if initial.two_J > 0:
            target = next(f for f, final in enumerate(complete.levels) if final.electrons == 1 and final.energy_Ha > initial.energy_Ha)
            with pytest.raises(ValueError, match='complete initial'):
                highest.cross_section_cm2(i, target, 10000)
    assert saw_nonzero_J


def test_compiled_reduced_channels_match_every_magnetic_hv_sum(native):
    *_, snapshot = native
    magnetic = ExplicitCIAutoionization(snapshot)
    compiled = ExplicitCIAutoionization(snapshot, engine='compiled-reduced')
    for i, a in enumerate(snapshot.levels):
        for f, b in enumerate(snapshot.levels):
            if a.electrons != b.electrons+1:
                continue
            expected, actual = magnetic.partial_width(i, f), compiled.partial_width(i, f)
            for key in ('partial_rate_per_s','h_only_per_s','V_only_per_s','interference_per_s'):
                assert actual[key] == pytest.approx(expected[key], rel=3e-10, abs=1e-12)
    limited = ExplicitCIAutoionization(snapshot,engine='compiled-reduced',max_kernel_steps=1)
    i,f = next((i,f) for i,a in enumerate(snapshot.levels) for f,b in enumerate(snapshot.levels)
               if a.electrons==b.electrons+1 and a.energy_Ha>b.energy_Ha)
    with pytest.raises(MemoryError,match='work cap'):
        limited.partial_width(i,f)


def test_analytic_all_m_reduction_keeps_scalar_widths_and_density(native):
    _,sectors,orbitals,data,kwargs,snapshot=native
    reduced=ExplicitCIPhotoionization(sectors,orbitals,data,10,**kwargs,
                         magnetic_mode='highest-all',projection_engine='streamed')
    a=ExplicitCIAutoionization(snapshot)
    b=ExplicitCIAutoionization(reduced,engine='compiled-reduced')
    for i,initial in enumerate(snapshot.levels):
        assert len(reduced._vectors[i])==1
        if initial.electrons==1:
            np.testing.assert_allclose(reduced._densities[i],snapshot._densities[i],atol=2e-14,rtol=2e-13)
            continue
        for f,final in enumerate(snapshot.levels):
            if final.electrons==1:
                assert b.partial_width(i,f)['partial_rate_per_s']==pytest.approx(a.partial_width(i,f)['partial_rate_per_s'],rel=3e-10,abs=1e-12)
    with pytest.raises(ValueError,match='scalar reduction'):
        ExplicitCIAutoionization(reduced)


def test_compiled_closed_core_contraction_matches_full_four_electron_hv():
    configs=['1s2 2s2','1s2 2p2','1s2 2s1 3p1','1s2 3s1 3p1']
    atom=Atom(Z=8,config=configs,basis=Basis(ns=40,ks=9,r_max=50),corrections='none')
    atom.solve(max_iter=100,tol=1e-7)
    assert atom.converged
    result=atom.result
    sectors={}
    for n,groups in ((4,configs),(3,['1s2 2s1','1s2 2p1'])):
        csfs=tuple(c for cfg in groups for c in enumerate_jj_csfs(cfg))
        spec=solve_ci_blocks(csfs,result.orbitals_dict,result.bsp_data,8,result.bsp_data['ns'],
                             nuclear_potential=result._nuclear_potential)
        sectors[n]=csfs,spec
    s=ExplicitCIPhotoionization(sectors,result.orbitals_dict,result.bsp_data,8,
         nuclear_potential=result._nuclear_potential,
         continuum_model=ContinuumModel.shooting(n_grid=4000,r_max=80,orbital_removal='none'),
         coverage='native four-electron closed1s2 core control')
    direct=ExplicitCIAutoionization(s)
    reduced=ExplicitCIAutoionization(s,engine='compiled-reduced')
    count=0
    for i,a in enumerate(s.levels):
        for f,b in enumerate(s.levels):
            if a.electrons!=b.electrons+1 or a.energy_Ha<=b.energy_Ha:
                continue
            old,new=direct.partial_width(i,f),reduced.partial_width(i,f)
            for name in ('partial_rate_per_s','h_only_per_s','V_only_per_s','interference_per_s'):
                assert new[name]==pytest.approx(old[name],rel=5e-10,abs=1e-12)
            assert len(reduced._action(i,f)[2])>=2
            count+=new['partial_rate_per_s']>0
    assert count>0
