"""Independent exterior-power oracle for CI comparisons under orbital rotation."""
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest

from atomx.angular.csf_hamiltonian import csf_magnetic_vector, csf_signature
from atomx.config.csf import enumerate_jj_orbital_space
from atomx.physics.csf.levels import CILevel
from atomx.physics.csf.overlap import ci_level_overlaps


def rotated_case(number, keys=((1,-1), (2,-1), (3,-1))):
    csfs = enumerate_jj_orbital_space(keys,number,magnetic_state_cap=60)
    # Independent full exterior power, deliberately not grouped by (kappa,m).
    rng = np.random.default_rng(81)
    u = np.zeros((len(keys),len(keys)))
    for kappa in sorted({k for n,k in keys}):
        indices = [i for i,(_,k) in enumerate(keys) if k==kappa]
        rotation,_ = np.linalg.qr(rng.normal(size=(len(indices),len(indices))))
        u[np.ix_(indices,indices)] = rotation
    single = np.kron(u,np.eye(2))
    dets = list(combinations(range(2*len(keys)),number))
    wedge = np.array([[np.linalg.det(single[np.ix_(a,b)]) for b in dets] for a in dets])
    np.testing.assert_allclose(wedge @ wedge.T,np.eye(len(dets)),atol=1e-14)
    masks = [sum(1 << i for i in d) for d in dets]
    mix = np.zeros((len(csfs),len(csfs)))
    sectors = {}
    for i,c in enumerate(csfs):
        sectors.setdefault((c.total_J2,c.parity),[]).append(i)
    selectors = [None]*len(csfs)
    for (j2,parity),indices in sectors.items():
        columns = []
        for root,i in enumerate(indices):
            selectors[i] = CILevel(j2,parity,root)
            vec = dict(csf_magnetic_vector(csf_signature(csfs[i]),j2,keys))
            columns.append([vec.get(mask,0.) for mask in masks])
        c = np.array(columns).T
        transform = c.T @ wedge @ c
        np.testing.assert_allclose(transform @ transform.T,np.eye(len(indices)),atol=2e-14)
        mix[np.ix_(indices,indices)] = transform.T
    bra = SimpleNamespace(state_energies=np.arange(len(csfs),dtype=float),mixing_coefficients=np.eye(len(csfs)))
    ket = SimpleNamespace(state_energies=bra.state_energies.copy(),mixing_coefficients=mix)
    old = {key:(np.eye(len(keys))[:,i],np.zeros(len(keys))) for i,key in enumerate(keys)}
    new = {key:(u[:,i],np.zeros(len(keys))) for i,key in enumerate(keys)}
    return csfs,bra,ket,old,new,selectors,{k:np.eye(2*len(keys)) for n,k in keys}


@pytest.mark.parametrize('number',[1,2,3])
def test_same_physical_states_overlap_identically_in_rotated_radial_bases(number):
    csfs,bra,ket,old,new,selectors,metric = rotated_case(number)
    for i,s in enumerate(selectors):
        actual = ci_level_overlaps(csfs,bra,old,s,csfs,ket,new,(x for x in selectors),metric)
        np.testing.assert_allclose(actual,np.eye(len(csfs))[i],atol=3e-14)


def test_mixed_s_p_fermion_order_and_parity_follow_independent_exterior_power():
    csfs,bra,ket,old,new,selectors,metric = rotated_case(3, ((1,-1),(2,-1),(2,1),(3,1)))
    for i,s in enumerate(selectors):
        actual = ci_level_overlaps(csfs,bra,old,s,csfs,ket,new,selectors,metric)
        np.testing.assert_allclose(actual,np.eye(len(csfs))[i],atol=3e-14)


def test_zero_cross_overlap_is_valid_and_phase_is_signed():
    csfs = enumerate_jj_orbital_space(((1,-1),),1)
    ci = SimpleNamespace(state_energies=np.array([-1.]),mixing_coefficients=np.ones((1,1)))
    bra = {(1,-1):(np.array([1.,0.]),np.zeros(2))}
    ket = {(1,-1):(np.array([0.,1.]),np.zeros(2))}
    level = CILevel(1,1,0)
    np.testing.assert_array_equal(ci_level_overlaps(csfs,ci,bra,level,csfs,ci,ket,(level,),{-1:np.eye(4)}),[0.])
    flipped = SimpleNamespace(state_energies=ci.state_energies,mixing_coefficients=-ci.mixing_coefficients)
    np.testing.assert_allclose(ci_level_overlaps(csfs,ci,bra,level,csfs,flipped,bra,(level,),{-1:np.eye(4)}),[-1.],atol=1e-14,rtol=0)


def test_nonunique_roots_and_pair_budget_refuse_instead_of_truncating():
    csfs,bra,ket,old,new,selectors,metric = rotated_case(3)
    with pytest.raises(ValueError,match='determinant pairs'):
        ci_level_overlaps(csfs,bra,old,selectors[0],csfs,ket,new,selectors,metric,max_determinant_pairs=1)
    csfs,bra,ket,old,new,selectors,metric = rotated_case(1)
    bra.state_energies[:] = 0
    with pytest.raises(ValueError,match='degenerate'):
        ci_level_overlaps(csfs,bra,old,selectors[0],csfs,ket,new,selectors,metric)


@pytest.mark.parametrize('defect', ['asymmetric', 'complex'])
def test_nonphysical_radial_metric_refuses(defect):
    csfs,bra,ket,old,new,selectors,metric = rotated_case(1)
    if defect == 'asymmetric':
        metric[-1][0,3] = .2
    else:
        metric[-1] = metric[-1].astype(complex)
        metric[-1][0,3] = .2j
    with pytest.raises(ValueError,match='real symmetric'):
        ci_level_overlaps(csfs,bra,old,selectors[0],csfs,ket,new,selectors,metric)
