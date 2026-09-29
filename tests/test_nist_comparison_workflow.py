"""A stable SCF energy must not conceal the wrong configuration or isotope."""
from copy import deepcopy
import pytest

from atomx.workflows.nist_comparison import completed_pair, interval, orbital_identity_issue
from atomx.workflows.nist_comparison import apply_recovered_pairs, assess_negative_configuration_interval


def row(grid=3200):
    return dict(status='done',grid=grid,configuration='1s1 2s2',isotope_a=9,
        energy_ha=-10.,orbital_energies_ha={'1s1/2':-6.,'2s1/2':-.4},
        specification=dict(Z=4,physics={'nuclear_atomic_weight':9},
            corrections={'nuclear_mass_me':16424.},state={'subshells':[[1,-1,1.],[2,-1,2.]]}))


def test_converged_crossed_orbitals_are_rejected():
    r=row()
    assert orbital_identity_issue([r]) is None
    r['orbital_energies_ha']={'1s1/2':-.4,'2s1/2':-6.}
    assert orbital_identity_issue([r])=='same_kappa_radial_labels_crossed'


def test_refinement_requires_both_successes_and_matched_mass():
    job=dict(z=4,configuration='1s1 2s2',isotope=9,mass=16424.,grids=[3200,6400])
    attempts={3200:row(),6400:row(6400)}
    assert completed_pair(job,attempts)
    attempts[6400]['status']='failed'
    assert completed_pair(job,attempts) is None
    attempts[6400]=row(6400)
    attempts[6400]['specification']['corrections']['nuclear_mass_me']=16000.
    with pytest.raises(ValueError,match='identity'):
        completed_pair(job,attempts)


def test_transition_endpoints_can_use_distinct_grids_but_not_distinct_physics():
    a=[row(),row(6400)]
    b=deepcopy(a)
    b[0].update(grid=16000,energy_ha=-9.)
    b[1].update(grid=32000,energy_ha=-9.)
    result=interval(a,b)
    assert result['grid_change_cm1']==0.
    assert result['upper_grids']==[16000,32000]
    b[1]['specification']['physics']['nuclear_atomic_weight']=10
    with pytest.raises(ValueError,match='Hamiltonians differ'):
        interval(a,b)


def test_recovery_uses_whole_pairs_and_preserves_priority_and_base_results():
    job=dict(z=4,charge=1,configuration='1s1 2s2',isotope=9,mass=16424.,
             grids=[3200,6400],is_ground=False)
    key=(4,1,'1s1 2s2')
    first=dict(job=job,pair=[row(),row(6400)],source_id='first')
    second=deepcopy(first);second['source_id']='second'
    second['pair'][1]['energy_ha']=-9.
    complete={};rejected={key:'old failed identity'}
    selected=apply_recovered_pairs({1:job},complete,rejected,[first,second])
    assert selected=={key:'first'}
    assert complete[key][1]['energy_ha']==-10.
    assert rejected=={}
    assert apply_recovered_pairs({1:job},complete,rejected,[second])=={}
    assert complete[key][1]['energy_ha']==-10.
    broken=deepcopy(first);broken['pair']=broken['pair'][:1]
    with pytest.raises(ValueError,match='incomplete'):
        apply_recovered_pairs({1:job},{},{},[broken])
    broken=deepcopy(first);broken['job']['mass']=17000.
    with pytest.raises(ValueError,match='mass'):
        apply_recovered_pairs({1:job},{},{},[broken])
    with pytest.raises(ValueError,match='outside'):
        apply_recovered_pairs({},{},{},[first])




def test_negative_average_is_compared_to_complete_manifolds():
    lower=dict(status='evaluated_complete',centroid_cm1=12000.)
    upper=dict(status='evaluated_complete',centroid_cm1=10000.)
    result=assess_negative_configuration_interval(lower,upper)
    assert result['ordering_assessment']=='same_negative_nist_centroid_order'
    assert result['nist_configuration_interval_cm1']==-2000.
    assert assess_negative_configuration_interval(upper,lower)['ordering_assessment']=='opposite_nist_centroid_order'
    assert assess_negative_configuration_interval(lower,lower)['ordering_assessment']=='degenerate_nist_centroids'
    upper['status']='incomplete'
    assert assess_negative_configuration_interval(lower,upper)==dict(ordering_assessment='incomplete_nist_manifold')


def test_correction_model_revision_cannot_be_mixed_between_endpoints_or_grids():
    a=[row(),row(6400)];b=deepcopy(a)
    for r in a:r['correction_provenance']={'self_energy[mohr]':{'method':'mohr'}}
    for r in b:r['correction_provenance']={'self_energy[mohr]':{'method':'mohr','screening_reference':'exact_point_dirac_coulomb_expectation_v1'}}
    with pytest.raises(ValueError,match='screening references differ'):interval(a,b)
    job=dict(z=4,configuration='1s1 2s2',isotope=9,mass=16424.,grids=[3200,6400])
    with pytest.raises(ValueError,match='screening reference'):
        completed_pair(job,{3200:a[0],6400:b[1]})
