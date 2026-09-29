"""NIST k-shell configurations must reach the numerical Dirac solver intact."""
import pytest

from atomx import Atom, Basis
from atomx.config.jj_coupling import generate_jj_average
from atomx.config.parsing import parse_nr_shells
from atomx.methods.dirac_fock import DiracFock
from atomx.physics.dirac import hydrogenic_dirac_energy
from atomx.types import NuclearModel
from atomx.reference_data.matching import expand_nist_config


def test_k_configuration_expansion_and_statistical_jj_weights():
    assert parse_nr_shells('8k1') == [(8,7,1)]
    assert expand_nist_config('2s2 8k1',5) == '1s2 2s2 8k1'
    shells,_=generate_jj_average('8k1',averaging='statistical')
    assert [(s.n,s.kappa) for s in shells]==[(8,7),(8,-8)]
    assert [s.occupation for s in shells]==pytest.approx([14/30,16/30])


def test_k_hydrogenic_dhf_matches_dirac_configuration_average():
    result=Atom(Z=10,config='8k1',engine='bspline',basis=Basis(r_max=75,grid_hmax=1)).solve(
        DiracFock(nuclear=NuclearModel.POINT),max_iter=40)
    expected=(14*hydrogenic_dirac_energy(8,7,10)+16*hydrogenic_dirac_energy(8,-8,10))/30
    assert result.converged
    assert result.total_energy==pytest.approx(expected,abs=1e-9)
    assert set(result.orbital_energies)=={'8k13/2','8k15/2'}


def test_nist_parentage_separates_orbitals_and_repeated_shells_add():
    assert expand_nist_config('3s2.3p(2P*)4p',14) == '1s2 2s2 2p6 3s2 3p1 4p1'
    assert expand_nist_config('2p5.3d.(1P*<1>).3d',9) == '1s2 2p5 3d2'
    assert expand_nist_config('1s2.9l',3) == '1s2 9l1'
    assert expand_nist_config('1s2.2s.12[k-o]',4) is None
    assert expand_nist_config('3s2.nd',13) is None


def test_complete_bethe_log_reference_includes_l_states():
    from atomx.data.mohr_se import klamaq_f
    from atomx.data.nist_bethe_log import BETHE_LOG_NIST

    assert set(BETHE_LOG_NIST)==set(range(1,201))
    assert all(len(row)==n for n,row in BETHE_LOG_NIST.items())
    assert sum(map(len,BETHE_LOG_NIST.values()))==20100
    assert BETHE_LOG_NIST[9][8]==-0.000104148093
    expected=(4/3)*(0.000104148093-3/(8*8*17))
    assert klamaq_f(9,8)==pytest.approx(expected,abs=1e-15)
def test_generic_l_placeholder_is_not_a_physical_orbital():
    import pytest
    from atomx.config.parsing import parse_nr_shells
    from atomx.reference_data.matching import expand_nist_config

    with pytest.raises(ValueError, match='l < n'):
        parse_nr_shells('4l1')
    assert expand_nist_config('3d9.4l', 28) is None
    assert parse_nr_shells('9l1') == [(9, 8, 1)]


def test_q_through_z_notation_and_full_principal_number():
    for n,letter,l in ((13,'q',12),(14,'r',13),(15,'t',14),(20,'y',19),(21,'z',20)):
        config=f'{n}{letter}1'
        assert parse_nr_shells(config)==[(n,l,1)]
        assert expand_nist_config(f'{n}{letter}',1)==config
        shells,_=generate_jj_average(config,averaging='statistical')
        assert [s.kappa for s in shells]==[l,-l-1]
        assert [s.occupation for s in shells]==pytest.approx([2*l/(4*l+2),(2*l+2)/(4*l+2)])
        assert all(f'{n}{letter}' in s.label for s in shells)
    assert expand_nist_config('100s',1)=='100s1'
    with pytest.raises(ValueError,match='l < n'):
        parse_nr_shells('12q1')


@pytest.mark.parametrize('n,l,letter',[(13,12,'q'),(20,19,'y')])
def test_high_l_point_dhf_matches_weighted_dirac(n,l,letter):
    result=Atom(Z=10,config=f'{n}{letter}1',engine='bspline',
        basis=Basis(r_max=250,grid_hmax=1)).solve(DiracFock(nuclear=NuclearModel.POINT),max_iter=40)
    expected=(2*l*hydrogenic_dirac_energy(n,l,10)+(2*l+2)*hydrogenic_dirac_energy(n,-l-1,10))/(4*l+2)
    assert result.converged
    assert result.total_energy==pytest.approx(expected,abs=1e-9)
