import pytest

from atomx.config.jj_coupling import resolve_one_electron_subshell


@pytest.mark.parametrize('config,two_j,kappa',[
    ('1s1',1,-1),('2p1',1,1),('2p1',3,-2),('3d1',3,2),('3d1',5,-3)])
def test_exact_one_electron_angular_identity(config,two_j,kappa):
    shell=resolve_one_electron_subshell(config,two_j)
    assert shell.kappa==kappa
    assert shell.occupation==1


@pytest.mark.parametrize('config,two_j',[
    ('1s2',1),('1s1 2p1',1),('2p1',5),('1s1',3),('2p1',True),('2p1',1.5)])
def test_invalid_or_ambiguous_state_is_rejected(config,two_j):
    with pytest.raises(ValueError):resolve_one_electron_subshell(config,two_j)
