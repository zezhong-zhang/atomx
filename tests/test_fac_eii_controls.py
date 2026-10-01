"""SFAC command order and reference-grid controls must stay explicit."""
import pytest

from benchmarks.bench_fac_eii_integration import refine_deck

DECK = """SetCIPWOptions(1e-3,20,8,36,18)
SetCIEGrid([1.,2.,3.])
SetUsrCIEGrid([1.,2.,3.])
CITable('ci.b',['g'],['ion'])
"""


def test_explicit_full_dirac_and_dense_controls_preserve_other_cutoffs():
    result = refine_deck(DECK, full_relativistic=True, dense_partial_waves=True,
                         energy_index=2)
    assert 'SetCIPWOptions(0.001,20,8,-1,18)' in result
    assert 'SetCIPWGrid([20],[1])' in result
    assert 'SetCIEGrid([1.,2.,3.])' in result
    assert 'SetUsrCIEGrid([3.0])' in result
    assert refine_deck(DECK) == DECK


@pytest.mark.parametrize('index', [-1, 3])
def test_energy_selection_cannot_silently_change_the_reference_grid(index):
    with pytest.raises(ValueError, match='energy index'):
        refine_deck(DECK, energy_index=index)


def test_ambiguous_or_implicit_partial_wave_options_are_rejected():
    with pytest.raises(ValueError, match='one explicit'):
        refine_deck(DECK+DECK, full_relativistic=True)
    with pytest.raises(ValueError, match='must specify'):
        refine_deck(DECK.replace('1e-3,20,8,36,18','1e-3,20'), full_relativistic=True)
