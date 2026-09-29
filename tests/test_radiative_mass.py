import math
from types import SimpleNamespace
import pytest
from atomx.constants import ALPHA
from atomx.config.subshell import Subshell
from atomx.corrections.radiative_mass import hydrogenic_radiative_mass_leading as shifts
from atomx.corrections.adapters import HydrogenicRadiativeMassLeadingCorrection


def test_large_mass_limit_from_codatas_leading_coefficients():
    mass=1e20
    factor=ALPHA**3/math.pi
    # CODATA 2018 Table II, 1s Bethe logarithm; explicit Eq. (15) coefficients.
    f=4/3*math.log(ALPHA**-2)+10/9-4/3*2.98412856
    result=shifts(1,1,-1,mass)
    assert result['self_energy_mass_ha']*mass/factor==pytest.approx(-3*f+4/3,rel=1e-14)
    assert result['vacuum_polarization_mass_ha']*mass/factor==pytest.approx(4/5,rel=1e-14)


def test_magnetic_term_has_mu_squared_not_mu_cubed():
    mass=1836.152673426;mu=mass/(mass+1)
    p12=shifts(1,2,1,mass);p32=shifts(1,2,-2,mass)
    difference=p32['self_energy_mass_ha']-p12['self_energy_mass_ha']
    assert difference==pytest.approx(ALPHA**3/(math.pi*8)*(mu**2-1)/4,rel=2e-12)
    assert p12['vacuum_polarization_mass_ha']==p32['vacuum_polarization_mass_ha']==0.


def test_small_positive_mass_stays_finite():
    assert all(math.isfinite(value) for value in shifts(1,1,-1,1e-320).values())


def test_adapter_has_separate_component_ledger_and_rejects_many_electrons():
    adapter=HydrogenicRadiativeMassLeadingCorrection(1836.)
    result=SimpleNamespace(Z=1,subshells=[Subshell(2,-1,1)])
    correction=adapter.evaluate(result)
    assert correction.energy_shift==sum(correction.provenance['components'].values())
    assert correction.per_orbital['2s1/2']==correction.energy_shift
    result.subshells=[Subshell(1,-1,2)]
    assert not adapter.supports(result)
    with pytest.raises(ValueError,match='one-electron'):adapter.evaluate(result)


def test_additive_stack_updates_total_and_qed_ledger_once():
    from atomx.corrections import CorrectionStack, finalize_with_corrections
    result=SimpleNamespace(Z=1,subshells=[Subshell(2,-1,1)],
                           total_energy=-.125,energy_components={})
    adapter=HydrogenicRadiativeMassLeadingCorrection(1836.)
    expected=adapter.evaluate(result).energy_shift
    stack=CorrectionStack(se_backend=None,include_vp=False,include_breit=False,strict=True).add(adapter)
    finalize_with_corrections(result,stack)
    assert result.total_energy==-.125+expected
    assert result.energy_components['E_QED_total']==expected
    assert result.energy_components[adapter.name]==expected
    with pytest.raises(ValueError,match='already finalized'):
        finalize_with_corrections(result,stack)


@pytest.mark.parametrize('args',[(1,2,0,1836.),(1,1,1,1836.),(1,201,-1,1836.),(1,1,-1,0.),(1,1,-1,float('inf'))])
def test_invalid_or_unavailable_inputs_fail_explicitly(args):
    with pytest.raises(ValueError):shifts(*args)
