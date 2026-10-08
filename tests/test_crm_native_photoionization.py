"""CRM native PI/Milne integration and rejection of silent Kramers mixing."""
import numpy as np
import pytest

from atomx.plasma.crm import (
    CollisionalRadiativeModel,
    CRMBoundFreeChannel,
    CRMLevel,
    _channel_photoionization_rate_s,
    radiative_recombination_rate_cm3_s,
    saha_ionization_density_cm3,
)


def test_native_cross_section_has_planck_milne_detailed_balance():
    I,T=15.,20.
    # Arbitrary non-Kramers shape. Exact same provider in both integrals.
    def sigma(E):
        return 2e-18*(I/E)**2.4*(1+.3*np.exp(-((E-2*I)/3)**2))
    ch=CRMBoundFreeChannel(0,1,I,photoionization_sigma_cm2=sigma)
    pi=_channel_photoionization_rate_s(ch,radiation_temperature_eV=T,radiation_field=None)
    rr=radiative_recombination_rate_cm3_s(I,T,g_bound=2,g_continuum=1,
                                         cross_section_cm2=sigma,radiation_temperature_eV=T)
    saha=saha_ionization_density_cm3(I,T,g_bound=2,g_continuum=1)
    assert pi/rr==pytest.approx(saha,rel=2e-8)


def test_real_sigma_callback_changes_crm_pumping():
    crm=CollisionalRadiativeModel([CRMLevel('a',0),CRMLevel('b',20)],[],
                                 bound_free_channels=[CRMBoundFreeChannel(0,1,20,
                                 photoionization_sigma_cm2=lambda E:3e-18)])
    # Normalized exponential photon spectrum above threshold.
    matrix=crm._build_rate_matrix(0,10,include_eii=False,include_radiative_recombination=False,
                                  radiation_field=lambda E:1e25/10*np.exp(-(E-20)/10) if E>=20 else 0.)
    assert matrix[1,0]==pytest.approx(3e7,rel=1e-8)
    assert matrix.sum(axis=0)==pytest.approx([0,0],abs=1e-8)


def test_missing_provider_and_silent_model_mixing_raise():
    ch=CRMBoundFreeChannel(0,1,20)
    with pytest.raises(ValueError,match='real PI cross-section'):
        _channel_photoionization_rate_s(ch,radiation_temperature_eV=10,radiation_field=None)
    with pytest.raises(ValueError,match='explicit Kramers'):
        CRMBoundFreeChannel(0,1,20,sigma0_cm2=1e-18)
    with pytest.raises(ValueError,match='must not coexist'):
        CRMBoundFreeChannel(0,1,20,bound_free_model='kramers',photoionization_sigma_cm2=lambda E:1e-18)


def test_negative_native_sigma_is_not_clipped():
    ch=CRMBoundFreeChannel(0,1,20,photoionization_sigma_cm2=lambda E:-1)
    with pytest.raises(ValueError,match='nonnegative'):
        _channel_photoionization_rate_s(ch,radiation_temperature_eV=10,radiation_field=None)
