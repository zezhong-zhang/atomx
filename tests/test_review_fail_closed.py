"""Failure injection at the physical/provenance seams identified in review E7."""

from types import SimpleNamespace

import pytest


def test_failed_dirac_reference_is_never_substituted_or_cached(monkeypatch):
    import atomx.methods.hydrogenic_reference as reference
    from atomx.corrections.qedmod.shabaev import _hydrogenic_r_exp_integral_dirac_cached

    calls = []

    def fail(*args):
        calls.append(args)
        raise ValueError("injected Dirac failure")

    monkeypatch.setattr(reference, "hydrogenic_r_exp_integral_dirac", fail)
    _hydrogenic_r_exp_integral_dirac_cached.cache_clear()
    try:
        for _ in range(2):
            with pytest.raises(RuntimeError, match=r"n=1, kappa=-1, Z=10") as raised:
                _hydrogenic_r_exp_integral_dirac_cached(1, -1, 10)
            assert isinstance(raised.value.__cause__, ValueError)
        assert len(calls) == 2
        assert _hydrogenic_r_exp_integral_dirac_cached.cache_info().currsize == 0
    finally:
        _hydrogenic_r_exp_integral_dirac_cached.cache_clear()


def test_bad_mohr_interpolation_propagates(monkeypatch):
    import scipy.interpolate

    from atomx.corrections.qedmod.mohr import _interp_table

    def fail(*args):
        raise ValueError("injected PCHIP failure")

    monkeypatch.setattr(scipy.interpolate, "PchipInterpolator", fail)
    with pytest.raises(ValueError, match="injected PCHIP failure"):
        _interp_table([1.0, 3.0], [0.1, 0.2], 2.0)


def test_unspecified_nuclear_model_cannot_be_recorded():
    from atomx.calculation_spec import resolved_solver_spec

    with pytest.raises(ValueError, match="nuclear-model provenance"):
        resolved_solver_spec(SimpleNamespace(Z=10), method="DHF", engine="bspline", controls={})



@pytest.mark.parametrize("missing", ["a_radiative_per_s", "g"])
def test_crm_rejects_missing_physics_instead_of_guessing(missing):
    from atomx.plasma.crm import CollisionalRadiativeModel, CRMLevel

    crm = CollisionalRadiativeModel([CRMLevel("ground", 0.0, g=1)], [])
    fields = dict(a_radiative_per_s=1.0, g=3, E_resonance_eV=10.0,
                  gamma_auger_per_s=2.0)
    del fields[missing]
    with pytest.raises(AttributeError, match=missing):
        crm.add_dr_satellites([SimpleNamespace(**fields)], initial_level_idx=0)
