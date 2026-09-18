"""Continuum channels in spectral synthesis (review M3 fix).

Free-free: Rybicki & Lightman eq. 5.14b with the exact Maxwell-averaged
Born Gaunt factor (sqrt(3)/pi) e^(x/2) K_0(x/2); constant re-derived from
CODATA.  Free-bound: Milne-relation RR continuum folded through
crm_emission_spectrum.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx.spectrum.continuum import free_free_emissivity, gaunt_ff_born  # noqa: E402


def test_gaunt_born_limits():
    # x -> 0: (sqrt(3)/pi) ln(4/(gamma_E' x)) with gamma_E' = e^gamma =
    # 1.78107; x -> inf: sqrt(3/(pi x)).
    x = 1.0e-6
    expect = np.sqrt(3.0) / np.pi * np.log(4.0 / (1.78107 * x))
    assert gaunt_ff_born(x) == pytest.approx(expect, rel=1e-4)
    x = 400.0
    assert gaunt_ff_born(x) == pytest.approx(np.sqrt(3.0 / (np.pi * x)),
                                             rel=2e-3)


def test_free_free_textbook_magnitude():
    # Rybicki & Lightman: eps_nu = 6.8e-38 Z^2 n_e n_i T^-1/2 e^-x g
    # erg s^-1 cm^-3 Hz^-1.  Pure-H plasma, T = 1 keV, h nu = 1 keV:
    T_eV = 1000.0
    E = np.array([1000.0])
    eps = free_free_emissivity(E, T_eV, 1.0, n_e_cm3=1.0e14,
                               n_i_cm3=1.0e14)
    # hand evaluation of the same formula
    T_K = T_eV / 8.617333262e-5
    g = gaunt_ff_born(1.0)
    ref_erg_hz = 6.8419e-38 * 1e28 / np.sqrt(T_K) * np.exp(-1.0) * g
    ref_ev_ev = ref_erg_hz / 4.135667696e-15 / 1.602176634e-12
    assert eps[0] == pytest.approx(ref_ev_ev, rel=1e-6)
    # exponential slope: eps(2 keV)/eps(1 keV) ~ e^-1 * g-ratio
    eps2 = free_free_emissivity(np.array([2000.0]), T_eV, 1.0,
                                n_e_cm3=1e14, n_i_cm3=1e14)
    ratio = eps2[0] / eps[0]
    assert ratio == pytest.approx(np.exp(-1.0) * gaunt_ff_born(2.0)
                                  / gaunt_ff_born(1.0), rel=1e-9)


def test_crm_synthesis_includes_free_free():
    from atomx.plasma.crm import CollisionalRadiativeModel, CRMLevel, CRMTransition
    from atomx.spectrum.crm_synth import crm_emission_spectrum

    levels = [CRMLevel("g", 0.0, g=1), CRMLevel("e", 100.0, g=3)]
    trs = [CRMTransition(upper=1, lower=0, A_rate=1e10, gf=0.1)]
    crm = CollisionalRadiativeModel(levels, trs)

    kw = dict(n_e_cm3=1e14, T_e_eV=500.0, lam_min_nm=5.0, lam_max_nm=9.0,
              n_points=401, include_eii=False, include_three_body=False,
              include_photoionization=False)
    spec0 = crm_emission_spectrum(crm, **kw)
    spec1 = crm_emission_spectrum(crm, include_free_free=True,
                                  ff_Z_eff=10.0, **kw)
    # window excludes the 12.4-nm line: baseline is empty, ff fills it
    assert np.all(spec1.values >= spec0.values)
    assert float(np.min(spec1.values - spec0.values)) > 0.0
    assert spec1.metadata["free_free"]["Z_eff"] == 10.0
    # continuum rises toward short wavelength below the exponential
    # cutoff (E(9 nm) = 138 eV << kT): jacobian E/lam dominates
    v = spec1.values - spec0.values
    assert v[0] > v[-1]


def test_crm_synthesis_requires_zeff():
    from atomx.plasma.crm import CollisionalRadiativeModel, CRMLevel, CRMTransition
    from atomx.spectrum.crm_synth import crm_emission_spectrum
    crm = CollisionalRadiativeModel(
        [CRMLevel("g", 0.0, g=1), CRMLevel("e", 10.0, g=3)],
        [CRMTransition(upper=1, lower=0, A_rate=1e8, gf=0.1)])
    with pytest.raises(ValueError, match="ff_Z_eff"):
        crm_emission_spectrum(crm, n_e_cm3=1e12, T_e_eV=100.0,
                              lam_min_nm=1.0, lam_max_nm=2.0,
                              include_free_free=True,
                              include_eii=False,
                              include_three_body=False,
                              include_photoionization=False)


def test_crm_synthesis_includes_rr_continuum():
    """Free-bound edge appears in the synthesized spectrum (He+ 1s)."""
    from atomx import Atom
    from atomx.constants import HARTREE_EV
    from atomx.plasma.crm import CollisionalRadiativeModel, CRMLevel, CRMTransition
    from atomx.spectrum.crm_synth import crm_emission_spectrum

    atom = Atom(Z=2, config="1s1", numerics="bspline", breit=False, qed=False)
    atom.solve()
    binding = -atom._result.orbital_energies["1s1/2"] * HARTREE_EV  # ~54.4 eV

    crm = CollisionalRadiativeModel(
        [CRMLevel("g", 0.0, g=1), CRMLevel("e", 40.8, g=3)],
        [CRMTransition(upper=1, lower=0, A_rate=1e9, gf=0.1)])
    # window straddles the RR edge at lam_edge = 1239.84/54.4 ~ 22.8 nm
    lam_edge = 1239.8419843320026 / binding
    kw = dict(n_e_cm3=1e13, T_e_eV=20.0,
              lam_min_nm=lam_edge * 0.6, lam_max_nm=lam_edge * 1.4,
              n_points=301, include_eii=False, include_three_body=False,
              include_photoionization=False)
    spec0 = crm_emission_spectrum(crm, **kw)
    spec1 = crm_emission_spectrum(crm, include_rr_continuum=True,
                                  rr_scf_result=atom._result, **kw)
    extra = spec1.values - spec0.values
    lam = spec1.wavelength_nm
    # emission only above the edge energy (shortward of lam_edge)
    short = lam < lam_edge * 0.98
    longw = lam > lam_edge * 1.02
    assert float(np.max(extra[short])) > 0.0
    assert float(np.max(np.abs(extra[longw]))) == pytest.approx(0.0, abs=1e-30)
    assert spec1.metadata["rr_continuum"] is True
