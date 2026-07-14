"""Ab-initio DR end-to-end from determinant-CI manifolds vs FAC (Fe XVII).

Builds every 2p- and 2s-hole dr33 resonance manifold (2s2 2p5 3l3l' and
2s1 2p6 3l3l') with :class:`ResonanceManifold.dr_levels` — ab-initio
resonance energies, Auger widths, and E1 stabilizing widths — and runs
the Burgess kernel.  The reference is FAC's AITable+TRTable restricted to
the same dr33 configuration set (tests/data/fe17_dr_fac.json).

History: the old ``dr_calculate`` end-to-end path overshot FAC by
~10^3-10^4 (hydrogenic width placeholders).  This determinant-CI path
lands at 1.0-2.0x across T = 1-50 MK.  Known residuals (documented in
docs/scattering_fac_verification_2026-07.md):
- resonance energies ~60 eV high (frozen-orbital, no relaxation), which
  also inflates omega^3 in the radiative widths;
- V^{N-1} virtuals more compact than FAC's optimized DFS potential;
- ~25% of the high-T excess is real physics: levels below FAC's
  SetAICut(1e-4) that the FAC reference drops (atomx keeps all 237
  dr33 resonances vs FAC's listed 75).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx import Atom  # noqa: E402
from atomx.constants import HARTREE_EV  # noqa: E402

FIXTURE = Path(__file__).parent / "data" / "fe17_dr_fac.json"
K_B = 8.617333262e-5
T_MK = (1.0, 5.0, 10.0, 50.0)


@pytest.fixture(scope="module")
def dr_states():
    from atomx.methods.mbpt import _build_virtual_basis
    from atomx.numerics.bspline.evaluate import eval_P, eval_Q
    from atomx.scattering.autoionization.auger import AutoionizationState
    from atomx.scattering.autoionization.mscheme_ci import ResonanceManifold
    from atomx.scattering.distorted_wave.solver import _make_fine_grid, _p_rel

    g = Atom(Z=26, charge=16, configs=["1s2 2s2 2p6"], breit=False, qed=False)
    g.solve()
    res = g._result
    bsp = getattr(res, "_bsp_data", None) or res.bsp_data
    r_fine = _make_fine_grid(bsp, _p_rel(500.0 / HARTREE_EV))
    dr_fine = np.gradient(r_fine)
    gr = bsp["gr"].ravel()
    order = np.argsort(gr)
    grs = gr[order]

    def tofine(P, Q):
        Pg = eval_P(P, bsp).ravel()[order]
        Qg = eval_Q(Q, bsp).ravel()[order]
        return np.interp(r_fine, grs, Pg), np.interp(r_fine, grs, Qg)

    orbital_map = {}
    orbital_eps = {}
    for kappa, lab, idx in ((-1, "2s1/2", 1), (1, "2p1/2", 0), (-2, "2p3/2", 0)):
        P, Q, _ = res.orbitals_raw[kappa][idx]
        orbital_map[(2, kappa)] = tofine(P, Q)
        orbital_eps[(2, kappa)] = float(res.orbital_energies[lab])
    for kappa in (-1, 1, -2, 2, -3):
        _occ, virt = _build_virtual_basis(res, kappa)
        P, Q, e = virt[0]
        orbital_map[(3, kappa)] = tofine(P, Q)
        orbital_eps[(3, kappa)] = float(e)

    G3 = [([(3, -1)], 0), ([(3, 1), (3, -2)], 0), ([(3, 2), (3, -3)], 0)]

    def make(hole_subs, hole_ne, confs):
        return ResonanceManifold(
            res, [(hole_subs, hole_ne)] + G3,
            r_fine=r_fine, dr_fine=dr_fine, orbital_map=orbital_map,
            orbital_energies=orbital_eps, configs=confs)

    manifolds = [
        make([(2, 1), (2, -2)], 5, [(5, 1, 1, 0), (5, 0, 1, 1)]),
        make([(2, 1), (2, -2)], 5,
             [(5, 1, 0, 1), (5, 0, 2, 0), (5, 0, 0, 2), (5, 2, 0, 0)]),
        make([(2, -1)], 1,
             [(1, 2, 0, 0), (1, 1, 0, 1), (1, 0, 2, 0), (1, 0, 0, 2)]),
        make([(2, -1)], 1, [(1, 1, 1, 0), (1, 0, 1, 1)]),
    ]
    states = []
    for M in manifolds:
        e_ion = M.ion_energy()
        for tM in range(1, 13, 2):
            for L in M.dr_levels(tM, Z=26, bsp_data=bsp, e_ion=e_ion):
                states.append(AutoionizationState(
                    label=L.dominant[:48],
                    E_resonance_eV=L.E_res_au * HARTREE_EV,
                    J=tM / 2.0,
                    gamma_auger_per_s=L.gamma_auger_per_s,
                    a_radiative_per_s=L.a_radiative_per_s))
    return states


@pytest.mark.fac_parity
def test_resonance_count_and_energy_window(dr_states):
    # dr33 supports ~237 resonances; energies in the physical window.
    assert 200 <= len(dr_states) <= 260
    Es = [s.E_resonance_eV for s in dr_states]
    assert 200.0 <= min(Es) <= 320.0
    assert 450.0 <= max(Es) <= 750.0


@pytest.mark.fac_parity
def test_alpha_dr_vs_fac_dr33(dr_states):
    """Burgess alpha_DR(T) within [0.5, 2.5] of the FAC dr33 subset.

    (The old dr_calculate end-to-end was ~10^3-10^4 off; see module
    docstring for the sources of the remaining <=2x.)"""
    from atomx.scattering.autoionization import dr_rate_coefficient
    from atomx.scattering.autoionization.auger import AutoionizationState

    fac = json.loads(FIXTURE.read_text())
    fac33 = [r for r in fac["resonances"] if "2*7.3*2 " in r["label"]]
    assert len(fac33) > 50
    fac_states = [AutoionizationState(
        label="", E_resonance_eV=r["E_resonance_eV"], J=r["J"],
        gamma_auger_per_s=r["gamma_auger_per_s"],
        a_radiative_per_s=r["a_radiative_per_s"]) for r in fac33]

    T_eV = [T * 1.0e6 * K_B for T in T_MK]
    rc = dr_rate_coefficient(dr_states, T_eV, initial_state_g=1.0)
    rcf = dr_rate_coefficient(fac_states, T_eV, initial_state_g=1.0)
    for i, T in enumerate(T_MK):
        ratio = float(rc.alpha_cm3_s[i]) / float(rcf.alpha_cm3_s[i])
        assert 0.5 <= ratio <= 2.5, f"T={T} MK: alpha ratio {ratio:.2f}"


@pytest.mark.fac_parity
def test_widths_magnitudes(dr_states):
    strongest = max(dr_states, key=lambda s: s.gamma_auger_per_s)
    # Strong KLL-type Auger widths are a few 1e14 s^-1 in FAC.
    assert 1.0e14 <= strongest.gamma_auger_per_s <= 1.0e15
    ar = [s.a_radiative_per_s for s in dr_states if s.a_radiative_per_s > 0]
    # Strong stabilizing widths are ~1e12-1e13 s^-1.
    assert 5.0e11 <= max(ar) <= 5.0e13
