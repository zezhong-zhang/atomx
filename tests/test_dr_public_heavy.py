"""Heavy-ion DR: native public API, energy-sized continuum grid, external FAC."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from atomx import Atom, Basis, Sampling
from atomx.scattering.autoionization import (
    dr_calculate,
    dr_rate_coefficient,
    mscheme_ci,
)
from atomx.scattering.autoionization.auger import AutoionizationState


@pytest.mark.external_fac
def test_pb_public_dr_covers_continuum_energies_and_matches_fac(monkeypatch):
    data = json.loads(
        (Path(__file__).parent / "data/fac_pb_dr33_matched.json").read_text()
    )
    atom = Atom(
        82,
        charge=72,
        config="1s2 2s2 2p6",
        engine="shooting",
        exchange="gu",
        nuclear="point",
        corrections="none",
        basis=Basis(sampling=Sampling(n_grid=4000)),
    )
    atom.solve(tol=1e-10, max_iter=150)
    assert atom.converged
    factory = mscheme_ci.manifold_from_configs
    grids = {}

    def measured(result, configs, **kwargs):
        manifold, workspace = factory(result, configs, **kwargs)
        group = next(k for k, v in data["configs"].items() if v == configs)
        grids[group] = kwargs["e_max_eV"]
        return manifold, workspace

    monkeypatch.setattr(mscheme_ci, "manifold_from_configs", measured)
    temperatures = np.array([1, 5, 10, 50]) * 1e6 * 8.617333262e-5 * (82 / 26) ** 2
    result = dr_calculate(atom, "1s2 2s2 2p6 3s1", T_eV=temperatures)
    reference = data["resonances"]
    assert len(result.resonances) == len(reference) == 237
    assert Counter(
        (r.label.split(":")[0], int(2 * r.J)) for r in result.resonances
    ) == Counter((r["group"], r["two_J"]) for r in reference)
    for resonance in result.resonances:
        assert grids[resonance.label.split(":")[0]] >= resonance.E_resonance_eV
    assert max(r.E_resonance_eV for r in result.resonances) > 5000.0
    native = sorted(
        result.resonances, key=lambda r: (r.label.split(":")[0], r.J, r.E_resonance_eV)
    )
    fac = sorted(reference, key=lambda r: (r["group"], r["two_J"] / 2, r["energy_eV"]))
    for a, b in zip(native, fac, strict=True):
        assert abs(a.E_resonance_eV - b["energy_eV"]) < 0.4
    states = [
        AutoionizationState(
            str(r["ilev"]),
            r["energy_eV"],
            r["two_J"] / 2,
            r["auger_per_s"],
            r["radiative_per_s"],
        )
        for r in reference
    ]
    external = dr_rate_coefficient(states, temperatures, initial_state_g=1.0)
    # Aggregate comparison, not a gate on each interference-sensitive width.
    np.testing.assert_allclose(
        result.alpha_cm3_s, external.alpha_cm3_s, rtol=0.01, atol=0
    )
