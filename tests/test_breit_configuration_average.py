"""Configuration traces must agree with the resolved many-electron operator."""

import hashlib
from collections import defaultdict

import numpy as np
import pytest

from atomx.angular.csf_hamiltonian import csf_signature, exact_block_matrix
from atomx.config.csf import enumerate_jj_csfs
from atomx.config.jj_coupling import generate_jj_average
from atomx.physics.csf import matrix
from atomx.physics.csf.breit_ci import breit_configuration_average


def _radial(key):
    # Enforce electron-interchange and bra/ket symmetries of real radial
    # integrals, but give every multipole/component combination its own value.
    _, rank, a, c, b, d = key
    pairs = tuple(sorted((tuple(sorted((a, c))), tuple(sorted((b, d))))))
    digest = hashlib.sha256(repr((rank, pairs)).encode()).digest()
    return int.from_bytes(digest[:8], "little") / 2**64


@pytest.mark.parametrize("config", ["1s2", "2p1", "2p2", "3d2", "4f2", "2p1 3d1"])
@pytest.mark.parametrize("averaging", ["statistical", "equal"])
def test_ensemble_matches_degeneracy_weighted_csf_trace(monkeypatch, config, averaging):
    """Tr(rho G) is basis independent, including jj occupation correlations."""
    monkeypatch.setattr(matrix, "_breit_radial_evaluator", lambda *args: _radial)
    _, configs = generate_jj_average(config, averaging=averaging)
    actual = breit_configuration_average({}, None, 1, 0, configs)
    # Evaluate separate CSF diagonals, weighted by their 2J+1 magnetic
    # degeneracy within each jj configuration. This uses determinant
    # projections, independently of the direct pair-probability contraction.
    traces = defaultdict(float)
    dimensions = defaultdict(int)
    for csf in enumerate_jj_csfs(config):
        signature = csf_signature(csf)
        block = exact_block_matrix(
            [signature], lambda key: 0.0, breit_evaluator=_radial, verify=False)
        occupations = tuple((sh.n, sh.kappa, int(sh.occupation))
                            for sh in csf.subshells if sh.occupation)
        weight = csf.total_J2 + 1
        traces[occupations] += weight * block[0, 0]
        dimensions[occupations] += weight
    expected = 0.0
    for cfg in configs:
        occupations = tuple((sh.n, sh.kappa, int(sh.occupation))
                            for sh in cfg.subshells if sh.occupation)
        expected += cfg.weight * traces[occupations] / dimensions[occupations]
    assert actual == pytest.approx(expected, rel=2e-12, abs=2e-13)


def test_open_shell_scf_corrections_preserve_density():
    from atomx import Atom, Corrections, Correlation
    from atomx.corrections.stack import CorrectionStack

    atom = Atom(6, engine="bspline", correlation=Correlation(averaging="statistical"),
                corrections=Corrections(stack=CorrectionStack(
                    breit_mode="operator"))).solve(tol=1e-6)
    assert atom.converged
    assert np.isfinite(atom.total_energy)
    assert atom.result.solver_diagnostics["_E_Breit_operator"] != 0.0
    waves = atom.result.waves
    count = sum(sh.occupation * np.trapezoid(
        waves.orbitals[sh.label].P**2 + waves.orbitals[sh.label].Q**2, waves.grid)
                for sh in atom.result.subshells)
    assert count == pytest.approx(6, abs=2e-3)
