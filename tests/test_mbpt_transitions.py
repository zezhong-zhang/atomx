"""Nonzero self-energy gates against explicit spin-orbital sums."""

from itertools import product
from unittest.mock import patch

import pytest
from test_review_regressions import _coulomb

from atomx.constants import HARTREE_EV
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.mbpt_transitions import (
    DiagonalSelfEnergy,
    build_mbpt2_self_energy,
    mbpt2_orbital_energy_shift,
    mbpt2_transition_energy,
)


@pytest.fixture(scope="module")
def helium():
    return DFSolver(
        Z=4, subshells=[Subshell(n=1, kappa=-1, occupation=2.0)], ns=30, ks=7
    ).solve()


@pytest.mark.parametrize("external_kappa", [None, -1, -2, -3])
@pytest.mark.parametrize("energy_offset", [0.0, 0.071])
def test_self_energy_matches_spin_orbital_sum(helium, external_kappa, energy_offset):
    from atomx.angular.csf_hamiltonian import _m2_values, twoj
    from atomx.methods.mbpt import build_virtual_basis

    occupied = [(1, -1)]
    od = {key: (P, Q) for key, (P, Q, q) in helium.orbitals_by_subshell().items()}
    eps = {(1, -1): helium.orbital_energies["1s1/2"]}
    blocks = {}
    kappas = (-1, 1) if external_kappa in (None, -1) else (-1, 1, -2, 2, -3)
    for k in kappas:
        _, states = build_virtual_basis(
            helium, k, e_max=50.0, orthogonal_to_occupied=True
        )
        blocks[k] = states[:2]
        for i, (P, Q, e) in enumerate(blocks[k]):
            od[(100 + i, k)] = (P, Q)
            eps[(100 + i, k)] = e
    virt = [key for key in od if key not in occupied]

    def spin(keys):
        return [(*key, m) for key in keys for m in _m2_values(twoj(key[1]))]

    core, vspin = spin(occupied), spin(virt)
    target_key = (1, -1)
    options = dict(kappa_max=1, n_virt_max=2)
    label = "1s1/2"
    if external_kappa is not None:
        # The external target is already one of the virtuals, but is never
        # inserted into the occupied reference or excluded from that sum.
        target_key = (100, external_kappa)
        P, Q, e = blocks[external_kappa][0]
        options.update(valence_orbital=(P, Q, external_kappa, e), virtual_kappas=kappas)
        label = "external"
        assert label not in helium.orbital_energies
    aspin = spin([target_key])
    g = _coulomb(od, helium._bsp_data)
    energy = eps[target_key] + energy_offset
    options["evaluation_energy_au"] = energy
    reference = 0.0
    for a in aspin:
        for b, r, s in product(core, vspin, vspin):
            anti = g(a, b, r, s) - g(a, b, s, r)
            reference += (
                0.5
                * anti**2
                / (energy + eps[b[:2]] - eps[r[:2]] - eps[s[:2]])
                / len(aspin)
            )
        for b, c, r in product(core, core, vspin):
            anti = g(a, r, b, c) - g(a, r, c, b)
            reference += (
                0.5
                * anti**2
                / (energy + eps[r[:2]] - eps[b[:2]] - eps[c[:2]])
                / len(aspin)
            )

    def basis(scf, kappa, **kwargs):
        assert kwargs["potential_mode"] == "VN"
        assert kwargs["orthogonal_to_occupied"]
        return [], blocks[kappa]

    terms = []
    with patch("atomx.methods.mbpt.build_virtual_basis", side_effect=basis) as build:
        actual = mbpt2_orbital_energy_shift(
            helium, label, term_recorder=terms.append, **options
        )
    assert abs(reference) > 1e-6
    assert actual == pytest.approx(reference, rel=1e-11, abs=1e-13)
    assert sum(t["contribution_Ha"] for t in terms) == pytest.approx(actual, abs=1e-13)
    assert {t["diagram"] for t in terms} == {"2p1h", "2h1p"}
    assert build.call_count == len(kappas)  # One eigensystem per kappa, independent of r/s loops.
    assert any(e < 0 for _, _, e in blocks[-1])  # Bound virtuals contribute.
    # The reusable off-shell expansion is checked against the independent
    # spin-orbital observable above, not just a duplicate native call.
    options.pop("evaluation_energy_au")
    with patch("atomx.methods.mbpt.build_virtual_basis", side_effect=basis):
        snapshot = build_mbpt2_self_energy(
            helium, label, reference_identity=f"test:Z4:{target_key}", **options
        )
    assert snapshot.evaluate(energy) == pytest.approx(reference, rel=1e-11, abs=1e-13)
    step = 1e-5
    finite_difference = (snapshot.evaluate(energy+step)-snapshot.evaluate(energy-step))/(2*step)
    assert snapshot.derivative(energy) == pytest.approx(finite_difference, rel=1e-7, abs=1e-11)


@pytest.mark.parametrize("pole", [-3.0, 2.0])
def test_dyson_root_and_residue_match_hermitian_eigenproblem(pole):
    """Feshbach elimination of one auxiliary state provides an independent control."""
    import numpy as np

    epsilon, coupling = -0.5, 0.3
    hamiltonian = np.array([[epsilon, coupling], [coupling, pole]])
    energies, vectors = np.linalg.eigh(hamiltonian)
    selected = np.argmax(abs(vectors[0])**2)
    snapshot = DiagonalSelfEnergy(epsilon, np.array([pole]), np.array([coupling**2]),
                                  "test:two-state-Hermitian")
    bracket = (epsilon-0.1, epsilon) if pole > epsilon else (epsilon, epsilon+0.1)
    root = snapshot.solve_dyson(bracket)
    assert root.energy_Ha == pytest.approx(energies[selected], abs=1e-13)
    assert root.residue == pytest.approx(abs(vectors[0, selected])**2, abs=1e-13)
    assert abs(root.residual_Ha) < 1e-13
    assert root.reference_identity == snapshot.reference_identity


def test_dyson_pole_branch_and_snapshot_contract():
    import numpy as np

    # A zero coupling is not a pole, and input arrays cannot mutate a snapshot.
    poles, weights = np.array([0.0, 0.01]), np.array([0.2, 0.0])
    snapshot = DiagonalSelfEnergy(-0.5, poles, weights, "test:identity")
    poles[0] = 9.0
    weights[0] = 7.0
    assert snapshot.evaluate(-0.5) == pytest.approx(-0.4)
    with pytest.raises(ValueError, match="WRITEABLE"):
        snapshot.poles_Ha.setflags(write=True)
    with pytest.raises(ValueError, match="crosses"):
        snapshot.solve_dyson((-1.0, 0.1))
    with pytest.raises(ValueError, match="contain the DF"):
        snapshot.solve_dyson((-0.3, -0.1))
    with pytest.raises(ValueError, match="resonant"):
        snapshot.evaluate(1e-9)
    with pytest.raises(ValueError, match="finite"):
        snapshot.evaluate(float("nan"))
    with pytest.raises(ValueError, match="real scalar"):
        snapshot.evaluate(-0.5 + 0.01j)
    with pytest.raises(ValueError, match="real scalar"):
        snapshot.evaluate(np.array([-0.5]))
    with pytest.raises(ValueError, match="invalid"):
        DiagonalSelfEnergy(-0.5, np.array([0.0]), np.array([-1.0]), "test")
    with pytest.raises(ValueError, match="invalid"):
        DiagonalSelfEnergy(-0.5, np.array([0.0]), np.array([1.0]), "")
    # No weighted poles gives exactly the reference energy and unit residue.
    empty = DiagonalSelfEnergy(-0.5, np.array([0.0]), np.array([0.0]), "test:empty")
    root = empty.solve_dyson((-1.0, 0.0))
    assert root.energy_Ha == -0.5 and root.residue == 1.0


def test_external_float32_reference_keeps_legacy_double_denominators(helium):
    import numpy as np

    from atomx.methods.mbpt import build_virtual_basis

    _, states = build_virtual_basis(helium, -1, e_max=50., potential_mode="VN",
                                    orthogonal_to_occupied=True)
    P, Q, e = states[0]
    low_precision = np.float32(e)
    options = dict(virtual_kappas=(-1, 1), n_virt_max=2)
    actual = mbpt2_orbital_energy_shift(helium, "external",
                                      valence_orbital=(P, Q, -1, low_precision), **options)
    expected = mbpt2_orbital_energy_shift(helium, "external",
                                        valence_orbital=(P, Q, -1, float(low_precision)), **options)
    assert actual == expected


def test_external_orbital_metric_and_occupation_contract(helium):
    P, Q, _ = helium.orbitals_raw[-1][0]
    e = helium.orbital_energies["1s1/2"]
    with pytest.raises(ValueError, match="orthogonal to the core"):
        mbpt2_orbital_energy_shift(helium, "external", valence_orbital=(P, Q, -1, e))
    with pytest.raises(ValueError, match="normalized"):
        mbpt2_orbital_energy_shift(helium, "external", valence_orbital=(2*P, Q, -1, e))
    with pytest.raises(ValueError, match="distinct nonzero"):
        mbpt2_orbital_energy_shift(helium, "1s1/2", virtual_kappas=(-1, -1))
    assert helium.subshells[0].occupation == 2.0


def test_invalid_orbital_label(helium):
    with pytest.raises(ValueError, match="not found"):
        mbpt2_orbital_energy_shift(helium, "9z9/2")


def test_off_shell_complex_energy_is_explicitly_rejected(helium):
    with pytest.raises(ValueError, match="real scalar"):
        mbpt2_orbital_energy_shift(helium, "1s1/2", evaluation_energy_au=-0.5+0.01j)


def test_virtual_basis_errors_propagate(helium):
    with (
        patch(
            "atomx.methods.mbpt.build_virtual_basis",
            side_effect=RuntimeError("eigensolver failed"),
        ),
        pytest.raises(RuntimeError, match="eigensolver failed"),
    ):
        mbpt2_orbital_energy_shift(helium, "1s1/2")


def test_self_transition_zero_with_nonzero_equal_shifts(helium):
    result = mbpt2_transition_energy(
        helium, "1s1/2", "1s1/2", kappa_max=1, n_virt_max=2
    )
    assert abs(result["delta_E_alpha_au"]) > 1e-6
    assert result["delta_E_alpha_au"] == result["delta_E_beta_au"]
    assert result["omega_DHF_au"] == result["omega_corrected_au"] == 0.0


def test_nonzero_transition_difference_and_units(helium):
    from types import SimpleNamespace

    fake = SimpleNamespace(orbital_energies={"a": -1.0, "b": -0.2})
    with patch(
        "atomx.methods.mbpt_transitions.mbpt2_orbital_energy_shift",
        side_effect=[0.03, -0.01],
    ):
        result = mbpt2_transition_energy(fake, "a", "b")
    assert result["omega_corrected_au"] == pytest.approx(0.76)
    assert result["omega_corrected_eV"] == pytest.approx(0.76 * HARTREE_EV)
