"""Independent complete particle-sector anchors for native finite attachment."""

import itertools
from dataclasses import replace

import numpy as np
import pytest

from atomx.methods.eom_attachment import compute
from atomx.physics.csf.spinor_integrals import SpinorIntegrals


def integrals(phases=None, frozen=-2.3, spinors=6):
    rng = np.random.default_rng(194025)
    n, occupied = spinors, 3
    h = np.diag([-2.1, -1.6, -1.2, -.7, -.4, -.15, .1][:n])
    x = rng.normal(size=(n, n)) * .025
    h += x + x.T
    pairs = tuple(itertools.combinations(range(n), 2))
    lookup = {pair: i for i, pair in enumerate(pairs)}
    x = rng.normal(size=(len(pairs), len(pairs))) * .018
    pair_matrix = x + x.T
    if phases is not None:
        h = phases[:, None] * h * phases[None, :]
        pair_phases = np.array([phases[p] * phases[q] for p, q in pairs])
        pair_matrix = pair_phases[:, None] * pair_matrix * pair_phases[None, :]

    def eri(p, q, r, s):
        if p == q or r == s:
            return 0
        sign = (-1 if p > q else 1) * (-1 if r > s else 1)
        return sign * pair_matrix[lookup[tuple(sorted((p, q)))], lookup[tuple(sorted((r, s)))]]

    packed = np.array([eri(p, q, r, s)
                       for p in range(n - 1) for q in range(p + 1, n)
                       for r in range(p, n - 1) for s in range(r + 1, n)])
    fock = h + np.array([[sum(eri(p, i, q, i) for i in range(occupied))
                          for q in range(n)] for p in range(n)])
    energy = frozen + np.trace(h[:occupied, :occupied]) + .5 * sum(
        eri(i, j, i, j) for i in range(occupied) for j in range(occupied))
    obj = SpinorIntegrals(tuple((p + 1, -1, 1) for p in range(n)), occupied,
                         h, fock, packed, frozen, energy, {"scope": "finite two-body anchor"})
    return obj, h, eri


def independent_hamiltonian(obj, h, eri, electrons):
    """Occupation-list action with the full 1/4 tensor, not pair-reduced code."""
    n = len(obj.labels)
    masks = tuple(sum(1 << p for p in row)
                  for row in itertools.combinations(range(n), electrons))
    index = {mask: i for i, mask in enumerate(masks)}
    matrix = np.eye(len(masks)) * obj.frozen_energy_Ha
    terms = [(h[p, q], ((p, True), (q, False)))
             for p in range(n) for q in range(n)]
    terms += [(.25 * eri(p, q, r, s),
               ((p, True), (q, True), (s, False), (r, False)))
              for p, q, r, s in itertools.product(range(n), repeat=4)]
    for col, mask in enumerate(masks):
        for value, operations in terms:
            if value == 0:
                continue
            occupied = [p for p in range(n) if mask & (1 << p)]
            sign = 1
            for p, create in reversed(operations):
                if (p in occupied) == create:
                    break
                sign *= (-1)**sum(v < p for v in occupied)
                if create:
                    occupied.append(p)
                else:
                    occupied.remove(p)
            else:
                target = sum(1 << p for p in occupied)
                matrix[index[target], col] += value * sign
    assert np.max(abs(matrix - matrix.T)) < 1e-14
    return masks, .5 * (matrix + matrix.T)


def test_core_triples_and_complete_attachment_match_independent_fci():
    obj, h, eri = integrals()
    result = compute(obj)
    _, hc = independent_hamiltonian(obj, h, eri, 3)
    masks, ha = independent_hamiltonian(obj, h, eri, 4)
    ec = np.linalg.eigvalsh(hc)[0]
    ea, va = np.linalg.eigh(ha)
    assert result.core.total_energy_Ha == pytest.approx(ec, abs=1e-11)
    assert result.provenance["attached_electrons"] == 4
    assert result.provenance["projected_determinants"] == len(masks) == 15
    triples = [value for value, (holes, _particles) in zip(
        result.core.amplitudes, result.core.provenance["excitations"], strict=True)
               if len(holes) == 3]
    assert np.max(np.abs(triples)) > 1e-6
    np.testing.assert_allclose(result.attachment_energies_Ha, ea - ec, atol=1e-11, rtol=0)
    assert np.max(result.projected_residuals_Ha) < 1e-10
    assert np.max(result.full_eigenstate_residuals_Ha) < 1e-10
    for actual, expected in zip(result.coefficients.T, va.T, strict=True):
        np.testing.assert_allclose(np.outer(actual, actual.conj()),
                                   np.outer(expected, expected), atol=1e-10, rtol=0)
    assert np.max(abs(result.projected_operator_Ha - result.projected_operator_Ha.T)) > 1e-4


def test_omitted_attachment_rank_is_not_hidden_by_projected_residual():
    obj, _, _ = integrals()
    full = compute(obj)
    reduced = compute(obj, attachment_particle_rank=2)
    assert len(reduced.projected_masks) == 12
    assert np.max(reduced.projected_residuals_Ha) < 1e-10
    assert np.max(reduced.full_eigenstate_residuals_Ha) > 1e-4
    assert abs(reduced.attachment_energies_Ha[0] - full.attachment_energies_Ha[0]) > 1e-7
    assert "core=CCSDT,R=2p1h" in reduced.method


def test_nonzero_core_triples_lift_against_independent_signed_similarity():
    """Seven spinors leave a nonzero T3 action and an omitted 4p3h state.

    Hypothesis: correctly normalized core excitation signs are preserved in
    N+1. All 35 attached determinants and 34 R<=3p2h coordinates are kept;
    independently build the full signed lift/exponential, require nonzero T3,
    compare every Hbar entry and every physical state at 1e-10, and retain
    the omitted-rank full-H defect. Stop on any mismatch without changing rank.
    """
    from scipy.linalg import expm

    from atomx.methods.coupled_cluster import _cluster_generators

    obj, h, eri = integrals(spinors=7)
    result = compute(obj)
    masks, ha = independent_hamiltonian(obj, h, eri, 4)
    index = {mask: i for i, mask in enumerate(masks)}
    reference = (1 << 3) - 1
    independent_t = np.zeros((35, 35))
    triples_t = np.zeros_like(independent_t)

    def act(mask, holes, particles):
        occupied = [p for p in range(7) if mask & (1 << p)]
        sign = 1
        operations = tuple((p, True) for p in particles) + tuple(
            (p, False) for p in reversed(holes))
        for p, create in reversed(operations):
            if (p in occupied) == create:
                return None
            sign *= (-1)**sum(v < p for v in occupied)
            if create:
                occupied.append(p)
            else:
                occupied.remove(p)
        return sum(1 << p for p in occupied), sign

    for amplitude, (holes, particles) in zip(
            result.core.amplitudes, result.core.provenance["excitations"], strict=True):
        _, reference_sign = act(reference, holes, particles)
        for col, mask in enumerate(masks):
            action = act(mask, holes, particles)
            if action is not None:
                end, sign = action
                value = amplitude * sign / reference_sign
                independent_t[index[end], col] += value
                if len(holes) == 3:
                    triples_t[index[end], col] += value
    assert np.max(abs(triples_t)) > 1e-6
    actual_t = np.einsum("a,aij->ij", result.core.amplitudes, _cluster_generators(
        masks, reference, result.core.provenance["excitations"]))
    np.testing.assert_allclose(actual_t, independent_t, atol=1e-14, rtol=0)
    rows = [index[mask] for mask in result.projected_masks]
    assert len(masks) == 35 and len(rows) == 34
    assert sum(1 << p for p in (3, 4, 5, 6)) not in result.projected_masks
    independent_hbar = expm(-independent_t) @ ha @ expm(independent_t)
    independent_projected = independent_hbar[np.ix_(rows, rows)]
    independent_projected -= result.core.total_energy_Ha * np.eye(34)
    np.testing.assert_allclose(result.projected_operator_Ha, independent_projected,
                               atol=1e-12, rtol=0)
    expected_energies = np.linalg.eigvals(independent_projected)
    assert np.max(abs(expected_energies.imag)) < 1e-10
    np.testing.assert_allclose(result.attachment_energies_Ha,
                               np.sort(expected_energies.real), atol=1e-10, rtol=0)
    for right, actual in zip(result.projected_right_vectors.T, result.coefficients.T, strict=True):
        lifted = np.zeros(35, dtype=right.dtype)
        lifted[rows] = right
        expected = expm(independent_t) @ lifted
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(np.outer(actual, actual.conj()),
                                   np.outer(expected, expected.conj()), atol=1e-10, rtol=0)
    assert np.max(result.projected_residuals_Ha) < 1e-10
    assert np.max(result.full_eigenstate_residuals_Ha) > 1e-6


def test_signed_phase_covariance_ownership_and_common_energy_zero():
    obj, _, _ = integrals()
    phases = np.array([-1., 1., -1., 1., -1., 1.])
    changed, _, _ = integrals(phases)
    a, b = compute(obj), compute(changed)
    determinant_phases = np.array([np.prod([phases[p] for p in range(6) if mask & (1 << p)])
                                   for mask in a.determinant_masks])
    np.testing.assert_allclose(a.attachment_energies_Ha, b.attachment_energies_Ha, atol=1e-11, rtol=0)
    for ca, cb in zip(a.coefficients.T, b.coefficients.T, strict=True):
        expected = ca * determinant_phases
        np.testing.assert_allclose(np.outer(cb, cb.conj()),
                                   np.outer(expected, expected.conj()), atol=1e-10, rtol=0)
    shifted, _, _ = integrals(frozen=obj.frozen_energy_Ha + 7)
    s = compute(shifted)
    np.testing.assert_allclose(a.attachment_energies_Ha, s.attachment_energies_Ha, atol=1e-11, rtol=0)
    assert s.core.total_energy_Ha - a.core.total_energy_Ha == pytest.approx(7, abs=1e-11)
    for value in (a.coefficients, a.projected_operator_Ha, a.attachment_energies_Ha):
        with pytest.raises(ValueError):
            value.setflags(write=True)
    with pytest.raises(TypeError):
        a.provenance["attached_electrons"] = 5
    nested = {"nucleus": {"kind": "point", "radius": [0.]},
              "physical_source_sha256": {"test": "original"}}
    metadata_obj = replace(obj, identity=nested)
    owned = compute(metadata_obj)
    nested["nucleus"]["radius"][0] = 9.
    nested["physical_source_sha256"]["test"] = "changed"
    provenance = owned.provenance["integral_producer_identity"]
    assert provenance["nucleus"]["radius"] == (0.,)
    assert provenance["physical_source_sha256"]["test"] == "original"
    with pytest.raises(TypeError):
        provenance["nucleus"]["kind"] = "Fermi"


def test_guards_stop_before_operator_and_complex_model_is_explicit(monkeypatch):
    obj, _, _ = integrals()

    def forbidden(*args):
        raise AssertionError("operator evaluated before complete resource guard")

    with monkeypatch.context() as patch:
        patch.setattr(SpinorIntegrals, "eri", forbidden)
        for kwargs, message in [({"determinant_cap": 1}, "particle sectors"),
                                ({"amplitude_cap": 1}, "amplitude cap"),
                                ({"logical_term_action_cap": 1}, "term action"),
                                ({"generator_memory_MiB": .0001}, "generator memory")]:
            with pytest.raises(ValueError, match=message):
                compute(obj, **kwargs)
    for kwargs in ({"attachment_particle_rank": True}, {"attachment_particle_rank": 4},
                   {"residual_tolerance_Ha": np.nan}, {"core_method": "CCSD(T)"},
                   {"generator_memory_MiB": np.bool_(True)}):
        with pytest.raises((ValueError, NotImplementedError)):
            compute(obj, **kwargs)
    bad_h = obj.one_body_Ha.copy()
    bad_h[0, 4] += .01j
    bad_h[4, 0] -= .01j
    bad_fock = obj.fock_Ha + bad_h - obj.one_body_Ha
    complex_obj = replace(obj, one_body_Ha=bad_h, fock_Ha=bad_fock)
    with pytest.raises(NotImplementedError, match="exactly real"):
        compute(complex_obj)


def test_native_be_attachment_matches_complete_dirac_coulomb_sectors():
    """Live B-spline/angular integrals, independently assembled N2/N3 sectors.

    This deliberately small complete 1s/2s/3s pool establishes the native
    operator/attachment seam, not the accuracy of the physical Be spectrum.
    """
    from atomx.methods.dirac_fock import DFSolver, Subshell
    from atomx.methods.mbpt import build_virtual_basis
    from atomx.physics.csf.sparse_ci import assemble_sparse_hamiltonian
    from atomx.physics.csf.spinor_integrals import build_spinor_integrals

    core = DFSolver(Z=4, subshells=[Subshell(1, -1, 2)], ns=30, ks=7).solve()
    orbitals = {(1, -1): core.orbitals_by_subshell()[1, -1][:2]}
    _, virtuals = build_virtual_basis(
        core, -1, e_max=50, potential_mode="VN", orthogonal_to_occupied=True,
    )
    for principal, state in zip((2, 3), virtuals[:2], strict=True):
        orbitals[principal, -1] = state[:2]
    obj = build_spinor_integrals(
        orbitals, core._bsp_data, 4, 30, occupied_keys=[(1, -1)],
        virtual_keys=[(2, -1), (3, -1)], nuclear_potential=core._nuclear_potential,
    )
    result = compute(obj)
    reference_energies = []
    for electrons in (2, 3):
        masks = tuple(sum(1 << p for p in row)
                      for row in itertools.combinations(range(6), electrons))
        h = assemble_sparse_hamiltonian(
            masks, obj.labels, orbitals, core._bsp_data, 4, 30, method="python",
        ).toarray()
        # The independent native producer evaluates both triangles separately.
        # Bound and record its roundoff before the oracle projection; the
        # attachment consumer continues to require exact symmetric input.
        assert np.max(abs(h - h.T)) < 1e-12
        reference_energies.append(np.linalg.eigvalsh(.5 * (h + h.T)))
    ec, ea = reference_energies
    assert result.provenance["core_determinants"] == 15
    assert result.provenance["attached_determinants"] == 20
    assert result.provenance["projected_determinants"] == 20
    assert result.core.total_energy_Ha == pytest.approx(ec[0], abs=1e-10)
    np.testing.assert_allclose(result.attachment_energies_Ha, ea - ec[0],
                               atol=1e-10, rtol=0)
    assert np.max(result.full_eigenstate_residuals_Ha) < 1e-10
