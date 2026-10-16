"""Standalone SD: primary diagram equations, independent Fock algebra and state."""

import importlib.util
import itertools
import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy.optimize import root

from atomx.methods.core_sd import CoreSD, plan_core_sd
from atomx.methods.dirac_fock import DFSolver, Subshell
from atomx.methods.mbpt import build_virtual_basis
from atomx.methods.monovalent_sd import (
    FiniteMonovalentSD,
    one_body_matrix_element,
    plan_finite_monovalent_sd,
)
from atomx.numerics.bspline.coulomb_features import PolynomialCellCoulombFeatures


def elementary(mask, operations):
    """Independent bit-by-bit action, without the production sign helper."""
    sign = 1
    for bit, create in reversed(operations):
        if bool(mask & (1 << bit)) == create:
            return None, 0
        sign *= (-1) ** ((mask & ((1 << bit) - 1)).bit_count())
        mask ^= 1 << bit
    return mask, sign


def full_operator(n, electrons, terms):
    masks = tuple(
        sum(1 << i for i in row) for row in itertools.combinations(range(n), electrons)
    )
    positions = {mask: i for i, mask in enumerate(masks)}
    matrix = np.zeros((len(masks), len(masks)))
    for j, mask in enumerate(masks):
        for value, operations in terms:
            target, sign = elementary(mask, operations)
            if sign:
                matrix[positions[target], j] += sign * value
    return masks, matrix


def amplitude_tensors(problem, amplitudes):
    """Physical antisymmetric coefficients from the declared operator strings."""
    n, o = len(problem.integrals.labels), problem.integrals.noccupied
    cs = np.zeros((n, o))
    cd = np.zeros((n, n, o, o))
    vs = np.zeros(n)
    vd = np.zeros((n, n, o))
    for value, ops in problem.core_terms:
        if len(ops) == 2:
            cs[ops[0][0], ops[1][0]] = value
        else:
            m, nn, b, a = (row[0] for row in ops)
            cd[m, nn, a, b] = cd[nn, m, b, a] = value
            cd[nn, m, a, b] = cd[m, nn, b, a] = -value
    for value, (particles, holes, _) in zip(
        amplitudes, problem.excitations, strict=True
    ):
        if len(particles) == 1:
            vs[particles[0]] = value
        else:
            m, nn = particles
            b = holes[1]
            vd[m, nn, b], vd[nn, m, b] = value, -value
    return cs, 0.5 * cd, vs, 0.5 * vd


def paper_rhs(g, cs, cd, vs, vd, column):
    """Literal independent Eq.8/9/13 contractions, fixed core amplitudes.

    g is direct, cd/vd are unantisymmetrized rho. No production commutator,
    linked map, denominator insertion or eigenvalue solver is used here.
    """
    n, o = cs.shape
    occupied, virtual = range(o), range(o, n)
    gt = g - g.swapaxes(2, 3)
    rs, rd = np.zeros(n), np.zeros_like(vd)
    for m in virtual:
        for b, nn in itertools.product(occupied, virtual):
            rs[m] += gt[m, b, column, nn] * cs[nn, b]
        for b, nn, r in itertools.product(occupied, virtual, virtual):
            rs[m] += g[m, b, nn, r] * (vd[nn, r, b] - vd[r, nn, b])
        for b, c, nn in itertools.product(occupied, occupied, virtual):
            rs[m] -= g[b, c, column, nn] * (cd[m, nn, b, c] - cd[nn, m, b, c])
    for m, nn, b in itertools.product(virtual, virtual, occupied):
        rd[m, nn, b] = g[m, nn, column, b]
        for c, d in itertools.product(occupied, repeat=2):
            rd[m, nn, b] += g[c, d, column, b] * cd[m, nn, c, d]
        for r, s in itertools.product(virtual, repeat=2):
            rd[m, nn, b] += g[m, nn, r, s] * vd[r, s, b]
        for r in virtual:
            rd[m, nn, b] += g[m, nn, r, b] * vs[r] + g[m, nn, column, r] * cs[r, b]
        for c in occupied:
            rd[m, nn, b] -= (
                g[c, nn, column, b] * cs[m, c] + g[m, c, column, b] * cs[nn, c]
            )
        for r, c in itertools.product(virtual, occupied):
            rd[m, nn, b] += gt[c, nn, r, b] * (vd[m, r, c] - vd[r, m, c])
            rd[m, nn, b] += gt[m, c, column, r] * (cd[r, nn, c, b] - cd[nn, r, c, b])
    return rs, rd - rd.swapaxes(0, 1)


def dense_linked(problem, amplitudes, hterms=None):
    """Full fixed-N matrices on both sectors; independent elementary signs."""
    n, o = len(problem.integrals.labels), problem.integrals.noccupied
    hterms = problem._hterms if hterms is None else hterms
    cm, hc = full_operator(n, o, hterms)
    am, ha = full_operator(n, o + 1, hterms)
    _, tc = full_operator(n, o, problem.core_terms)
    _, ta = full_operator(n, o + 1, problem.core_terms)
    terms = tuple(
        (value, row[2])
        for value, row in zip(amplitudes, problem.excitations, strict=True)
    )
    # Independently expand scalar S1 on every M of the valence radial shell.
    # Other-M strings are zero on the reference but not after a core F_vo.
    extra_singles = []
    labels = problem.integrals.labels
    for value, row in zip(amplitudes, problem.excitations, strict=True):
        if len(row[0]) != 1:
            continue
        m, v = row[0][0], row[1][0]
        if labels[m][1] != labels[v][1]:
            continue
        for other, label in enumerate(labels):
            if label[:2] == labels[v][:2] and other != v:
                destination = labels.index((*labels[m][:2], label[2]))
                extra_singles.append((value, ((destination, True), (other, False))))
    _, sa = full_operator(n, o + 1, terms)
    cc, aa = cm.index(problem.core_mask), am.index(problem.reference_mask)
    attachment = np.zeros((len(am), len(cm)))
    for j, mask in enumerate(cm):
        target, sign = elementary(mask, ((problem.valence_index, True),))
        if sign:
            attachment[am.index(target), j] = sign
    phase = elementary(problem.core_mask, ((problem.valence_index, True),))[1]
    vector = (ha + ha @ (ta + sa) - (ta + sa) @ ha)[:, aa]
    vector -= phase * attachment @ (hc + hc @ tc - tc @ hc)[:, cc]
    # Atomic SD uses core normal ordering, not attached-reference LCC.
    # Keep its required valence-single diagram: an ordinary commutator
    # subtracts S1 times the valence-induced core-single part of H|Phi_v>.
    source = ha[:, aa] - phase * attachment @ hc[:, cc]
    for i, mask in enumerate(am):
        if (
            not mask & (1 << problem.valence_index)
            or (problem.reference_mask & ~mask).bit_count() != 1
        ):
            source[i] = 0
    singles = tuple(row for row in terms if len(row[1]) == 2)
    _, s1 = full_operator(n, o + 1, singles)
    vector += s1 @ source
    # Obtain the bare-core Fock source from a separate many-body sector,
    # without consuming the production Fock matrix or normal-order helper.
    # Other-M S1 must not act on the full attached Coulomb source: those
    # contractions are excluded by the canonical atomic SD equations.
    bare_core = phase * attachment @ hc[:, cc]
    for i, mask in enumerate(am):
        if (problem.reference_mask & ~mask).bit_count() != 1:
            bare_core[i] = 0
    _, extra_s1 = full_operator(n, o + 1, tuple(extra_singles))
    vector -= extra_s1 @ bare_core
    values = [vector[aa]]
    for _, _, operations in problem.excitations:
        target, sign = elementary(problem.reference_mask, operations)
        values.append(sign * vector[am.index(target)])
    return np.array(values), (am, ha, ta, sa)


@pytest.fixture(scope="module")
def live_core(tmp_path_factory):
    source, compiler = (
        os.environ.get("ATOMX_PCI_SOURCE"),
        os.environ.get("ATOMX_PCI_COMPILER"),
    )
    if not source or not compiler:
        pytest.skip(
            "pinned pCI source/compiler required for native standalone SD anchor"
        )
    script = Path(__file__).resolve().parents[1] / "tools/build_pci_sd_backend.py"
    spec = importlib.util.spec_from_file_location("monovalent_core_builder", script)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    core = DFSolver(Z=4, subshells=[Subshell(1, -1, 2)], ns=30, ks=7).solve()
    orbitals = {(1, -1): core.orbitals_by_subshell()[1, -1][:2]}
    virtual = ((2, -1), (3, -1), (2, 1), (2, -2))
    for k in (-1, 1, -2):
        _, states = build_virtual_basis(
            core, k, e_max=50, potential_mode="VN", orthogonal_to_occupied=True
        )
        for i, state in enumerate(states[: 2 if k == -1 else 1]):
            orbitals[i + 2, k] = state[:2]
    plan = plan_core_sd(
        ((1, -1),), virtual, PolynomialCellCoulombFeatures(core._bsp_data).identity
    )
    directory = tmp_path_factory.mktemp("monovalent-native") / "backend"
    builder.build(
        source,
        directory,
        plan["dimensions"],
        compiler,
        runtime_directory=os.environ.get("ATOMX_PCI_RUNTIME"),
    )
    problem = CoreSD(
        orbitals,
        core._bsp_data,
        4,
        30,
        occupied_keys=((1, -1),),
        virtual_keys=virtual,
        backend_directory=directory,
        nuclear_potential=core._nuclear_potential,
        radial_representation="polynomial",
    )
    result = problem.solve(residual_tolerance_Ha=1e-12)
    return problem, result


def problem_for(live_core, key=(2, -1), m=-1):
    core, result = live_core
    v = core.integrals.labels.index((*key, m))
    return FiniteMonovalentSD(core, result, v, work_cap=200_000_000)


def test_complete_planning_and_preflight(live_core):
    core, result = live_core
    v = core.integrals.labels.index((2, -1, -1))
    plan = plan_finite_monovalent_sd(core.integrals.labels, 2, v, work_cap=200_000_000)
    assert (plan["singles"], plan["doubles"]) == (1, 13)
    assert plan["attached_determinant_bound"] == 220
    for caps in (
        dict(memory_bytes=1),
        dict(work_cap=1),
        dict(amplitude_cap=1),
        dict(determinant_cap=1),
    ):
        with (
            patch.object(
                type(core.integrals),
                "eri",
                side_effect=AssertionError("preflight too late"),
            ),
            pytest.raises(MemoryError, match="complete finite"),
        ):
            FiniteMonovalentSD(core, result, v, **caps)
    with pytest.raises(ValueError, match="positive integer"):
        plan_finite_monovalent_sd(core.integrals.labels, 2, v, work_cap=True)
    with pytest.raises(ValueError, match="virtual valence"):
        plan_finite_monovalent_sd(core.integrals.labels, 2, 0)


@pytest.mark.parametrize("key", [(2, -1), (2, 1), (2, -2)])
def test_primary_canonical_equations_and_independent_noncanonical_commutator(
    live_core, key
):
    problem = problem_for(live_core, key)
    n = len(problem.integrals.labels)
    g = np.empty((n,) * 4)
    for indices in itertools.product(range(n), repeat=4):
        g[indices] = 0.5 * problem.integrals.eri(*indices).real
    rng = np.random.default_rng(41319)
    amplitudes = rng.normal(size=problem.plan["amplitudes"]) * 0.013
    cs, cd, vs, vd = amplitude_tensors(problem, amplitudes)
    rs, rd = paper_rhs(g, cs, cd, vs, vd, problem.valence_index)
    paper = [rs[problem.valence_index] + problem.orbital_energy_Ha]
    for particles, holes, _ in problem.excitations:
        paper.append(
            rs[particles[0]]
            if len(particles) == 1
            else rd[particles[0], particles[1], holes[1]]
        )
    paper = np.asarray(paper)
    paper[1:] += problem.diagonal_gaps_Ha * amplitudes
    f = problem.integrals.fock_Ha.real
    off = f - np.diag(np.diag(f))
    fterms = tuple(
        (off[p, q], ((p, True), (q, False)))
        for p, q in itertools.product(range(n), repeat=2)
        if off[p, q] != 0
    )
    correction, _ = dense_linked(problem, amplitudes, fterms)
    expected = paper + correction
    actual = problem.linked_action(amplitudes)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-11)
    dense, _ = dense_linked(problem, amplitudes)
    np.testing.assert_allclose(actual, dense, rtol=0, atol=1e-11)
    np.testing.assert_allclose(
        actual,
        problem.source_Ha + problem.linear_map_Ha @ amplitudes,
        rtol=0,
        atol=1e-11,
    )
    # This control forces the omitted CI+all-order valence-singles diagram.
    if key == (2, -1):
        assert np.max(abs(amplitudes[: problem.plan["singles"]])) > 0.001
        _, without = paper_rhs(g, cs, cd, vs * 0, vd, problem.valence_index)
        assert np.max(abs(rd - without)) > 1e-5


@pytest.mark.parametrize(
    "key,m", [((2, -1), -1), ((2, -1), 1), ((2, 1), -1), ((2, -2), -3)]
)
def test_energy_feedback_wavefunction_and_independent_nonlinear_root(live_core, key, m):
    problem = problem_for(live_core, key, m)
    result = problem.solve()
    assert result.projected_residual_Ha < 1e-10
    assert result.energy_residual_Ha < 1e-10
    assert abs(result.correlation_energy_Ha) > 1e-5
    fixed = np.linalg.solve(problem.linear_map_Ha[1:], -problem.source_Ha[1:])
    assert (
        np.max(abs(problem.residual(fixed, result.correlation_energy_Ha)[1:])) > 1e-10
    )
    assert np.max(abs(result.amplitudes - fixed)) > 1e-10
    # An independent nonlinear root consumes full dense Fock-space actions.
    # Coefficients of that action are built once; root iteration never calls
    # the production pencil or its residual routine.
    constant, _ = dense_linked(problem, np.zeros(problem.plan["amplitudes"]))
    columns = []
    for i in range(problem.plan["amplitudes"]):
        unit = np.zeros(problem.plan["amplitudes"])
        unit[i] = 1
        column, _ = dense_linked(problem, unit)
        columns.append(column - constant)
    linear = np.column_stack(columns)

    def residual(x):
        values = constant + linear @ x[:-1]
        values[0] -= problem.orbital_energy_Ha + x[-1]
        values[1:] -= x[-1] * x[:-1]
        return values

    independent = root(residual, np.r_[fixed, 0.0])
    assert independent.success and np.max(abs(residual(independent.x))) < 1e-10
    np.testing.assert_allclose(
        result.amplitudes, independent.x[:-1], rtol=0, atol=1e-10
    )
    assert result.correlation_energy_Ha == pytest.approx(independent.x[-1], abs=1e-11)
    _, (masks, h, tc, sv) = dense_linked(problem, result.amplitudes)
    index = masks.index(problem.reference_mask)
    state = (np.eye(len(masks)) + tc + sv)[:, index]
    assert result.wavefunction_norm == pytest.approx(state @ state, abs=1e-12)
    state /= np.linalg.norm(state)
    actual = np.array(
        [
            dict(zip(result.determinants, result.coefficients, strict=True)).get(
                mask, 0
            )
            for mask in masks
        ]
    )
    np.testing.assert_allclose(actual, state, rtol=0, atol=1e-11)
    assert result.full_eigenstate_residual_Ha == pytest.approx(
        np.max(abs(h @ state - result.total_energy_Ha * state)), abs=1e-11
    )
    # Truncated SD state and linked energy are not certified exact CI.
    assert result.full_eigenstate_residual_Ha > 1e-5
    assert result.expectation_energy_Ha == pytest.approx(state @ h @ state, abs=1e-11)
    assert len(result.continuation_path) == 8
    for array in (result.amplitudes, result.coefficients, problem.linear_map_Ha):
        with pytest.raises(ValueError):
            array.flags.writeable = True
    with pytest.raises(TypeError):
        result.provenance["method"] = "FCI"


def test_normalized_one_body_signed_phase_and_actual_parent(live_core):
    left = problem_for(live_core, (2, 1)).solve()
    right = problem_for(live_core).solve()
    n = len(right._owner.integrals.labels)
    operator = np.zeros((n, n), dtype=complex)
    operator[left.valence_index, right.valence_index] = 0.7 + 0.2j
    actual = one_body_matrix_element(left, right, operator)
    terms = tuple(
        (operator[p, q], ((p, True), (q, False)))
        for p, q in itertools.product(range(n), repeat=2)
        if operator[p, q] != 0
    )
    # Separate complex elementary operation; dense oracle helper is real only.
    ket = dict(zip(right.determinants, right.coefficients, strict=True))
    bra = dict(zip(left.determinants, left.coefficients, strict=True))
    expected = 0j
    for mask, value in ket.items():
        for coefficient, operations in terms:
            target, sign = elementary(mask, operations)
            if sign:
                expected += bra.get(target, 0) * sign * coefficient * value
    assert actual == pytest.approx(expected, abs=1e-12)
    assert actual.imag > 0.1
    assert one_body_matrix_element(right, left, operator.conj().T) == pytest.approx(
        actual.conjugate(), abs=1e-12
    )
    assert one_body_matrix_element(right, right, np.eye(n)) == pytest.approx(
        3.0, abs=1e-12
    )
    weak = np.zeros((n, n))
    weak[left.valence_index, right.valence_index] = 1e-16
    assert one_body_matrix_element(left, right, weak) != 0
    with pytest.raises(ValueError, match="actual owner"):
        one_body_matrix_element(replace(left), right, operator)
    with pytest.raises(ValueError, match="spinor one-body"):
        one_body_matrix_element(left, right, operator[:-1])


def test_core_identity_and_root_gates(live_core):
    core, result = live_core
    v = core.integrals.labels.index((2, -1, -1))
    with pytest.raises(ValueError, match="actual complete core"):
        FiniteMonovalentSD(
            core,
            replace(result, amplitudes=result.amplitudes + 0.1),
            v,
            work_cap=200_000_000,
        )
    with pytest.raises(ValueError, match="residual/energy"):
        FiniteMonovalentSD(
            core,
            replace(result, total_energy_Ha=result.total_energy_Ha + 0.1),
            v,
            work_cap=200_000_000,
        )
    problem = problem_for(live_core)
    for options in (
        dict(continuation=(0.5, 0.25, 1.0)),
        dict(residual_tolerance_Ha=True),
        dict(minimum_overlap=1.0),
    ):
        with pytest.raises(ValueError):
            problem.solve(**options)
    with pytest.raises(ValueError, match="degenerate"):
        problem.solve(root_separation_Ha=100.0)
    with pytest.raises(ValueError, match="complete real valence"):
        problem.linked_action(np.ones(problem.plan["amplitudes"] - 1))
    with (
        patch.object(core, "integrals", object()),
        pytest.raises(ValueError, match="binding changed"),
    ):
        problem.solve()
