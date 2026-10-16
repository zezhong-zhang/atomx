"""Five-UID gradients against forward exact-CSF radial directional derivatives.

The oracle differentiates raw ket-normalized one-body integrals in two slots
and Coulomb integrals in four slots, then assembles dH in the exact CSF basis.
It does not call the production radial-gradient or pair-potential routines.
"""

import math
from fractions import Fraction
from functools import lru_cache

import numpy as np
import pytest

from atomx.angular import csf_hamiltonian as exact
from atomx.config.csf import enumerate_jj_csfs
from atomx.methods.csf.variational import CSFOrbitalFunctional
from atomx.numerics.bspline.basis import prepare_bspline_data
from atomx.numerics.bspline.convol import compute_slater_rk_cell
from atomx.numerics.bspline.dirac_assembly import build_dirac_matrices, solve_dirac
from atomx.numerics.bspline.knots import make_knots
from atomx.numerics.bspline.orbital_optimization import _polar_retraction
from atomx.physics.csf.levels import ci_level_blocks
from atomx.physics.csf.state_average import CIStateAverage

UIDS = ((1, 1, 0), (3, 1, 0), (5, 1, 0), (1, -1, 0), (3, -1, 0))
SCALES = (2.0, 0.5, -1.7, 3.1, -0.3)


def _normalized_weights(functional, vectors):
    """Full actual CI-coefficient norms; exact zeros only, signed coefficients."""
    terms = {}
    for vector in vectors:
        norm = sum((Fraction.from_float(float(v)) ** 2 for v in vector), Fraction())
        assert norm > 0
        for key, value in functional.radial_density.contract_bilinear(
            vector, vector
        ).items():
            terms.setdefault(key, []).append(
                Fraction.from_float(0.2) * Fraction.from_float(float(value)) / norm
            )
    return {
        key: float(value)
        for key, parts in terms.items()
        if (value := sum(parts, Fraction())) != 0
    }


def _measure():
    ns, charge = 35, 4
    configs = ("1s2 2s1", "1s2 3s1", "1s2 3d1", "1s2 2p1", "2s2 2p1")
    csfs = [csf for config in configs for csf in enumerate_jj_csfs(config)]
    data = prepare_bspline_data(
        make_knots(ns, 9, 0.005, 30), ns, 9, kinetic_balance=True
    )
    target = CIStateAverage(tuple((*uid, 1) for uid in UIDS))
    functional = CSFOrbitalFunctional(csfs, data, charge, state_average=target)
    waves, projectors, raw = {}, {}, {}
    for kappa in functional.metric:
        raw[kappa] = build_dirac_matrices(ns, kappa, charge, data)
        spectrum = solve_dirac(*raw[kappa], ns, kappa, nsp=data["nsp"], nsq=data["nsq"])
        projectors[kappa] = spectrum.coefficients
        ell = kappa if kappa > 0 else -kappa - 1
        for n, key_kappa in functional.keys:
            if key_kappa == kappa:
                p, q, _ = spectrum.get_orbital(n - ell - 1)
                waves[n, kappa] = p, q
    _, native_gradient, ci = functional.evaluate(waves)
    _, blocks = ci_level_blocks(csfs, ci, require_full=True)
    direction, rng = {}, np.random.default_rng(314159)
    for kappa, projector in projectors.items():
        keys = [key for key in functional.keys if key[1] == kappa]
        x = np.column_stack([np.concatenate(waves[key]) for key in keys])
        metric = functional.metric[kappa]
        d = projector @ rng.normal(size=(projector.shape[1], len(keys)))
        cross = x.T @ metric @ d
        d -= x @ (0.5 * (cross + cross.T))
        d /= np.sqrt(np.trace(d.T @ metric @ d))
        assert np.max(abs(x.T @ metric @ d + d.T @ metric @ x)) < 1e-12
        for i, key in enumerate(keys):
            direction[key] = d[:ns, i], d[ns:, i]

    @lru_cache(None)
    def radial_derivative(term):
        if term[0] == "h":
            _, na, kappa, nb = term
            a, b = (na, kappa), (nb, kappa)
            va, vb = np.concatenate(waves[a]), np.concatenate(waves[b])
            da, db = np.concatenate(direction[a]), np.concatenate(direction[b])
            h, metric = raw[kappa]
            norm, numerator = vb @ metric @ vb, va @ h @ vb
            return float(
                (da @ h @ vb + va @ h @ db) / norm
                - numerator * (db @ metric @ vb + vb @ metric @ db) / norm**2
            )
        _, rank, *keys = term
        # Direct multilinearity, independently of the gradient's source-pair
        # actions and accumulation. Every slot and interference term stays.
        values = []
        for slot in range(4):
            args = [
                waves[key] if i != slot else direction[key]
                for i, key in enumerate(keys)
            ]
            values.append(
                float(
                    compute_slater_rk_cell(
                        *args[0],
                        *args[1],
                        *args[2],
                        *args[3],
                        data,
                        rank,
                        cache=functional.cache,
                    )
                )
            )
        return math.fsum(values)

    def contraction(gradient):
        return math.fsum(
            float(gradient[key] @ np.concatenate(direction[key]))
            for key in functional.keys
        )

    vectors, rows, signed_data = [], [], []
    for uid, scale in zip(UIDS, SCALES, strict=True):
        column = int(blocks[uid[:2]][uid[2]])
        vector = ci.mixing_coefficients[:, column] * scale
        vectors.append(vector)
        norm = sum((Fraction.from_float(float(v)) ** 2 for v in vector), Fraction())
        indices = [
            i for i, csf in enumerate(csfs) if (csf.total_J2, csf.parity) == uid[:2]
        ]
        signatures = [exact.csf_signature(csfs[i]) for i in indices]
        block_vector = vector[indices]
        row = dict(
            UID=uid,
            CIcolumn=column,
            scale=scale,
            actualnorm=float(norm),
            forward={},
            gradient={},
        )
        # The helper's fixed 0.2 is the declared five-state EOL weight, not a
        # normalization of this individual state's population.
        weights = _normalized_weights(functional, [vector])
        for label, kind in (("onebody", "h"), ("twobody", "r")):
            derivative = exact.exact_block_matrix(
                signatures,
                lambda term, kind=kind: (
                    radial_derivative(term) if term[0] == kind else 0.0
                ),
            )
            row["forward"][label] = float(
                0.2 * (block_vector @ derivative @ block_vector) / float(norm)
            )
            gradient = functional.gradient_from_weights(
                waves,
                {key: v for key, v in weights.items() if key[0] == kind},
                record_diagnostics=False,
            )
            row["gradient"][label] = contraction(gradient)
        rows.append(row)
        signed_data.append((signatures, block_vector, norm))
    weights = _normalized_weights(functional, vectors)
    full_gradient = functional.gradient_from_weights(
        waves, weights, record_diagnostics=False
    )
    unscaled = [ci.mixing_coefficients[:, int(blocks[uid[:2]][uid[2]])] for uid in UIDS]
    baseline = functional.gradient_from_weights(
        waves, _normalized_weights(functional, unscaled), record_diagnostics=False
    )
    scaling_error = max(
        float(abs(full_gradient[key] - baseline[key]).max()) for key in functional.keys
    )
    forward = {
        label: math.fsum(row["forward"][label] for row in rows)
        for label in ("onebody", "twobody")
    }
    exact_total = math.fsum(forward.values())
    step, energies = 2e-5, []
    for sign in (1, -1):
        trial = {}
        for kappa, projector in projectors.items():
            keys = [key for key in functional.keys if key[1] == kappa]
            metric = functional.metric[kappa]
            x = np.column_stack([np.concatenate(waves[key]) for key in keys])
            d = np.column_stack([np.concatenate(direction[key]) for key in keys])
            y = projector @ _polar_retraction(
                projector.T @ metric @ x + sign * step * (projector.T @ metric @ d)
            )
            for i, key in enumerate(keys):
                trial[key] = y[:ns, i], y[ns:, i]
        energies.append(functional.evaluate(trial, gradient=False)[0])
    numerical = (energies[0] - energies[1]) / (2 * step)
    unsigned = 0.0
    for signatures, vector, norm in signed_data:
        wrong = abs(vector)
        derivative = exact.exact_block_matrix(signatures, radial_derivative)
        unsigned += float(0.2 * (wrong @ derivative @ wrong) / float(norm))
    return dict(
        scope="Z4/3electron/ns35 RKB/8CSF/7wave single point; no Yb or optimization",
        UIDs=UIDS,
        csf_count=len(csfs),
        radial_wave_count=len(waves),
        nsp=data["nsp"],
        nsq=data["nsq"],
        state_rows=rows,
        forward_components=forward,
        forward_total=exact_total,
        native_total=contraction(native_gradient),
        fullnorm_total=contraction(full_gradient),
        fullnorm_signed_scaling_max_gradient_error=scaling_error,
        FD_step=step,
        central_FD=numerical,
        FD_minus_forward=numerical - exact_total,
        unsigned_negative_control_error=unsigned - exact_total,
        radial_derivative_keys=radial_derivative.cache_info().currsize,
    )


@pytest.fixture(scope="module")
def measurements():
    return _measure()


def test_each_UID_one_and_two_body_derivatives_match_forward_exact_CSF(measurements):
    assert tuple(measurements["UIDs"]) == UIDS
    assert measurements["csf_count"] == 8 and measurements["radial_wave_count"] == 7
    assert (measurements["nsp"], measurements["nsq"]) == (34, 35)
    for row in measurements["state_rows"]:
        for component in ("onebody", "twobody"):
            assert row["gradient"][component] == pytest.approx(
                row["forward"][component], rel=0, abs=3e-10
            )


def test_full_actualnorm_retains_signed_rescaling_invariance(measurements):
    assert len({row["actualnorm"] for row in measurements["state_rows"]}) == 5
    assert measurements["fullnorm_signed_scaling_max_gradient_error"] < 3e-12
    assert measurements["fullnorm_total"] == pytest.approx(
        measurements["forward_total"], rel=0, abs=3e-10
    )
    assert measurements["native_total"] == pytest.approx(
        measurements["forward_total"], rel=0, abs=3e-10
    )


def test_five_level_derivative_matches_rediagonalized_polar_step(measurements):
    assert measurements["central_FD"] == pytest.approx(
        measurements["forward_total"], rel=0, abs=2e-7
    )


def test_absolute_amplitudes_fail_the_signed_interference_control(measurements):
    assert abs(measurements["unsigned_negative_control_error"]) > 1e-6
