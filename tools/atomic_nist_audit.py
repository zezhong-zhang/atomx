"""Audit resolved, caller-owned spectral states against a local NIST snapshot.

This adapter reuses the existing ASD record and air/vacuum conventions. It
retains missing/ambiguous states, distinguishes observed from Ritz/level-derived
wavelengths, and returns whole-state iteration requests without dispatching jobs.
No solver, fitting, experimental uncertainty invention, or archive I/O occurs.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from atomx.constants import HARTREE_CM1
from atomx.reference_data.matching import observed_vacuum_nm
from atomx.reference_data.nist import normalise_configuration
from tools.export_nist_atlas import level_record

IDENTITY_FIELDS = (
    "hamiltonian",
    "nucleus",
    "core",
    "space",
    "energy_zero",
    "units",
    "normalization",
    "approximation_scope",
)


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class SpectralState:
    """Resolved nominal branch, not a configuration-weight certificate."""

    key: tuple[int, int]
    z: int
    charge: int
    configuration: str
    two_j: int
    parity: int
    energy_Ha: float
    physical_identity: Mapping

    def __post_init__(self):
        if (
            isinstance(self.energy_Ha, bool)
            or type(self.two_j) is not int
            or type(self.parity) is not int
            or not math.isfinite(self.energy_Ha)
            or self.two_j < 0
            or not isinstance(self.two_j, int)
            or self.parity not in (-1, 1)
        ):
            raise ValueError(
                "resolved finite energy, integer 2J and signed parity required"
            )
        if (
            type(self.z) is not int
            or type(self.charge) is not int
            or not (0 < self.z <= 118 and 0 <= self.charge < self.z)
        ):
            raise ValueError("atomic species required")
        if any(name not in self.physical_identity for name in IDENTITY_FIELDS):
            raise ValueError("complete physical identity required")
        if any(self.physical_identity[name] is None for name in IDENTITY_FIELDS):
            raise ValueError("unknown physical identity cannot be silently inferred")
        if self.physical_identity["units"] != "Ha":
            raise ValueError("explicit Hartree state energy required")
        # Own every nested value, then prevent binding mutation by consumers.
        owned = json.loads(json.dumps(_plain(self.physical_identity), allow_nan=False))
        object.__setattr__(self, "physical_identity", _freeze(owned))


def selection_rule(lower, upper, multipole):
    """Relativistic J/parity necessity; this never establishes a nonzero rate."""
    if multipole not in ("E1", "E2", "M1", "M2"):
        raise ValueError("supported multipole must be explicit")
    rank = int(multipole[1])
    angular = abs(upper.two_j - lower.two_j) <= 2 * rank <= upper.two_j + lower.two_j
    required_parity = (-1) ** (rank + (multipole[0] == "M"))
    return dict(
        multipole=multipole,
        angular_allowed=angular,
        parity_allowed=lower.parity * upper.parity == required_parity,
        symmetry_allowed=angular and lower.parity * upper.parity == required_parity,
        scope="necessary J/parity conditions only; no matrix element/LS purity/rate claim",
    )


def interval_standard_uncertainty(lower, upper, covariance=None):
    """Propagate stated standard uncertainties only with known covariance.

    A known exact zero anchor needs no nonzero covariance. Missing uncertainties
    are not print precision and are never replaced by zero or residual bounds.
    """
    if lower is None or upper is None:
        return None
    if any(not math.isfinite(v) or v < 0 for v in (lower, upper)):
        raise ValueError("finite nonnegative standard uncertainty required")
    if covariance is None:
        if lower and upper:
            return None
        covariance = 0.0
    if not math.isfinite(covariance):
        raise ValueError("finite covariance required")
    if (lower == 0 or upper == 0) and covariance != 0:
        raise ValueError("covariance violates the Cauchy bound")
    scale = max(lower, upper)
    if scale == 0:
        return 0.0
    # Scale before products to preserve tiny uncertainties and avoid overflow.
    lo, hi = lower / scale, upper / scale
    scaled_covariance = covariance / scale / scale
    bound = lo * hi
    slack = 8 * math.ulp(bound) if bound > 0 else 0.0
    if not math.isfinite(scaled_covariance) or abs(scaled_covariance) > bound + slack:
        raise ValueError("covariance violates the Cauchy bound")
    value = scale * math.sqrt(max(0.0, lo * lo + hi * hi - 2 * scaled_covariance))
    if not math.isfinite(value):
        raise ValueError("propagated uncertainty exceeds finite floating range")
    return value


def _reference_candidates(state, levels):
    return [
        r
        for r in levels
        if r["z"] == state.z
        and r["charge"] == state.charge
        and normalise_configuration(r["configuration"] or "")
        == normalise_configuration(state.configuration)
        and level_record(r).two_j == state.two_j
        and level_record(r).parity == state.parity
        and r["energy_kind"] == "evaluated"
        and r["energy_cm1"] is not None
        and math.isfinite(r["energy_cm1"])
        and not r["bracketed"]
        and not level_record(r).flags
        and "?" not in (r["configuration_nist"] or "") + (r["term"] or "")
    ]


def _measurement(calculated, reference, goal):
    if reference is None:
        return dict(error=None, relative_error=None, goal_pass=None)
    error = calculated - reference
    relative = error / reference if reference else None
    return dict(
        error=error,
        relative_error=relative,
        goal_pass=abs(relative) <= goal if relative is not None else None,
    )


def _summary(rows, required):
    matched = [r for r in rows if r["error"] is not None]
    return dict(
        required=required,
        scored=len(matched),
        missing_or_ambiguous=required - len(matched),
        passing=sum(r["goal_pass"] is True for r in matched),
        complete_and_all_pass=len(matched) == required
        and all(r["goal_pass"] is True for r in matched),
        max_absolute_error=max((abs(r["error"]) for r in matched), default=None),
        mean_absolute_error=sum(abs(r["error"]) for r in matched) / len(matched)
        if matched
        else None,
        rms_error=math.sqrt(sum(r["error"] ** 2 for r in matched) / len(matched))
        if matched
        else None,
        max_absolute_relative_error=max(
            (
                abs(r["relative_error"])
                for r in matched
                if r["relative_error"] is not None
            ),
            default=None,
        ),
    )


def audit_spectrum(
    states,
    levels,
    lines,
    *,
    zero_key,
    required_keys,
    splitting_pairs,
    transition_targets,
    relative_goal=0.01,
    reference_covariances=None,
):
    """Return strict nominal state/line coverage and a reusable whole-state queue.

    Required state keys and observable pairs are predeclared by the consumer.
    Mixed-Hamiltonian or incomplete theoretical panels are rejected. Unmatched
    references stay in denominators; each distinct observed line is independent
    of level-derived intervals, while Ritz lines are not additional observations.
    """
    if not math.isfinite(relative_goal) or relative_goal <= 0:
        raise ValueError("predeclared positive accuracy goal required")
    by_key = {s.key: s for s in states}
    if (
        len(by_key) != len(states)
        or len(set(required_keys)) != len(required_keys)
        or set(by_key) != set(required_keys)
    ):
        raise ValueError(
            "complete, unique predeclared theoretical state panel required"
        )
    zero = by_key[zero_key]
    identity = _plain(zero.physical_identity)
    for state in states:
        if state.physical_identity != zero.physical_identity or (
            state.z,
            state.charge,
        ) != (
            zero.z,
            zero.charge,
        ):
            raise ValueError(
                "all states must share Hamiltonian/core/space/nucleus/metric/zero"
            )
    matches = {s.key: _reference_candidates(s, levels) for s in states}
    if len(matches[zero_key]) != 1 or matches[zero_key][0]["energy_cm1"] != 0:
        raise ValueError("unique evaluated zero state required")
    ref_zero = matches[zero_key][0]
    matched = {key: rs[0] if len(rs) == 1 else None for key, rs in matches.items()}
    covariance = reference_covariances or {}
    excitation = []
    for state in states:
        reference = matched[state.key]
        calculated = (state.energy_Ha - zero.energy_Ha) * HARTREE_CM1
        ground = state.key == zero_key
        value = _measurement(
            calculated,
            reference["energy_cm1"] if reference and not ground else None,
            relative_goal,
        )
        value.update(
            key=state.key,
            calculated_cm1=calculated,
            reference_cm1=reference["energy_cm1"] if reference else None,
            matching="ground_reference"
            if ground
            else "unique_nominal_branch"
            if reference
            else "ambiguous_or_missing",
            candidates=len(matches[state.key]),
            level_id=reference["level_id"] if reference else None,
            reference_code=reference["reference"] if reference else None,
            reference_standard_uncertainty_cm1=interval_standard_uncertainty(
                ref_zero["uncertainty_cm1"],
                reference["uncertainty_cm1"],
                covariance.get((zero_key, state.key)),
            )
            if reference and not ground
            else None,
            standardized_residual=None,
            uncertainty_status="theory uncertainty absent; missing NIST uncertainty/covariance never replaced by digits/residuals",
        )
        excitation.append(value)
    intervals = []
    for lower_key, upper_key in splitting_pairs:
        lower, upper = by_key[lower_key], by_key[upper_key]
        lr, ur = matched[lower_key], matched[upper_key]
        calculated = (upper.energy_Ha - lower.energy_Ha) * HARTREE_CM1
        reference = ur["energy_cm1"] - lr["energy_cm1"] if lr and ur else None
        intervals.append(
            dict(
                lower_key=lower_key,
                upper_key=upper_key,
                calculated_cm1=calculated,
                reference_cm1=reference,
                reference_standard_uncertainty_cm1=interval_standard_uncertainty(
                    lr["uncertainty_cm1"],
                    ur["uncertainty_cm1"],
                    covariance.get((lower_key, upper_key)),
                )
                if lr and ur
                else None,
                standardized_residual=None,
                **_measurement(calculated, reference, relative_goal),
            )
        )
    transitions = []
    for lower_key, upper_key, multipole in transition_targets:
        lower, upper = by_key[lower_key], by_key[upper_key]
        rule = selection_rule(lower, upper, multipole)
        lr, ur = matched[lower_key], matched[upper_key]
        calculated_cm1 = (upper.energy_Ha - lower.energy_Ha) * HARTREE_CM1
        calculated_nm = 1e7 / calculated_cm1 if calculated_cm1 > 0 else None
        reference_cm1 = ur["energy_cm1"] - lr["energy_cm1"] if lr and ur else None
        reference_nm = (
            1e7 / reference_cm1 if reference_cm1 and reference_cm1 > 0 else None
        )
        candidate_lines = [
            r
            for r in lines
            if lr
            and ur
            and r["z"] == lower.z
            and r["charge"] == lower.charge
            and r["lower_level_id"] == lr["level_id"]
            and r["upper_level_id"] == ur["level_id"]
        ]
        # Referenced level IDs are cross-checked against labels to catch stale IDs.
        for line in candidate_lines:
            for prefix, ref in (("lower", lr), ("upper", ur)):
                if (
                    normalise_configuration(line[prefix + "_configuration"] or "")
                    != normalise_configuration(ref["configuration"] or "")
                    or line[prefix + "_j"] != ref["j_text"]
                    or line[prefix + "_term"] != ref["term"]
                ):
                    raise ValueError(
                        "line endpoint IDs and resolved state labels disagree"
                    )
        observed = [r for r in candidate_lines if r["observed_nm"] is not None]
        observation = observed[0] if len(observed) == 1 else None
        vacuum = None
        if observation:
            if (
                not math.isfinite(observation["observed_nm"])
                or observation["observed_nm"] <= 0
            ):
                raise ValueError("finite positive observed wavelength required")
            vacuum = observed_vacuum_nm(
                observation["observed_nm"], observation["medium"]
            )
        measure = (
            _measurement(calculated_nm, vacuum, relative_goal)
            if calculated_nm is not None
            else _measurement(0, None, relative_goal)
        )
        if not rule["symmetry_allowed"] or calculated_cm1 <= 0:
            measure["goal_pass"] = False
        transitions.append(
            dict(
                lower_key=lower_key,
                upper_key=upper_key,
                selection=rule,
                calculated_energy_derived_vacuum_nm=calculated_nm,
                reference_level_derived_vacuum_nm=reference_nm,
                observed_candidates=len(observed),
                candidate_lines=len(candidate_lines),
                observed_vacuum_nm=vacuum,
                observed_native_nm=observation["observed_nm"] if observation else None,
                observed_native_medium=observation["medium"] if observation else None,
                observed_native_uncertainty_nm=observation["observed_unc_nm"]
                if observation
                else None,
                observed_standard_uncertainty_vacuum_nm=observation["observed_unc_nm"]
                if observation and observation["medium"] == "vacuum"
                else None,
                source_row=observation["source_row"] if observation else None,
                line_reference=observation["line_reference"] if observation else None,
                source_transition_type=observation["transition_type"]
                if observation
                else None,
                matching="unique_observed_line"
                if vacuum is not None
                else "unknown_medium"
                if observation
                else "ambiguous_or_missing_observed_line",
                ritz_candidates=sum(r["ritz_nm"] is not None for r in candidate_lines),
                ritz_is_independent_observation=False,
                standardized_residual=None,
                rate_or_intensity_computed=False,
                **measure,
            )
        )
    failed = sorted(
        [
            dict(
                observable="excitation",
                key=r["key"],
                relative_error=r["relative_error"],
            )
            for r in excitation
            if r["goal_pass"] is False
        ]
        + [
            dict(
                observable="splitting",
                keys=[r["lower_key"], r["upper_key"]],
                relative_error=r["relative_error"],
            )
            for r in intervals
            if r["goal_pass"] is False
        ],
        key=lambda r: abs(r["relative_error"]),
        reverse=True,
    )
    identity_digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return dict(
        status="complete" if all(matched.values()) else "incomplete_reference_coverage",
        species=dict(z=zero.z, charge=zero.charge),
        physical_identity=identity,
        physical_identity_sha256=identity_digest,
        required_keys=list(required_keys),
        zero_key=zero_key,
        relative_goal=relative_goal,
        constants=dict(hartree_cm1=HARTREE_CM1),
        excitation=excitation,
        splittings=intervals,
        transitions=transitions,
        coverage=dict(
            required_states=len(states),
            unique_nominal_states=sum(r is not None for r in matched.values()),
            snapshot_levels=len(levels),
            snapshot_lines=len(lines),
        ),
        metrics=dict(
            excitation_cm1=_summary(
                [r for r in excitation if r["key"] != zero_key], len(states) - 1
            ),
            splitting_cm1=_summary(intervals, len(splitting_pairs)),
            observed_wavelength_nm=_summary(transitions, len(transition_targets)),
        ),
        iteration_requests=[
            dict(
                task="diagnose_complete_same_space_panel",
                priority_targets=failed,
                all_required_state_keys=list(required_keys),
                physical_identity_sha256=identity_digest,
                hypotheses=[
                    "complete common core/valence virtual-space convergence",
                    "nonlinear and triple correlation",
                    "common Breit/QED/recoil operator coverage",
                    "nominal state mixing/character",
                ],
                diagnostic_causes_proven=False,
                accuracy_goal_unchanged=relative_goal,
                admission="requires prospective complete-space source/resource contract; no solver dispatched",
            )
        ]
        if failed
        else [],
        scope="unique nominal branches, same-model full panel; configuration weights unverified, no new roots/rates/experimental uncertainty/accuracy promotion",
        highest_level_reached=False,
    )
