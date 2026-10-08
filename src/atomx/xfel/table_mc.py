"""Precompiled table-driven XFEL Monte Carlo with the original RNG stream.

This accelerates the existing bin-wise, instantaneous-Auger model. It
does not change atomic rates, time-bin probabilities, or decay lifetimes.
Incomplete tables fail by default; legacy freezing must be explicit.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numba import njit

from atomx.xfel.cascade import MCResult, Pulse


def _occupation_key(config):
    out = []
    for (n, k), q in config.items():
        if (
            not isinstance(n, Integral)
            or not isinstance(k, Integral)
            or n < 1
            or k == 0
            or (k if k > 0 else -k - 1) >= n
            or not np.isfinite(q)
            or q != int(q)
            or q < 0
            or q > 2 * abs(k)
        ):
            raise ValueError("MC configurations need legal integer jj occupations")
        if q:
            out.append((int(n), int(k), int(q)))
    return tuple(sorted(out))


def _parse_shell(text):
    try:
        n, k = text.split(",")
        return int(n), int(k)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"invalid subshell key {text!r}") from exc


def _daughter(key, changes):
    config = {(n, k): q for n, k, q in key}
    for shell, delta in changes:
        config[shell] = config.get(shell, 0) + delta
    return _occupation_key(config)


@dataclass
class TableMCResult(MCResult):
    """MC observables plus table identity and explicit missing-state audit."""

    table_sha256: str = ""
    missing_configurations: tuple = ()
    missing_policy: str = "raise"


@njit(cache=True)
def _walk(
    uniforms,
    n_trajectories,
    n_bins,
    max_charge,
    base_index,
    charges,
    pi_offsets,
    pi_daughters,
    pi_cdf,
    auger_offsets,
    auger_daughters,
    auger_cdf,
):
    counts = np.zeros(len(charges), dtype=np.int64)
    first_seen = np.full(len(charges), -1, dtype=np.int64)
    n_seen = 0
    visited = np.zeros(len(charges), dtype=np.bool_)
    encountered = np.zeros(len(charges), dtype=np.bool_)
    visited[base_index] = True
    pos = absorbed = emitted = 0
    for _ in range(n_trajectories):
        state = base_index
        encountered[state] = True
        for _bin in range(n_bins):
            visited[state] = True
            if charges[state] >= max_charge:
                break
            start, end = pi_offsets[state], pi_offsets[state + 1]
            total = pi_cdf[end - 1] if end > start else 0.0
            if total <= 1e-12:
                break
            u = uniforms[pos]
            pos += 1
            if u > min(total, 1.0):
                continue
            pick = uniforms[pos] * total
            pos += 1
            edge = end - 1
            for i in range(start, end):
                if pick <= pi_cdf[i]:
                    edge = i
                    break
            state = pi_daughters[edge]
            encountered[state] = True
            absorbed += 1
            emitted += 1
            # Same 24-event safety bound as the reference Auger walk.
            for _decay in range(24):
                start, end = auger_offsets[state], auger_offsets[state + 1]
                if start == end:
                    break
                total = auger_cdf[end - 1]
                pick = uniforms[pos] * total
                pos += 1
                edge = end - 1
                for i in range(start, end):
                    if pick <= auger_cdf[i]:
                        edge = i
                        break
                state = auger_daughters[edge]
                emitted += 1
                visited[state] = True
                encountered[state] = True
        if counts[state] == 0:
            first_seen[n_seen] = state
            n_seen += 1
        counts[state] += 1
    return counts, first_seen, visited, encountered, absorbed, emitted, pos


class CompiledRateTable:
    """Immutable rate arrays and occupation-resolved transition indices.

    Construct once and call ``run`` across a fluence scan. ``on_missing``
    defaults to ``raise``: any positive-rate edge lacking its daughter is
    an error. ``freeze`` reproduces the legacy provider's stopping rule,
    reports reached missing states, and is only a diagnostic compatibility
    mode. The bare ion is a known terminal and requires no table row.
    """

    def __init__(self, table, base_config, *, on_missing="raise"):
        if on_missing not in {"raise", "freeze"}:
            raise ValueError("on_missing must be 'raise' or 'freeze'")
        self.on_missing = on_missing
        self.photon_energy_eV = float(table["photon_energy_eV"])
        if not np.isfinite(self.photon_energy_eV) or self.photon_energy_eV <= 0:
            raise ValueError("table photon energy must be positive and finite")
        base = _occupation_key(base_config)
        self.base_nelec = sum(q for _, _, q in base)
        if table["base_nelec"] != self.base_nelec:
            raise ValueError("base configuration electron count differs from table")
        self.table_sha256 = hashlib.sha256(
            # Channel order determines the seeded reference path. Preserve
            # dictionary order in the identity as well as in the sampler.
            json.dumps(table, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        rows = {}
        for row in table["configs"].values():
            config = {_parse_shell(k): q for k, q in row["occupations"].items()}
            key = _occupation_key(config)
            if key in rows:
                raise ValueError("duplicate occupation-resolved table row")
            if sum(q for _, _, q in key) > self.base_nelec:
                raise ValueError("table state has more electrons than initial state")
            pi = []
            for text, value in row["sigma_cm2"].items():
                sigma = float(value)
                shell = _parse_shell(text)
                if not np.isfinite(sigma) or sigma < 0:
                    raise ValueError("PI cross sections must be finite and nonnegative")
                if sigma:
                    pi.append((_daughter(key, [(shell, -1)]), sigma))
            auger = []
            for ch in row["auger"]:
                rate = float(ch["rate"])
                if not np.isfinite(rate) or rate < 0:
                    raise ValueError("Auger rates must be finite and nonnegative")
                if rate:
                    daughter = _daughter(
                        key,
                        [
                            (tuple(ch["hole"]), 1),
                            (tuple(ch["s1"]), -1),
                            (tuple(ch["s2"]), -1),
                        ],
                    )
                    auger.append((daughter, rate))
            rows[key] = pi, auger
        missing = {
            d
            for pi, auger in rows.values()
            for d, _ in pi + auger
            if d and d not in rows
        }
        if base and base not in rows:
            missing.add(base)
        if missing and on_missing == "raise":
            sample = sorted(missing)[0]
            raise ValueError(
                f"incomplete rate table: {len(missing)} missing configurations; first {sample}"
            )
        for key in sorted(missing | {()}):
            rows.setdefault(key, ([], []))
        self.keys = tuple(rows)
        self.missing_keys = frozenset(missing)
        indices = {key: i for i, key in enumerate(self.keys)}
        self.base_index = indices[base]
        self.charges = np.array(
            [self.base_nelec - sum(q for _, _, q in key) for key in self.keys],
            dtype=np.int64,
        )
        self.pi_offsets, self.pi_daughters, self.sigmas = self._arrays(rows, indices, 0)
        self.auger_offsets, self.auger_daughters, rates = self._arrays(rows, indices, 1)
        self.auger_cdf = self._cdf(rates, self.auger_offsets)
        for values in (
            self.charges,
            self.pi_offsets,
            self.pi_daughters,
            self.sigmas,
            self.auger_offsets,
            self.auger_daughters,
            self.auger_cdf,
        ):
            values.setflags(write=False)

    @staticmethod
    def _arrays(rows, indices, kind):
        offsets, daughters, values = [0], [], []
        for edges in rows.values():
            for daughter, value in edges[kind]:
                daughters.append(indices[daughter])
                values.append(value)
            offsets.append(len(values))
        return (
            np.array(offsets, dtype=np.int64),
            np.array(daughters, dtype=np.int64),
            np.array(values, dtype=float),
        )

    @staticmethod
    def _cdf(values, offsets):
        out = np.empty_like(values)
        for start, end in zip(offsets[:-1], offsets[1:], strict=True):
            total = 0.0
            for i in range(start, end):
                total += float(values[i])
                out[i] = total
            if not np.isfinite(total):
                raise ValueError("total channel rate/probability overflowed")
        return out

    def run(
        self,
        pulse: Pulse,
        *,
        n_trajectories=2000,
        seed=7,
        max_charge=12,
        n_bins=8,
        batch_size=10000,
    ):
        """Same bin-wise law and Python Random stream as ``cascade_mc``.

        Random buffers are bounded by trajectory batches. Cold Numba
        compilation is paid once; no rate/structure provider runs here.
        """
        for name, value in (
            ("n_trajectories", n_trajectories),
            ("n_bins", n_bins),
            ("max_charge", max_charge),
            ("batch_size", batch_size),
        ):
            if not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if pulse.photon_energy_eV != self.photon_energy_eV:
            raise ValueError("pulse photon energy differs from rate table")
        fluence = float(pulse.fluence_photons_per_cm2)
        if not np.isfinite(fluence) or fluence < 0:
            raise ValueError("fluence must be finite and nonnegative")
        # Match the reference's operation order, not F * sum(sigma) / bins.
        probabilities = np.array([fluence * float(s) / n_bins for s in self.sigmas])
        cdf = self._cdf(probabilities, self.pi_offsets)
        rng = random.Random(seed)
        distribution = {}
        visited = np.zeros(len(self.keys), dtype=bool)
        encountered = np.zeros(len(self.keys), dtype=bool)
        absorbed = emitted = 0
        # Every PI/Auger event removes one electron: at most N PI draws
        # plus N Auger draws, in addition to one absorption draw per bin.
        draws_per_trajectory = n_bins + 2 * self.base_nelec
        memory_batch = max(1, 262144 // draws_per_trajectory)
        for start in range(0, n_trajectories, min(batch_size, memory_batch)):
            size = min(batch_size, memory_batch, n_trajectories - start)
            state = rng.getstate()
            uniforms = np.fromiter(
                (rng.random() for _ in range(size * draws_per_trajectory)), dtype=float
            )
            result = _walk(
                uniforms,
                size,
                n_bins,
                max_charge,
                self.base_index,
                self.charges,
                self.pi_offsets,
                self.pi_daughters,
                cdf,
                self.auger_offsets,
                self.auger_daughters,
                self.auger_cdf,
            )
            batch_counts, first_seen, batch_visited, batch_encountered, a, e, used = (
                result
            )
            if used < len(uniforms):
                # Discard no RNG draws between batches: restart at the
                # exact reference stream position after the consumed prefix.
                rng.setstate(state)
                for _ in range(used):
                    rng.random()
            for index in first_seen:
                if index < 0:
                    break
                charge = int(self.charges[index])
                distribution[charge] = distribution.get(charge, 0) + int(
                    batch_counts[index]
                )
            visited |= batch_visited
            encountered |= batch_encountered
            absorbed += a
            emitted += e
        # Integer count aggregation gives one division per charge.
        distribution = {q: n / n_trajectories for q, n in distribution.items()}
        return TableMCResult(
            n_trajectories=n_trajectories,
            charge_distribution=distribution,
            mean_charge=sum(q * p for q, p in distribution.items()),
            mean_absorbed=absorbed / n_trajectories,
            mean_emitted=emitted / n_trajectories,
            configs_visited=int(visited.sum()),
            table_sha256=self.table_sha256,
            missing_configurations=tuple(
                key
                for key, reached in zip(self.keys, encountered, strict=True)
                if reached and key in self.missing_keys
            ),
            missing_policy=self.on_missing,
        )
