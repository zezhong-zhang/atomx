"""Finite-lifetime configuration kinetics under explicit photon pulses.

Sparse column generators evolve population and event counters together.
No instantaneous decay, clipping of probabilities, or hidden terminal
states. Narrow-band cross sections must be tabulated at the pulse energy;
resonant pumping is added with its own bandwidth-averaged line strength.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.sparse import coo_matrix

from atomx.constants import AU_RATE_PER_S, HARTREE_EV
from atomx.xfel.cascade import _table_key
from atomx.xfel.table_mc import _daughter, _occupation_key, _parse_shell


@dataclass(frozen=True)
class GaussianPulse:
    photon_energy_eV: float
    fluence_photons_per_cm2: float
    fwhm_fs: float
    center_fs: float = 0.0
    bandwidth_fwhm_eV: float = 0.0

    def __post_init__(self):
        values = (
            self.photon_energy_eV,
            self.fluence_photons_per_cm2,
            self.fwhm_fs,
            self.center_fs,
            self.bandwidth_fwhm_eV,
        )
        if (
            not np.isfinite(values).all()
            or self.photon_energy_eV <= 0
            or self.fwhm_fs <= 0
        ):
            raise ValueError("finite positive pulse energy and duration required")
        if self.fluence_photons_per_cm2 < 0 or self.bandwidth_fwhm_eV < 0:
            raise ValueError("fluence and bandwidth must be nonnegative")

    @property
    def sigma_fs(self):
        return self.fwhm_fs / np.sqrt(8 * np.log(2))

    @property
    def support_fs(self):
        return self.center_fs - 8 * self.sigma_fs, self.center_fs + 8 * self.sigma_fs

    def flux(self, t_fs):
        return (
            self.fluence_photons_per_cm2
            / (self.sigma_fs * 1e-15 * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * ((np.asarray(t_fs) - self.center_fs) / self.sigma_fs) ** 2)
        )


class TabulatedPulse:
    """Piecewise-linear photon flux; zero outside the supplied time support."""

    def __init__(
        self,
        photon_energy_eV,
        times_fs,
        flux_photons_per_cm2_s,
        *,
        bandwidth_fwhm_eV=0.0,
    ):
        self.times_fs = np.array(times_fs, dtype=float, copy=True)
        self.values = np.array(flux_photons_per_cm2_s, dtype=float, copy=True)
        if (
            self.times_fs.ndim != 1
            or len(self.times_fs) < 2
            or self.values.shape != self.times_fs.shape
            or not np.isfinite(self.times_fs).all()
            or not np.isfinite(self.values).all()
            or np.any(np.diff(self.times_fs) <= 0)
            or np.any(self.values < 0)
        ):
            raise ValueError(
                "pulse requires increasing finite times and nonnegative flux"
            )
        if (
            not np.isfinite([photon_energy_eV, bandwidth_fwhm_eV]).all()
            or photon_energy_eV <= 0
            or bandwidth_fwhm_eV < 0
        ):
            raise ValueError("invalid pulse energy/bandwidth")
        self.photon_energy_eV = float(photon_energy_eV)
        self.bandwidth_fwhm_eV = float(bandwidth_fwhm_eV)
        self.times_fs.setflags(write=False)
        self.values.setflags(write=False)
        self.fluence_photons_per_cm2 = float(
            trapezoid(self.values, self.times_fs) * 1e-15
        )

    @property
    def support_fs(self):
        return self.times_fs[0], self.times_fs[-1]

    def flux(self, t_fs):
        return np.interp(t_fs, self.times_fs, self.values, left=0.0, right=0.0)


@dataclass(frozen=True)
class KineticTransition:
    source: str
    target: str
    coefficient: float
    kind: str
    electrons: int = 0
    photons: int = 0
    provenance: str = ""


@dataclass
class PopulationDynamics:
    times_fs: np.ndarray
    state_ids: tuple
    populations: np.ndarray  # time x state
    charges: np.ndarray
    photons_absorbed: np.ndarray
    electrons_emitted: np.ndarray
    provenance: dict

    @property
    def mean_charge(self):
        return self.populations @ self.charges


class KineticGraph:
    """Validated Markov graph, spontaneous s^-1 and light-driven cm^2 edges."""

    light_kinds = frozenset(
        ("pi", "double_pi", "resonant_absorption", "stimulated_emission")
    )
    decay_kinds = frozenset(("auger", "radiative"))

    def __init__(
        self, charges, transitions, *, initial_state, photon_energy_eV, provenance=None
    ):
        if initial_state not in charges or "" not in charges:
            raise ValueError("graph needs explicit initial and bare states")
        if not np.isfinite(photon_energy_eV) or photon_energy_eV <= 0:
            raise ValueError("invalid photon energy")
        self.charges = dict(charges)
        self.transitions = tuple(transitions)
        self.initial_state = initial_state
        self.photon_energy_eV = float(photon_energy_eV)
        self.provenance = dict(provenance or {})
        for q in charges.values():
            if not np.isfinite(q) or q < 0 or q != int(q):
                raise ValueError("charges must be nonnegative integers")
        for edge in self.transitions:
            if edge.source not in charges or edge.target not in charges:
                raise ValueError(f"missing transition endpoint: {edge}")
            if (
                edge.source == edge.target
                or edge.kind not in self.light_kinds | self.decay_kinds
            ):
                raise ValueError("invalid transition type or self edge")
            if not np.isfinite(edge.coefficient) or edge.coefficient < 0:
                raise ValueError(
                    "transition coefficients must be finite and nonnegative"
                )
            expected = {
                "pi": 1,
                "double_pi": 2,
                "auger": 1,
                "radiative": 0,
                "resonant_absorption": 0,
                "stimulated_emission": 0,
            }[edge.kind]
            if (
                charges[edge.target] - charges[edge.source] != expected
                or edge.electrons != expected
            ):
                raise ValueError("electron/charge accounting disagrees with transition")
            photon_count = {
                "pi": 1,
                "double_pi": 1,
                "resonant_absorption": 1,
                "stimulated_emission": -1,
            }.get(edge.kind, 0)
            if edge.photons != photon_count:
                raise ValueError("photon accounting disagrees with transition")
        self.state_ids = tuple(charges)
        self.indices = {s: i for i, s in enumerate(self.state_ids)}
        self.decay_matrix = self._matrix(False)
        self.photon_matrix = self._matrix(True)

    def _matrix(self, light):
        n = len(self.state_ids)
        rows, cols, values = [], [], []
        for e in self.transitions:
            if (e.kind in self.light_kinds) != light or e.coefficient == 0:
                continue
            a, b = self.indices[e.source], self.indices[e.target]
            rows.extend((b, a, n, n + 1))
            cols.extend((a, a, a, a))
            values.extend(
                (
                    e.coefficient,
                    -e.coefficient,
                    e.photons * e.coefficient,
                    e.electrons * e.coefficient,
                )
            )
        return coo_matrix((values, (rows, cols)), shape=(n + 2, n + 2)).tocsc()

    @classmethod
    def from_rate_table(cls, table, base_config, *, initial_config=None):
        base = _occupation_key(base_config)
        nbase = sum(q for _, _, q in base)
        if nbase != table["base_nelec"]:
            raise ValueError("base electron count differs from table")
        nuclear_charge = table.get(
            "Z",
            table.get("provenance", {})
            .get("identity", {})
            .get("controls", {})
            .get("Z"),
        )
        if nuclear_charge is None:
            raise ValueError("rate table needs explicit nuclear charge Z")
        if nuclear_charge < nbase:
            raise ValueError("nuclear charge is below the base electron count")
        keys = {}
        charges = {"": nuclear_charge}
        rows = {}
        for row in table["configs"].values():
            c = {_parse_shell(k): q for k, q in row["occupations"].items()}
            key = _occupation_key(c)
            sid = _table_key(c)
            if sid in rows:
                raise ValueError("duplicate jj configuration")
            rows[sid] = row
            keys[key] = sid
            charges[sid] = nuclear_charge - sum(c.values())
        keys[()] = ""

        def destination(key, changes):
            d = _daughter(key, changes)
            if d not in keys:
                raise ValueError(f"incomplete rate table: daughter {d}")
            return keys[d]

        edges = []
        for key, sid in keys.items():
            if not key:
                continue
            row = rows[sid]
            for shell, sigma in row["sigma_cm2"].items():
                target = destination(key, [(_parse_shell(shell), -1)])
                edges.append(
                    KineticTransition(
                        sid, target, sigma, "pi", 1, 1, "table subshell PI"
                    )
                )
            for ch in row["auger"]:
                target = destination(
                    key,
                    [
                        (tuple(ch["hole"]), 1),
                        (tuple(ch["s1"]), -1),
                        (tuple(ch["s2"]), -1),
                    ],
                )
                edges.append(
                    KineticTransition(
                        sid, target, ch["rate"], "auger", 1, 0, "table Auger"
                    )
                )
            for ch in row.get("radiative", []):
                target = destination(
                    key, [(tuple(ch["hole"]), 1), (tuple(ch["source"]), -1)]
                )
                edges.append(
                    KineticTransition(
                        sid,
                        target,
                        ch["rate"],
                        "radiative",
                        0,
                        0,
                        "table E1 fluorescence",
                    )
                )
            for ch in row.get("double_pi", []):
                changes = [(tuple(s), -1) for s in ch["removed"]]
                target = destination(key, changes)
                if len(changes) != 2 or not ch.get("provenance"):
                    raise ValueError(
                        "double PI needs two removals and explicit provenance"
                    )
                edges.append(
                    KineticTransition(
                        sid,
                        target,
                        ch["sigma_cm2"],
                        "double_pi",
                        2,
                        1,
                        ch["provenance"],
                    )
                )
        return cls(
            charges,
            edges,
            initial_state=_table_key(
                base_config if initial_config is None else initial_config
            ),
            photon_energy_eV=table["photon_energy_eV"],
            provenance=dict(
                table.get("provenance", {}),
                continuum_bandwidth_fwhm_eV=table.get("provenance", {}).get(
                    "continuum_bandwidth_fwhm_eV", 0.0
                ),
            ),
        )

    def solve(
        self,
        pulse,
        *,
        times_fs=None,
        initial_populations=None,
        after_pulse_fs=0.0,
        rtol=1e-8,
        atol=1e-12,
    ):
        """Sparse Radau evolution with lifetimes, radiation and event accounting."""
        if not np.isclose(
            pulse.photon_energy_eV, self.photon_energy_eV, rtol=0, atol=1e-9
        ):
            raise ValueError("pulse energy differs from rate table")
        if not np.isfinite(after_pulse_fs) or after_pulse_fs < 0:
            raise ValueError("post-pulse duration must be nonnegative")
        continuum_bandwidth = self.provenance.get("continuum_bandwidth_fwhm_eV")
        if continuum_bandwidth is not None and not np.isclose(
            continuum_bandwidth, pulse.bandwidth_fwhm_eV, rtol=0, atol=1e-12
        ):
            raise ValueError(
                "pulse bandwidth differs from continuum cross-section table"
            )
        bound_bandwidth = self.provenance.get("resonance_bandwidth_fwhm_eV")
        if bound_bandwidth is not None and not np.isclose(
            bound_bandwidth, pulse.bandwidth_fwhm_eV
        ):
            raise ValueError("pulse bandwidth differs from resonance profile")
        if self.provenance.get("resonance_validity"):
            if isinstance(pulse, GaussianPulse):
                minimum = 4 * np.log(2) * (HARTREE_EV / AU_RATE_PER_S) * 1e15
                if pulse.bandwidth_fwhm_eV * pulse.fwhm_fs < minimum:
                    raise ValueError(
                        "resonant Gaussian pulse violates time-bandwidth lower bound"
                    )
                peak_flux = float(pulse.flux(pulse.center_fs))
            else:
                peak_flux = float(max(pulse.values))
            for gate in self.provenance["resonance_validity"]:
                if (
                    peak_flux * gate["peak_pumping_sigma_cm2"]
                    > 0.1 * gate["correlation_rate_s"]
                ):
                    raise ValueError(
                        "strong-field resonance exceeds the conservative incoherent-rate gate; "
                        "a coherence-aware optical Bloch model is required"
                    )
        start, end = pulse.support_fs
        end += after_pulse_fs
        times = (
            np.linspace(start, end, 1001)
            if times_fs is None
            else np.asarray(times_fs, dtype=float)
        )
        if (
            times.ndim != 1
            or len(times) < 2
            or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
            or times[0] < start
            or times[-1] > end
        ):
            raise ValueError("output times must increase within integration support")
        n = len(self.state_ids)
        y0 = np.zeros(n + 2)
        if initial_populations is None:
            y0[self.indices[self.initial_state]] = 1.0
        else:
            for sid, v in initial_populations.items():
                y0[self.indices[sid]] = v
            if (
                not np.isfinite(y0).all()
                or np.any(y0 < 0)
                or not np.isclose(sum(y0), 1, rtol=0, atol=1e-12)
            ):
                raise ValueError("initial populations must sum to one")

        def jac(t, _y=None):
            return (
                self.decay_matrix + float(pulse.flux(t)) * self.photon_matrix
            ) * 1e-15

        # Resolve temporal envelopes even when all initial derivatives vanish.
        max_step = (
            pulse.fwhm_fs / 20
            if isinstance(pulse, GaussianPulse)
            else float(np.min(np.diff(pulse.times_fs))) / 2
        )
        sol = solve_ivp(
            lambda t, y: jac(t) @ y,
            (start, end),
            y0,
            t_eval=times,
            method="Radau",
            jac=jac,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        pop = sol.y[:n].T
        if (
            not np.isfinite(pop).all()
            or np.min(pop) < -max(1e-9, 100 * atol)
            or np.max(abs(pop.sum(axis=1) - 1)) > max(1e-8, 100 * atol)
        ):
            raise RuntimeError("population positivity or conservation gate failed")
        charges = np.array([self.charges[s] for s in self.state_ids])
        initial_charge = y0[:n] @ charges
        if not np.allclose(
            pop @ charges - initial_charge, sol.y[n + 1], rtol=1e-7, atol=1e-8
        ):
            raise RuntimeError("emitted electrons fail charge accounting")
        return PopulationDynamics(
            times.copy(),
            self.state_ids,
            pop,
            charges,
            sol.y[n],
            sol.y[n + 1],
            dict(
                self.provenance,
                method="finite-lifetime sparse Radau",
                photon_energy_eV=self.photon_energy_eV,
                pulse_bandwidth_fwhm_eV=pulse.bandwidth_fwhm_eV,
            ),
        )
