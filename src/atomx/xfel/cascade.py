"""Deterministic sequential-photoionization cascade (XFEL increment 1).

The XATOM ``track_elec_config`` analog, first increment per
docs/XFEL_GAP_ANALYSIS.md: an explicit configuration graph pumped by a
photon fluence, with per-configuration rate tables built from the
validated atomx providers —

* subshell PI: :func:`atomx.scattering.anomalous.atomic_sigma_pi`
  (validated vs FAC panels and the Sommerfeld-analytic H 1s curve);
* Auger: the LS chain of ``auger_rate_csf_average`` (Ne KLL closes at
  1.025x of XATOM channel totals, gated in
  tests/test_auger_ne_kll_xatom.py).

Approximation (documented, first increment): a SINGLE absorption
pass per configuration — ``P = fluence * sigma`` (photons/cm^2 times
cm^2, pulse-integrated), applied once, with decay instantaneous
against the pulse (Auger ~fs vs XFEL ~10-100 fs) and the Auger tree
walked by rate ordering (strongest channel first).  This is the
first-order picture valid for fluence*sigma << 1; the re-absorption
of survivors (sequential multiphoton saturation) needs the
time-resolved / MC increment — iterating the single pass instead
would wrongly ionise everything at any nonzero fluence.  Radiative
stabilisation and shakeoff are deferred (gap analysis §4).

Public API: :func:`cascade`.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Tuple

Subshell = Tuple[int, int]          # (n, kappa)
Config = Mapping[Subshell, float]   # occupied subshells only
ConfigKey = Tuple[Tuple[int, int, float], ...]

_CM2_PER_A0SQ = 2.8002852e-17
_LCHAR = "spdfghiklmno"


@dataclass(frozen=True)
class Pulse:
    """Monochromatic, pulse-integrated photon fluence."""

    photon_energy_eV: float
    fluence_photons_per_cm2: float


@dataclass
class CascadeResult:
    populations: Dict[ConfigKey, float] = field(default_factory=dict)
    charge_distribution: Dict[int, float] = field(default_factory=dict)
    electrons_emitted: float = 0.0
    photons_absorbed: float = 0.0
    configs_visited: int = 0
    steps: int = 0


def config_label(config: Config) -> str:
    """LS-style label with jj shells MERGED per (n, l): the Atom config
    grammar has no separate tokens for j-split partners, and emitting
    both (e.g. 2p2 2p2) constructs an ambiguous duplicate-subshell
    config that the ExactCI engine rightly rejects."""
    merged: Dict[Tuple[int, int], float] = {}
    for (n, k), q in config.items():
        l = k if k > 0 else -k - 1
        merged[(n, l)] = merged.get((n, l), 0.0) + q
    return " ".join(
        f"{n}{_LCHAR[l]}{int(round(q))}"
        for (n, l), q in sorted(merged.items()))


def _key(config: Config) -> ConfigKey:
    """Occupation-resolved: 1s1 2p6 and 1s2 2p6 are DIFFERENT nodes.

    (A subshell-only key collided them, so the Auger-refilled parent
    re-absorbed forever — the first live bug, caught by the mock gates.)
    """
    return tuple(sorted((n, k, round(q, 6))
                        for (n, k), q in config.items() if q >= 0.5))


def _daughter_after_pi(config: Config, target: Subshell) -> Config:
    out = dict(config)
    out[target] = out.get(target, 0.0) - 1.0
    if out[target] < 0.5:
        out.pop(target)
    return out


def _daughter_after_auger(config: Config, hole: Subshell,
                          source: Subshell, source2: Subshell) -> Config:
    out = dict(config)
    out[hole] = out.get(hole, 0.0) + 1.0
    for s in (source, source2):
        out[s] = out.get(s, 0.0) - 1.0
        if out[s] < 0.5:
            out.pop(s)
    return out


class RateProvider:
    """Per-configuration rate tables, memoised on the configuration graph.

    ``atom_factory(config_label)`` must solve the structure for the
    labelled configuration and return the solved result object.
    Subclass to swap providers (mscheme-CI Auger tier, resonant PI).
    """

    def __init__(self, atom_factory: Callable[[str], object]):
        self._atom_factory = atom_factory
        self._unwrap = lambda r: getattr(r, "_result", r)
        self._solves: Dict[ConfigKey, object] = {}
        self._pi_cache: Dict[Tuple[ConfigKey, float], Dict[Subshell, float]] = {}
        self._auger_cache: Dict[ConfigKey, List] = {}

    # -- override points ---------------------------------------------------

    def pi_sigmas_cm2(self, config: Config,
                      photon_energy_eV: float) -> Dict[Subshell, float]:
        """Per-subshell PI cross section in cm^2 (subshell sum).

        SHOOTING + MILNE (2026-10-03): the default B-spline pseudo-state
        continuum cannot resolve a 1 keV photoelectron (k ~ 8.6 a.u.) —
        measured on Ne 1s at 1 keV: 0.0245 Mb vs XATOM's 0.2321 (9.5x
        low; basis enlargement made it worse).  The RR module's shooting
        Dirac continuum inverted through the Milne relation measures
        0.2335 Mb — ratio 1.006.  This path: one rr_cross_section call
        per subshell with capture_orbitals pinned, Milne factor
        a^2*w^2/(E(2+a^2 E)) per the RR module's own documented
        convention (Gu 2003 eq. 1).
        """
        key = _key(config)
        ckey = (key, photon_energy_eV)
        if ckey not in self._pi_cache:
            from atomx.constants import ALPHA
            from atomx.scattering.models import ContinuumModel
            from atomx.scattering.radiative_recombination import (
                rr_cross_section,
            )

            result = self._unwrap(self._solve(config))
            omega_ha = photon_energy_eV / 27.211386245988
            table: Dict[Subshell, float] = {}
            for sh, q in config.items():
                label = _subshell_label(sh)
                e_bind = getattr(result, "orbital_energies", {}).get(label)
                if e_bind is None or e_bind >= 0.0:
                    table[sh] = 0.0
                    continue
                e_free_eV = photon_energy_eV + e_bind * 27.211386245988
                if e_free_eV <= 0.0:
                    table[sh] = 0.0     # below threshold
                    continue
                rr = rr_cross_section(
                    result,
                    np.array([e_free_eV]),
                    capture_orbitals=[label],
                    continuum_model=ContinuumModel(method="shooting"),
                    multipoles=("E1",),
                    g_initial=1,
                )
                e_ha = e_free_eV / 27.211386245988
                milne = (ALPHA ** 2 * omega_ha ** 2
                         / (e_ha * (2.0 + ALPHA ** 2 * e_ha)))
                s_rr = float(rr.sigma_by_orbital.get(label, [0.0])[0])
                table[sh] = (s_rr / milne) * _CM2_PER_A0SQ
            self._pi_cache[ckey] = table
        return self._pi_cache[ckey]

    def auger_channels(self, config: Config) -> List[Tuple[Subshell,
                                                           Subshell,
                                                           Subshell,
                                                           float]]:
        """Open Auger channels [(hole, source, source2, rate_per_s)]."""
        key = _key(config)
        if key not in self._auger_cache:
            self._auger_cache[key] = self._build_auger(config)
        return self._auger_cache[key]

    # -- internals ----------------------------------------------------------

    def _solve(self, config: Config) -> object:
        key = _key(config)
        if key not in self._solves:
            self._solves[key] = self._atom_factory(config_label(config))
        return self._solves[key]

    def _build_auger(self, config: Config) -> list:
        from atomx.scattering.autoionization import auger_rate_csf_average

        # HOLE GATE (2026-10-02 XFEL increment): an Auger channel needs
        # a vacancy (occupation below capacity) with at least two
        # electrons in HIGHER subshells to fill and emit.  Closed shells
        # have no channels, but the LS chain runs the full shooting
        # continuum solve per subshell-triple before discovering the
        # rate is zero — gating by vacancy presence first turns the
        # ~64-solves-per-config enumeration into ~0 for closed shells
        # (the dominant configuration-graph population).
        def _capacity(k: int) -> float:
            return 2.0 * (2.0 * (k if k > 0 else -k - 1) + 1)

        subs_sorted = sorted(config)  # (n, kappa) ascending
        triples = []
        for i, hole in enumerate(subs_sorted):
            if config[hole] >= _capacity(hole[1]):
                continue                      # full: no vacancy
            higher = [s for s in subs_sorted[i + 1:]
                      if config.get(s, 0.0) >= 1.0]
            if len(higher) < 1:
                continue                      # nothing above to decay
            for a in higher:
                for b in higher:
                    if b < a:
                        continue
                    if config[a] + config[b] < 1.5:
                        continue
                    if (a == b and config[a] < 1.5):
                        continue              # equivalent pair needs 2
                    triples.append((hole, a, b))
        if not triples:
            return []

        result = self._unwrap(self._solve(config))
        labels = {sh: _subshell_label(sh) for sh in config}
        channels = []
        for nh_kh, ns1_ks1, ns2_ks2 in triples:
            out = auger_rate_csf_average(
                result,
                hole_label=labels[nh_kh],
                source_label=labels[ns1_ks1],
                source2_label=labels[ns2_ks2],
                shooting_n_grid=200,
            )
            gamma = float(out["Gamma_per_s"])
            if gamma > 0.0:
                channels.append((nh_kh, ns1_ks1, ns2_ks2, gamma))
        return channels


def _subshell_label(sh: Subshell) -> str:
    n, k = sh
    return f"{n}{_LCHAR[(k if k > 0 else -k - 1)]}{2 * abs(k) - 1}/2"


def _occupation_table(result) -> Dict[Subshell, float]:
    if getattr(result, "occupations", None):
        return dict(result.occupations)
    return {(sh.n, sh.kappa): float(sh.occupation)
            for sh in getattr(result, "subshells", [])}


def _auger_tree(config: Config, prov: RateProvider,
                configs: Dict[ConfigKey, Config]) -> Tuple[Config, int]:
    """Instant Auger cascade by rate ordering; returns the terminal
    configuration and the number of Auger electrons emitted."""
    config = dict(config)
    emitted = 0
    for _ in range(24):                    # bounded hole migration
        channels = prov.auger_channels(config)
        if not channels:
            break
        hole, s1, s2, _g = max(channels, key=lambda c: c[3])
        config = _daughter_after_auger(config, hole, s1, s2)
        emitted += 1
        key = _key(config)
        if key not in configs:
            configs[key] = dict(config)
    return config, emitted


def cascade(atom_factory: Callable[[str], object], base_config: Config,
            pulse: Pulse, *, provider: Optional[RateProvider] = None,
            max_charge: int = 12) -> CascadeResult:
    """Deterministic sequential-PI cascade over the configuration graph."""
    prov = provider or RateProvider(atom_factory)
    res = CascadeResult()
    base_nelec = sum(base_config.values())

    configs: Dict[ConfigKey, Config] = {_key(base_config): dict(base_config)}
    populations: Dict[ConfigKey, float] = {_key(base_config): 1.0}

    # single absorption pass (see module docstring for the regime)
    if True:
        res.steps = 1
        nxt: Dict[ConfigKey, float] = {}
        emitted_this_step = 0.0
        for key, pop in populations.items():
            config = configs[key]
            charge = base_nelec - sum(config.values())
            if charge >= max_charge:
                nxt[key] = nxt.get(key, 0.0) + pop
                continue
            sigmas = prov.pi_sigmas_cm2(config, pulse.photon_energy_eV)
            p_abs = {sh: pulse.fluence_photons_per_cm2 * s
                     for sh, s in sigmas.items() if s > 0.0}
            p_sum = sum(p_abs.values())
            if p_sum <= 1e-12:
                nxt[key] = nxt.get(key, 0.0) + pop
                continue
            p_sum = min(p_sum, 1.0)
            nxt[key] = nxt.get(key, 0.0) + pop * (1.0 - p_sum)
            absorbed = pop * p_sum
            res.photons_absorbed += absorbed
            for sh, p_sh in p_abs.items():
                w_sh = p_sh / sum(p_abs.values())
                daughter = _daughter_after_pi(config, sh)
                terminal, n_auger = _auger_tree(daughter, prov, configs)
                # 1 photoelectron + n_auger Auger electrons per event
                emitted_this_step += absorbed * w_sh * (1 + n_auger)
                tkey = _key(terminal)
                if tkey not in configs:
                    configs[tkey] = dict(terminal)
                nxt[tkey] = nxt.get(tkey, 0.0) + absorbed * w_sh
        res.electrons_emitted += emitted_this_step
        populations = {k: v for k, v in nxt.items() if v > 1e-15}

    for key, pop in populations.items():
        charge = int(round(base_nelec - sum(configs[key].values())))
        res.populations[key] = pop
        res.charge_distribution[charge] = (
            res.charge_distribution.get(charge, 0.0) + pop)
    res.configs_visited = len(configs)
    return res




# ---------------------------------------------------------------------------
# Increment 2: Monte-Carlo over the pulse (re-absorption + branching trees)
# ---------------------------------------------------------------------------

def _auger_tree_branching(config: Config, prov: RateProvider,
                          configs: Dict[ConfigKey, Config],
                          rng) -> Tuple[Config, int, bool]:
    """Stochastic Auger cascade: sample channels by rate ratio; returns
    (terminal config, electrons emitted, whether any channel fired)."""
    config = dict(config)
    emitted = 0
    for _ in range(24):
        channels = prov.auger_channels(config)
        if not channels:
            break
        total = sum(c[3] for c in channels)
        pick = rng.random() * total
        acc = 0.0
        chosen = channels[-1]
        for cand in channels:
            acc += cand[3]
            if pick <= acc:
                chosen = cand
                break
        hole, s1, s2 = chosen[0], chosen[1], chosen[2]
        config = _daughter_after_auger(config, hole, s1, s2)
        emitted += 1
        key = _key(config)
        if key not in configs:
            configs[key] = dict(config)
    return config, emitted, bool(emitted)


@dataclass
class MCResult:
    n_trajectories: int
    charge_distribution: Dict[int, float] = field(default_factory=dict)
    mean_charge: float = 0.0
    mean_absorbed: float = 0.0
    mean_emitted: float = 0.0
    configs_visited: int = 0


def cascade_mc(atom_factory: Callable[[str], object], base_config: Config,
               pulse: Pulse, *, n_trajectories: int = 2000,
               n_bins: int = 8, max_charge: int = 12,
               seed: int = 7, provider: Optional[RateProvider] = None,
               ) -> MCResult:
    """Photon-by-photon MC over the pulse — the xatom_mc analog.

    The pulse is discretised into ``n_bins`` time bins; per bin the
    absorption probability per subshell is
    ``P_bin = min(1, fluence*sigma/n_bins)`` and the absorbing subshell
    is SAMPLED by cross-section weight; after each absorption the Auger
    tree branches stochastically by rate ratios.  Neutral survival
    converges to ``exp(-fluence*sigma)`` as ``n_bins -> inf`` (the
    time-resolved rate-equation limit); the deterministic single pass
    of :func:`cascade` is its first-order expansion ``1 - fluence*sigma``.
    Failed draws CONSUME a bin (no retry): retrying would inflate the
    absorbed fraction to 1 at any nonzero fluence.
    """
    import random

    rng = random.Random(seed)
    prov = provider or RateProvider(atom_factory)
    base_nelec = sum(base_config.values())
    configs: Dict[ConfigKey, Config] = {_key(base_config): dict(base_config)}
    charge_counts: Dict[int, int] = {}
    absorbed_total = 0
    emitted_total = 0

    for _ in range(n_trajectories):
        config = dict(base_config)
        for _bin in range(n_bins):
            key = _key(config)
            if key not in configs:
                configs[key] = dict(config)
            charge = base_nelec - sum(config.values())
            if charge >= max_charge:
                break
            sigmas = prov.pi_sigmas_cm2(config, pulse.photon_energy_eV)
            p_abs = {sh: pulse.fluence_photons_per_cm2 * s / n_bins
                     for sh, s in sigmas.items() if s > 0.0}
            p_sum = sum(p_abs.values())
            if p_sum <= 1e-12:
                break                      # thresholds closed: frozen
            if rng.random() > min(p_sum, 1.0):
                continue                   # bin passes without absorption
            # sample the absorbing subshell by weight
            pick = rng.random() * sum(p_abs.values())
            acc = 0.0
            target = None
            for sh, p in p_abs.items():
                acc += p
                if pick <= acc:
                    target = sh
                    break
            if target is None:
                target = next(iter(p_abs))
            daughter = _daughter_after_pi(config, target)
            terminal, n_auger, _ = _auger_tree_branching(
                daughter, prov, configs, rng)
            absorbed_total += 1
            emitted_total += 1 + n_auger
            config = terminal
        charge = int(round(base_nelec - sum(config.values())))
        charge_counts[charge] = charge_counts.get(charge, 0) + 1

    res = MCResult(n_trajectories=n_trajectories,
                   configs_visited=len(configs))
    for q, n in charge_counts.items():
        res.charge_distribution[q] = n / n_trajectories
    res.mean_charge = sum(q * p for q, p in
                          res.charge_distribution.items())
    res.mean_absorbed = absorbed_total / n_trajectories
    res.mean_emitted = emitted_total / n_trajectories
    return res
