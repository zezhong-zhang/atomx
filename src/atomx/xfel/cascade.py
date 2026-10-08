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

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from atomx.angular.recoupling import lj_from_kappa
from atomx.config.subshell import L_LETTERS, l_letter

Subshell = Tuple[int, int]          # (n, kappa)
Config = Mapping[Subshell, float]   # occupied subshells only
ConfigKey = Tuple[Tuple[int, int, float], ...]

_CM2_PER_A0SQ = 2.8002852e-17
_LCHAR = L_LETTERS


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


def _table_key(config: Config) -> str:
    """Keep jj occupations in serialized identities, not merged LS labels.

    Two jj distributions can share the human label ``2p3`` but remain
    different cascade states. A merged label must not overwrite rows.
    """
    return "|".join(f"{n},{k},{float(q)!r}" for n, k, q in _key(config))


def shell_support(config: Config) -> frozenset:
    """Merged-(n, l) shells occupied — the skeleton-sharing key
    (performance plan layer 0a).

    Every configuration with the same occupied (n, l) shell SET shares
    ONE structure solve.  Ne: the 62 enumerated configurations collapse
    to 7 supports ({1s,2s,2p} ... {2p}); the user-measured "63 -> ~8"
    of the 2026-10-03 work order.
    """
    return frozenset((n, k if k > 0 else -k - 1) for (n, k) in config)


def _partner_kappa(k: int) -> int | None:
    """The other kappa of the same (n, l) shell, if it exists."""
    ell = k if k > 0 else -k - 1
    if ell == 0:
        return None
    return -(ell + 1) if k > 0 else ell


def skeleton_config_for(config: Config) -> Config:
    """Maximal occupation over the same merged shells: every jj subshell
    of every occupied (n, l) shell filled to its 2|kappa| capacity.

    The xatom_mc_fly architecture ("one atom, one orbital set") made
    per-shell-family: the skeleton is a CLOSED shell — the cleanest SCF
    anchor — and every sub-configuration of the family reuses its
    radials.  Justified by the triple radial falsification (frozen
    MCDHF = DHF = relaxed-DHF on the Fe anchor panel: the radial basis
    is insensitive to the ionisation path).
    """
    out: Dict[Subshell, float] = {}
    for (n, k) in config:
        out[(n, k)] = 2.0 * abs(k)
        pk = _partner_kappa(k)
        if pk is not None:
            out[(n, pk)] = 2.0 * abs(pk)
    return out


def auger_triples(config: Config) -> List[Tuple[Subshell, Subshell, Subshell]]:
    """Hole-gate-legal (hole, source, source2) triples — pure occupation
    logic, no solve (the gate of ``RateProvider._build_auger``).

    An Auger channel needs a VACANCY (jj occupation below the 2|kappa|
    capacity — the same 4l+2 -> 2|kappa| correction enumerate_configs
    received in v3; the old merged-shell gate admitted spurious
    same-shell holes on jj-closed subshells, which only the negative
    Auger-energy check caught) with electron(s) in HIGHER subshells to
    fill and emit.  Closed shells have no channels, but the LS chain
    runs the full shooting continuum solve per triple before
    discovering the rate is zero — gating by vacancy presence first
    turns the ~64-solves-per-config enumeration into ~0 for closed
    shells (the dominant configuration-graph population).
    """
    def _capacity(k: int) -> float:
        return 2.0 * abs(k)

    def _subshell_order(nk: Subshell):
        # ENERGY order, not raw (n, kappa) tuple order: the raw sort
        # puts kappa=-2 (2p3/2) BEFORE kappa=+1 (2p1/2) and 2s between
        # them, so L-vacancy channels lost 2p3/2 sources and the L2L3
        # Coster-Kronig set entirely (1s-hole KLL channels were
        # unaffected — every L subshell sorts above 1s either way —
        # which is why the Ne anchor panel never caught it).
        n, k = nk
        ell = k if k > 0 else -k - 1
        return (n, ell, 1 if k < 0 else 0)

    subs_sorted = sorted(config, key=_subshell_order)
    triples: List[Tuple[Subshell, Subshell, Subshell]] = []
    for i, hole in enumerate(subs_sorted):
        if config[hole] >= _capacity(hole[1]):
            continue                      # full: no vacancy
        higher = [s for s in subs_sorted[i + 1:]
                  if config.get(s, 0.0) >= 1.0]
        if not higher:
            continue                      # nothing above to decay
        for a in higher:
            for b in higher:
                if b < a:
                    continue
                if config[a] + config[b] < 1.5:
                    continue
                if a == b and config[a] < 1.5:
                    continue              # equivalent pair needs 2
                triples.append((hole, a, b))
    return triples


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

    Structure solves are shared by SHELL SUPPORT (layer 0a, default):
    every configuration of the same occupied-(n,l)-shell set reuses the
    radials of its closed-shell skeleton (:func:`skeleton_config_for`).
    ``n_solves`` counts skeleton solves for measurement gates.  An
    explicit ``skeleton_config`` forces ONE global skeleton instead
    (the xatom_mc_fly single-orbital-set mode).
    """

    def __init__(self, atom_factory: Callable[[str], object],
                 skeleton_config: Config | None = None):
        self._atom_factory = atom_factory
        self._unwrap = lambda r: getattr(r, "_result", r)
        self._forced_skeleton: object | None = None
        if skeleton_config is not None:
            self._forced_skeleton = self._unwrap(
                self._atom_factory(config_label(skeleton_config)))
        self._skeletons: Dict[frozenset, object] = {}
        self.n_solves = 0
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
            from atomx.constants import ALPHA, HARTREE_EV
            from atomx.scattering.models import ContinuumModel
            from atomx.scattering.radiative_recombination import (
                rr_cross_section,
            )

            result = self._unwrap(self._solve(config))
            omega_ha = photon_energy_eV / HARTREE_EV
            table: Dict[Subshell, float] = {}
            for sh in config:
                label = _subshell_label(sh)
                e_bind = getattr(result, "orbital_energies", {}).get(label)
                if e_bind is None or e_bind >= 0.0:
                    table[sh] = 0.0
                    continue
                e_free_eV = photon_energy_eV + e_bind * HARTREE_EV
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
                e_ha = e_free_eV / HARTREE_EV
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
        if self._forced_skeleton is not None:
            return self._forced_skeleton
        support = shell_support(config)
        if support not in self._skeletons:
            self._skeletons[support] = self._unwrap(
                self._atom_factory(config_label(skeleton_config_for(config))))
            self.n_solves += 1
        return self._skeletons[support]

    def _build_auger(self, config: Config) -> list:
        from atomx.constants import HARTREE_EV
        from atomx.scattering.autoionization import auger_rate_csf_average

        triples = auger_triples(config)
        if not triples:
            return []

        result = self._unwrap(self._solve(config))
        energies = getattr(result, "orbital_energies", {})
        labels = {sh: _subshell_label(sh) for sh in config}
        channels = []
        for nh_kh, ns1_ks1, ns2_ks2 in triples:
            # Auger electron energy from orbital-energy differences:
            # E = I(hole) - I(s1) - I(s2) = eps_s1 + eps_s2 - eps_hole
            # (2s2s check on Ne HFS: 770 eV vs XATOM 815 — coarse but
            # the rate's E-dependence is weak; the earlier omission of
            # resonance_energy_eV left E_ha = 0 and EVERY channel
            # silently zero — the K-hole daughter never Auger-decayed)
            e_h = energies.get(labels[nh_kh])
            e_1 = energies.get(labels[ns1_ks1])
            e_2 = energies.get(labels[ns2_ks2])
            if e_h is None or e_1 is None or e_2 is None:
                continue
            e_aug_eV = (e_1 + e_2 - e_h) * HARTREE_EV
            if e_aug_eV <= 0.0:
                continue
            out = auger_rate_csf_average(
                result,
                hole_label=labels[nh_kh],
                source_label=labels[ns1_ks1],
                source2_label=labels[ns2_ks2],
                resonance_energy_eV=e_aug_eV,
                shooting_n_grid=200,
            )
            gamma = float(out["Gamma_per_s"])
            if gamma > 0.0:
                channels.append((nh_kh, ns1_ks1, ns2_ks2, gamma))
        return channels


def _subshell_label(sh: Subshell) -> str:
    n, k = sh
    l, j2 = lj_from_kappa(k)
    return f"{n}{l_letter(l)}{j2}/2"


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


# ---------------------------------------------------------------------------
# Increment: table-driven provider (performance plan layer 2)
# ---------------------------------------------------------------------------

def build_rate_table(atom_factory, base_config: Config,
                     photon_energy_eV: float, *,
                     max_charge: int = 12,
                     provider: Optional[RateProvider] = None) -> dict:
    """Enumerate the reachable configuration graph ONCE and serialise
    every configuration's rate table (PI per subshell, Auger channels)
    to a picklable dict — the XATOM results.dat architecture.

    Generation walks the graph exactly like cascade_mc (PI daughters x
    Auger trees, sampled per branch but deterministically bounded by
    max_charge); with ``seed`` fixed the enumeration is complete for
    the fluence-independent table.  Feed to :class:`TableRateProvider`
    for millisecond-per-trajectory MC.
    """

    prov = provider or RateProvider(atom_factory)
    base_nelec = sum(base_config.values())
    table: Dict[str, dict] = {}
    frontier = [dict(base_config)]
    seen = {_key(base_config)}
    while frontier:
        config = frontier.pop()
        if sum(config.values()) < 1.0:
            continue            # bare ion: no PI channels, no labels
        label = config_label(config)
        sigmas = prov.pi_sigmas_cm2(config, photon_energy_eV)
        channels = prov.auger_channels(config)
        table[_table_key(config)] = {
            "label": label,
            "sigma_cm2": {f"{n},{k}": s for (n, k), s in sigmas.items()},
            "auger": [
                {"hole": list(h), "s1": list(s1), "s2": list(s2),
                 "rate": float(g)}
                for h, s1, s2, g in channels
            ],
            "occupations": {f"{n},{k}": q for (n, k), q in config.items()},
        }
        if base_nelec - sum(config.values()) >= max_charge:
            continue
        for (n, k) in sigmas:
            daughter = _daughter_after_pi(config, (n, k))
            # EXHAUSTIVE Auger-tree enumeration (the single-sample walk
            # seeded one branch; a different-seed MC then hit terminals
            # missing from the table and froze at charge ~1 — measured
            # F=1000 ratio 0.18 vs the live 0.78).  Collect EVERY
            # reachable config down all channel orders, dedup on _key.
            aug_frontier = [daughter]
            aug_seen = set()
            while aug_frontier:
                aug_cfg = aug_frontier.pop()
                ak = _key(aug_cfg)
                if ak in aug_seen:
                    continue
                aug_seen.add(ak)
                tk = ak
                if tk not in seen:
                    seen.add(tk)
                    frontier.append(aug_cfg)
                for hole, s1, s2, _g in prov.auger_channels(aug_cfg):
                    nxt = _daughter_after_auger(aug_cfg, hole, s1, s2)
                    nk = _key(nxt)
                    if nk not in aug_seen:
                        aug_frontier.append(nxt)
    return {"photon_energy_eV": photon_energy_eV,
            "base_nelec": base_nelec, "configs": table}


def enumerate_configs(base_config: Config,
                     max_charge: int = 12) -> List[Config]:
    """All occupation tuples over the SAME subshell set (complete).

    PI only removes electrons from occupied subshells and Auger only
    moves them within, so the reachable graph from ``base_config``
    never grows the subshell set — every reachable configuration is
    an occupation tuple over the same (n, kappa) keys.  This makes
    the configuration list SOLVE-INDEPENDENT (the v1/v2 builders
    discovered configs by walking the graph, which requires a solve
    per step and serialised the whole build).
    """
    items = sorted(base_config.items())
    caps = [2 * abs(k) for (n, k), _q in items]  # 2j+1 = 2|kappa| per jj subshell
    base_total = sum(q for _k, q in items)
    floor_total = max(int(round(base_total)) - int(max_charge), 0)
    out: List[Config] = []

    def _rec(i: int, occ: List[int], remaining: int) -> None:
        if i == len(items):
            if remaining == 0:
                out.append({items[j][0]: float(occ[j])
                           for j in range(len(items)) if occ[j] > 0})
            return
        cap = min(caps[i], remaining)
        for q in range(cap, -1, -1):
            occ.append(q)
            _rec(i + 1, occ, remaining - q)
            occ.pop()

    for total in range(int(round(base_total)), max(floor_total, 1) - 1, -1):
        _rec(0, [], total)
    return out


def _rate_group_worker(job: dict) -> dict:
    """Pool worker: solve ONE shell-skeleton, build rows for its configs.

    ``job = {"configs": [{"n,k": q, ...}, ...], "z": Z, "photon_eV": E,
    "correlation": str, "auger": bool}`` — every config in a job shares
    the merged-(n,l) support, so the provider's lazy per-support sharing
    performs exactly ONE structure solve per job.  Runs with pinned BLAS
    threads and an isolated numba cache (the concurrent_solve
    convention).  Returns ``{"rows": [...], "solves": n}`` where each
    row is the v3 shape (label + sigma_cm2 + auger + occupations) or an
    ``{"error": ...}`` row so one bad config never kills the build.
    """
    import os
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    if os.environ.get("ATOMX_NUMBA_CACHE_DIR"):
        os.environ["NUMBA_CACHE_DIR"] = os.environ["ATOMX_NUMBA_CACHE_DIR"]

    rows: List[dict] = []
    solves = 0
    try:
        from atomx import Atom, Correlation

        z = int(job["z"])
        corr = job.get("correlation", "fac_frozen")
        photon_eV = float(job["photon_eV"])

        def factory(lbl):
            return Atom(
                Z=z,
                charge=z - sum(int(t[2:]) for t in lbl.split()),
                config=[lbl],
                correlation=Correlation(radial_optimization=corr),
                corrections="none").solve()

        prov = RateProvider(factory)
        for cfg_s in job["configs"]:
            label_config = {
                (int(k.split(",")[0]), int(k.split(",")[1])): q
                for k, q in cfg_s.items()
            }
            label = config_label(label_config)
            occ = {f"{n},{k}": q for (n, k), q in label_config.items()}
            try:
                sigmas = prov.pi_sigmas_cm2(label_config, photon_eV)
                channels = (prov.auger_channels(label_config)
                            if job.get("auger", True) else [])
                rows.append({
                    "label": label,
                    "sigma_cm2": {f"{n},{k}": v
                                  for (n, k), v in sigmas.items()},
                    "auger": [{"hole": list(h), "s1": list(s1),
                               "s2": list(s2), "rate": float(g)}
                              for h, s1, s2, g in channels],
                    "occupations": occ,
                })
            except Exception as exc:  # noqa: BLE001 - record, continue
                rows.append({"label": label,
                             "error": f"{type(exc).__name__}: {exc}",
                             "occupations": occ})
        solves = prov.n_solves
    except Exception as exc:  # noqa: BLE001 - whole-job failure
        rows.append({"label": "<worker>",
                     "error": f"{type(exc).__name__}: {exc}",
                     "occupations": {}})
    return {"rows": rows, "solves": solves}


def _support_of_serialized(cfg_s: dict) -> frozenset:
    """Merged-(n, l) support key of a serialized config dict."""
    out = set()
    for k in cfg_s:
        n, kk = (int(x) for x in k.split(","))
        out.add((n, kk if kk > 0 else -kk - 1))
    return frozenset(out)


def _group_by_support(cfg_dicts: List[dict]) -> List[List[dict]]:
    """Group serialized configs by merged-(n,l) support (deterministic
    order: size descending, then key)."""
    groups: Dict[frozenset, List[dict]] = {}
    for c in cfg_dicts:
        groups.setdefault(_support_of_serialized(c), []).append(c)
    return sorted(groups.values(),
                  key=lambda g: (-len(g), sorted(g[0])))


def _split_for_pool(groups: List[List[dict]],
                    n_workers: int) -> List[List[dict]]:
    """Split the largest groups until enough jobs exist to fill the pool.

    Every job solves its own skeleton, so splitting trades redundant
    solves for wall time — bounded: total solves <= n_jobs.
    """
    jobs = [list(g) for g in groups]
    while len(jobs) < n_workers:
        jobs.sort(key=lambda j: (-len(j), sorted(j[0])))
        big = jobs[0]
        if len(big) <= 1:
            break
        half = len(big) // 2
        jobs[0] = big[:half]
        jobs.append(big[half:])
    return jobs


def _prune_keep_set(base_config, configs, sigma_tot, fluence, threshold):
    """BFS over the reachable graph with the layer-0b prune rule.

    A configuration whose remaining PI channels give
    ``fluence * sigma_total < threshold`` (the first-order estimate of
    its re-absorption probability during the pulse) is a FROZEN
    TERMINAL: it keeps its row (it holds population) but its PI subtree
    is not expanded — the XATOM print_threshold analog.  Note the unit
    convention: fluence is in photons/cm^2 (1 a.u. = 1 photon/a0^2 =
    1/2.8002852e-17 cm^-2) and sigma in cm^2, so the product is
    dimensionless.  Auger edges cost no fluence and follow
    :func:`auger_triples` (over-inclusive w.r.t. zero-rate channels,
    which keeps the closure safe: never drops a reachable config, may
    keep a few extra).
    """
    from collections import deque

    by_key = {_key(c): c for c in configs}
    keep = {_key(base_config)}
    queue = deque([dict(base_config)])
    while queue:
        cfg = queue.popleft()
        if fluence * sigma_tot.get(_key(cfg), 0.0) >= threshold:
            for sh in cfg:                       # PI daughters
                d = _daughter_after_pi(cfg, sh)
                dk = _key(d)
                if dk in by_key and dk not in keep:
                    keep.add(dk)
                    queue.append(d)
        for hole, s1, s2 in auger_triples(cfg):   # Auger closure
            d = _daughter_after_auger(cfg, hole, s1, s2)
            dk = _key(d)
            if dk in by_key and dk not in keep:
                keep.add(dk)
                queue.append(d)
    return keep


def build_rate_table_parallel(Z: int, base_config: Config,
                             photon_energy_eV: float, *,
                             max_charge: int = 12,
                             n_workers: int | None = None,
                             correlation: str = "fac_frozen",
                             prune_fluence: float | None = None,
                             prune_threshold: float = 1e-3) -> dict:
    """Shell-skeleton-shared, prunable, parallel rate-table build (v4).

    Layers of docs/PERFORMANCE_PLAN.md, all measured on Ne @ 1 keV:

    * **0a skeleton sharing** — configs grouped by merged-(n,l) support;
      ONE solve per group per job where v3 solved every configuration.
      Oversized groups split to fill the pool; each split re-solves its
      skeleton (total solves <= n_jobs).
    * **0b F*sigma pruning** (opt-in via ``prune_fluence``) — a sigma
      pass over the complete enumeration runs first, then the kept set
      is the :func:`_prune_keep_set` closure; only kept configs get
      their (expensive) Auger trees.  Prune at the LARGEST fluence of
      the intended sweep so one table serves every fluence point.
    * **1a parallel** — group jobs over a process pool, BLAS pinned,
      numba cache isolated per process.

    Returns the v3 dict shape (``configs`` + ``failed``) plus
    ``skeleton`` / ``timings`` blocks and, when pruning, a ``prune``
    block (kept/total).
    """
    import os
    import time
    from concurrent.futures import ProcessPoolExecutor

    def _serial(c: Config) -> dict:
        return {f"{n},{k}": q for (n, k), q in c.items()}

    def _deserial(c: dict) -> Config:
        return {(int(k.split(",")[0]), int(k.split(",")[1])): q
                for k, q in c.items()}

    configs = enumerate_configs(base_config, max_charge=max_charge)
    cfg_dicts = [_serial(c) for c in configs]
    workers = n_workers or min(len(cfg_dicts), os.cpu_count() or 1)
    workers = max(1, min(workers, len(cfg_dicts)))

    def _run(groups: List[List[dict]], auger: bool):
        payload = [{"configs": g, "z": Z,
                    "photon_eV": photon_energy_eV,
                    "correlation": correlation, "auger": auger}
                   for g in groups]
        rows: List[dict] = []
        solves = 0
        executor = ProcessPoolExecutor(max_workers=workers)
        try:
            for res in executor.map(_rate_group_worker, payload):
                rows.extend(res["rows"])
                solves += res["solves"]
        finally:
            executor.shutdown(wait=True)
        return rows, solves

    timings: Dict[str, float] = {}
    prune_meta = None
    if prune_fluence is not None:
        t0 = time.perf_counter()
        rows, sig_solves = _run(_split_for_pool(
            _group_by_support(cfg_dicts), workers), auger=False)
        timings["sigma_pass_s"] = time.perf_counter() - t0
        sigma_tot: Dict[ConfigKey, float] = {}
        for row in rows:
            if "error" not in row:
                sigma_tot[_key(_deserial(row["occupations"]))] = sum(
                    row["sigma_cm2"].values())
        keep = _prune_keep_set(base_config, configs, sigma_tot,
                               prune_fluence, prune_threshold)
        prune_meta = {"fluence": prune_fluence,
                      "threshold": prune_threshold,
                      "kept": len(keep), "total": len(cfg_dicts),
                      "sigma_skeleton_solves": sig_solves}
        cfg_dicts = [c for c in cfg_dicts if _key(_deserial(c)) in keep]

    t0 = time.perf_counter()
    groups = _group_by_support(cfg_dicts)
    n_supports = len(groups)
    rows, solves = _run(_split_for_pool(groups, workers), auger=True)
    timings["build_pass_s"] = time.perf_counter() - t0
    table: Dict[str, dict] = {}
    failed: List[dict] = []
    for row in rows:
        if "error" in row:
            if row["label"] != "<worker>":
                failed.append(row)
            continue
        table[_table_key(_deserial(row["occupations"]))] = row
    out = {"photon_energy_eV": photon_energy_eV,
           "base_nelec": int(round(sum(base_config.values()))),
           "configs": table, "failed": failed,
           "skeleton": {
               "supports": n_supports, "solves": solves,
               "configs_per_solve": len(cfg_dicts) / max(solves, 1)},
           "timings": timings,
           "provenance": (
               "build_rate_table_parallel v4 2026-10-03: support-grouped "
               "shell-skeleton sharing (layer 0a) + grouped pool jobs "
               "(layer 1a); sigma_PI via shooting/Milne, Auger via the "
               "gated LS chain"
               + ("; F*sigma subtree prune (layer 0b)"
                  if prune_meta else ""))}
    if prune_meta is not None:
        out["prune"] = prune_meta
    return out


class TableRateProvider(RateProvider):
    """Read-only provider over a serialised rate table."""

    def __init__(self, table: dict):
        super().__init__(lambda label: None)
        self._table = table

    def _config_of(self, config: Config) -> str | None:
        want = {f"{n},{k}": q for (n, k), q in config.items()}
        for label, row in self._table["configs"].items():
            if row["occupations"] == want:
                return label
        return None

    def pi_sigmas_cm2(self, config, photon_energy_eV):
        label = self._config_of(config)
        if label is None:
            return {}
        row = self._table["configs"][label]
        return {tuple(int(x) for x in k.split(",")): v
                for k, v in row["sigma_cm2"].items()}

    def auger_channels(self, config):
        label = self._config_of(config)
        if label is None:
            return []
        out = []
        for ch in self._table["configs"][label]["auger"]:
            out.append((tuple(ch["hole"]), tuple(ch["s1"]),
                        tuple(ch["s2"]), ch["rate"]))
        return out
