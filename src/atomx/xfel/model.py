"""Native XFEL preparation: archived structure -> pulse rates -> resonant graph.

The configuration basis is explicit and must close under PI, fluorescence
and Auger. This configuration-average control reuses native Gu/Delta-SCF
structure; it does not silently substitute it for a level-resolved CI model.
"""

import copy
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import roots_legendre

from atomx.constants import BOHR_RADIUS_M, HARTREE_EV
from atomx.methods.dirac_fock import SCFResult
from atomx.numerics.bspline.evaluate import eval_P, eval_Q
from atomx.scattering.photoionization.subshell import subshell_photoionization
from atomx.xfel.atomic_rates import _state_path
from atomx.xfel.cascade import _table_key
from atomx.xfel.kinetics import KineticGraph
from atomx.xfel.native_channels import add_fluorescence, shakeoff_probabilities
from atomx.xfel.resonance import ResonantLine, add_resonances

_CM2 = (BOHR_RADIUS_M * 100) ** 2


def row_config(row):
    return {
        tuple(map(int, s.split(","))): q for s, q in row["occupations"].items() if q
    }


def _canonical_rows(table):
    rows = {}
    for row in table["configs"].values():
        sid = _table_key(row_config(row))
        configuration_weight(row_config(row))
        if not sid or sid in rows:
            raise ValueError("duplicate or bare configuration row in native table")
        rows[sid] = row
    return rows


def configuration_weight(config):
    """Number of independent jj magnetic determinants, prod binomial(g,q)."""
    weight = 1
    for (n, k), q in config.items():
        if (
            n < 1
            or k == 0
            or not np.isfinite(q)
            or q != int(q)
            or not 0 <= q <= 2 * abs(k)
        ):
            raise ValueError("invalid integer jj occupation")
        weight *= math.comb(2 * abs(k), int(q))
    return weight


class NativeRateArchive:
    """Public, lazy, hash-checked SCF loader for a native rate-table archive."""

    def __init__(self, table_path, checkpoint_dir):
        self.table_path = Path(table_path)
        self.table = json.loads(self.table_path.read_text())
        self.checkpoint_dir = Path(checkpoint_dir)
        self.identity = self.table["provenance"]["identity"]
        self._rows = _canonical_rows(self.table)
        self._cached = lru_cache(maxsize=len(self._rows))(self._load)

    def _load(self, sid):
        if sid not in self._rows:
            raise ValueError("configuration is absent from the native archive")
        config = row_config(self._rows[sid])
        stem = _state_path(self.checkpoint_dir, config)
        metadata = json.loads(stem.with_suffix(".json").read_text())
        data = stem.with_suffix(".h5")
        if (
            metadata["identity"] != self.identity
            or not metadata["converged"]
            or hashlib.sha256(data.read_bytes()).hexdigest() != metadata["h5_sha256"]
        ):
            raise ValueError("SCF archive identity/hash/convergence mismatch")
        result = SCFResult.load(str(data))
        _check_result(result, self._rows[sid])
        if self.identity["controls"]["Z"] != result.Z:
            raise ValueError("SCF nuclear charge differs from the archive")
        return result

    def result(self, config):
        return self._cached(_table_key(config))

    def results_by_state(self):
        return {"": None, **{sid: self._cached(sid) for sid in self._rows}}


def _check_result(result, row):
    if not result.converged or not np.isclose(
        result.total_energy, row["energy_hartree"], rtol=0, atol=1e-9
    ):
        raise ValueError("native result energy/convergence differs from table")
    actual = {(s.n, s.kappa): s.occupation for s in result.subshells if s.occupation}
    if actual != row_config(row):
        raise ValueError("native result occupations differ from table")


def gaussian_spectral_quadrature(
    energy_eV, bandwidth_fwhm_eV, breakpoints=(), *, order=32
):
    """Split a Gaussian at physical thresholds; return energies and unit weights.

    Nine standard deviations truncate less than 3e-19 spectral mass. The
    finite positive-energy domain is checked rather than renormalized.
    """
    if (
        not np.isfinite([energy_eV, bandwidth_fwhm_eV]).all()
        or energy_eV <= 0
        or bandwidth_fwhm_eV < 0
        or order < 8
        or int(order) != order
    ):
        raise ValueError("invalid photon spectrum or quadrature order")
    if bandwidth_fwhm_eV == 0:
        return np.array([energy_eV]), np.array([1.0])
    sigma = bandwidth_fwhm_eV / np.sqrt(8 * np.log(2))
    a, b = energy_eV - 9 * sigma, energy_eV + 9 * sigma
    if a <= 0:
        raise ValueError("Gaussian photon spectrum extends to nonpositive energies")
    cuts = np.array(list(breakpoints), dtype=float)
    if not np.isfinite(cuts).all():
        raise ValueError("spectral breakpoints must be finite")
    edges = np.unique(
        np.r_[
            a,
            energy_eV - 3 * sigma,
            energy_eV,
            energy_eV + 3 * sigma,
            cuts[(cuts > a) & (cuts < b)],
            b,
        ]
    )
    x, w = roots_legendre(int(order))
    energies, weights = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        e = (hi + lo) / 2 + (hi - lo) * x / 2
        weights.append(
            w
            * (hi - lo)
            / 2
            * np.exp(-0.5 * ((e - energy_eV) / sigma) ** 2)
            / (sigma * np.sqrt(2 * np.pi))
        )
        energies.append(e)
    energy_nodes, photon_weights = np.concatenate(energies), np.concatenate(weights)
    if not np.isclose(photon_weights.sum(), 1, rtol=0, atol=1e-10):
        raise RuntimeError("Gaussian quadrature does not resolve photon probability")
    return energy_nodes, photon_weights


def retabulate_native_pi(
    table,
    result_loader,
    *,
    photon_energy_eV,
    bandwidth_fwhm_eV,
    continuum_model,
    shakeoff=False,
    quadrature_order=32,
):
    """Pulse-spectrum averaged native PI and optional exclusive single shake.

    Initial orbitals, native daughter thresholds and true cross sections
    are used at each quadrature energy. Shake thresholds split the energy
    grid too; center-energy gating is never applied to a broadband pulse.
    Auger is reused from the validated input table; fitted direct double
    cannot be silently resampled or mixed with overlap shake.
    """
    out = copy.deepcopy(table)
    if any(r.get("double_pi") for r in out["configs"].values()):
        raise ValueError(
            "start from an unsplit PI table; existing double PI cannot be resampled"
        )
    rows = _canonical_rows(out)
    out["configs"] = rows
    energies = {sid: r["energy_hartree"] for sid, r in rows.items()}
    energies[""] = 0.0

    @lru_cache(maxsize=len(rows))
    def load(sid):
        result = result_loader(row_config(rows[sid]))
        _check_result(result, rows[sid])
        return result

    for sid, row in rows.items():
        config = row_config(row)
        initial = load(sid)
        thresholds, branches = {}, {}
        for shell in config:
            daughter = dict(config)
            daughter[shell] -= 1
            dsid = _table_key(daughter)
            if dsid not in energies:
                raise ValueError(
                    "missing PI daughter in the native configuration basis"
                )
            threshold = (energies[dsid] - energies[sid]) * HARTREE_EV
            if threshold <= 0:
                raise ValueError(
                    "PI of an unbound configuration needs a continuum-state model"
                )
            label = next(s.label for s in initial.subshells if (s.n, s.kappa) == shell)
            thresholds[label] = threshold
            if not shakeoff or not dsid:
                continue
            final = load(dsid)
            cfinal = row_config(rows[dsid])
            bd, bdf = initial._bsp_data, final._bsp_data
            if not np.array_equal(bd["t"], bdf["t"]):
                raise ValueError("shake overlap requires the same radial basis")
            iw, fw = initial.orbitals_by_subshell(), final.orbitals_by_subshell()
            overlaps, secondary_thresholds = {}, {}
            for spectator in cfinal:
                overlaps[spectator] = float(
                    np.sum(
                        bd["grw"]
                        * (
                            eval_P(iw[spectator][0], bd) * eval_P(fw[spectator][0], bd)
                            + eval_Q(iw[spectator][1], bd)
                            * eval_Q(fw[spectator][1], bd)
                        )
                    )
                )
                double = dict(cfinal)
                double[spectator] -= 1
                target = _table_key(double)
                if target not in energies:
                    raise ValueError("missing shake daughter in configuration basis")
                secondary_thresholds[spectator] = (
                    energies[target] - energies[dsid]
                ) * HARTREE_EV
            branches[shell] = (cfinal, overlaps, secondary_thresholds, threshold)
        cuts = list(thresholds.values())
        for _c, _o, secondary, t in branches.values():
            cuts.extend(t + v for v in secondary.values())
        photons, weights = gaussian_spectral_quadrature(
            photon_energy_eV, bandwidth_fwhm_eV, cuts, order=quadrature_order
        )
        pi = subshell_photoionization(
            initial, photons, continuum_model=continuum_model, thresholds_eV=thresholds
        )
        singles, doubles = {}, []
        for shell in config:
            label = next(s.label for s in initial.subshells if (s.n, s.kappa) == shell)
            # The public PI provider already includes the orbital occupation.
            sigma = pi.sigma_by_orbital_a0_sq[label] * _CM2
            average = float(weights @ sigma)
            if shell in branches:
                cfinal, overlaps, secondary, t = branches[shell]
                probs = {s: np.zeros(len(photons)) for s in cfinal}
                for j, e in enumerate(photons):
                    if e < t:
                        continue
                    p = shakeoff_probabilities(
                        cfinal,
                        overlaps,
                        available_energy_eV=e - t,
                        thresholds_eV=secondary,
                        provenance="native sudden overlap, energy resolved",
                    )
                    for spectator, value in p.items():
                        probs[spectator][j] = value
                for spectator, p in probs.items():
                    value = float(weights @ (sigma * p))
                    average -= value
                    if value:
                        doubles.append(
                            dict(
                                removed=[list(shell), list(spectator)],
                                sigma_cm2=value,
                                provenance="native Dirac overlap; spectrum-resolved Delta-SCF gate",
                            )
                        )
            if average < 0 or not np.isfinite(average):
                raise RuntimeError("invalid exclusive PI spectral average")
            singles[f"{shell[0]},{shell[1]}"] = average
        row["sigma_cm2"] = singles
        row["thresholds_eV"] = thresholds
        if shakeoff:
            row["double_pi"] = doubles
    out["photon_energy_eV"] = float(photon_energy_eV)
    out["provenance"]["continuum_bandwidth_fwhm_eV"] = float(bandwidth_fwhm_eV)
    out["provenance"]["pi_spectral_model"] = (
        f"Gaussian, 9sigma, threshold-split Legendre{quadrature_order}"
    )
    if shakeoff:
        out["provenance"]["shakeoff_model"] = (
            "single PI shake, spectrum-resolved native threshold"
        )
    return out


def configuration_resonances(table):
    """Reverse native E1 edges with configuration weights and both lifetimes.

    No empirical line strengths are introduced. The same upper-state
    radial matrix element sets emission and absorption by Einstein balance.
    The caller must supply a fluorescence-closed, explicitly chosen basis.
    """
    rows = _canonical_rows(table)
    weights = {sid: configuration_weight(row_config(row)) for sid, row in rows.items()}
    widths = {
        sid: sum(c["rate"] for c in row["auger"])
        + sum(c["rate"] for c in row.get("radiative", []))
        for sid, row in rows.items()
    }
    lines = []
    for upper, row in rows.items():
        for ch in row.get("radiative", []):
            config = row_config(row)
            config[tuple(ch["hole"])] = config.get(tuple(ch["hole"]), 0) + 1
            config[tuple(ch["source"])] -= 1
            lower = _table_key(config)
            if lower not in rows:
                raise ValueError("missing lower resonance configuration")
            energy = (
                row["energy_hartree"] - rows[lower]["energy_hartree"]
            ) * HARTREE_EV
            if not np.isclose(energy, ch["energy_eV"], rtol=0, atol=1e-8):
                raise ValueError("E1 line energy differs from native state energies")
            if ch["rate"]:
                lines.append(
                    ResonantLine.from_einstein_A(
                        lower,
                        upper,
                        energy,
                        ch["rate"],
                        total_decay_per_s=widths[upper],
                        lower_decay_per_s=widths[lower],
                        g_lower=weights[lower],
                        g_upper=weights[upper],
                        provenance="native configuration-average E1, Einstein reverse, Delta-SCF",
                    )
                )
    return tuple(lines)


class NativeXFELModel:
    """Production native configuration model for a fixed pulse energy/bandwidth."""

    def __init__(
        self,
        table,
        base_config,
        result_loader,
        *,
        continuum_model,
        photon_energy_eV,
        bandwidth_fwhm_eV=0.0,
        resonances=True,
        shakeoff=False,
        initial_config=None,
        quadrature_order=32,
    ):
        self.result_loader = result_loader
        self.continuum_model = continuum_model
        table = copy.deepcopy(table)
        if initial_config is not None:
            if _table_key(initial_config) not in {
                _table_key(row_config(r)) for r in table["configs"].values()
            }:
                raise ValueError("initial configuration is absent from the rate table")
            nelec = sum(initial_config.values())
            table["configs"] = {
                key: row
                for key, row in table["configs"].items()
                if sum(row["occupations"].values()) <= nelec
            }
        self.table = retabulate_native_pi(
            table,
            result_loader,
            photon_energy_eV=photon_energy_eV,
            bandwidth_fwhm_eV=bandwidth_fwhm_eV,
            continuum_model=continuum_model,
            shakeoff=shakeoff,
            quadrature_order=quadrature_order,
        )
        if not all("radiative" in r for r in self.table["configs"].values()):
            self.table = add_fluorescence(self.table, result_loader)
        self.lines = configuration_resonances(self.table)
        graph = KineticGraph.from_rate_table(
            self.table, base_config, initial_config=initial_config
        )
        self.graph = (
            add_resonances(graph, self.lines, bandwidth_fwhm_eV=bandwidth_fwhm_eV)
            if resonances
            else graph
        )
        self._observable_cache = {}

    @classmethod
    def from_archive(cls, archive, base_config, **kwargs):
        return cls(archive.table, base_config, archive.result, **kwargs)

    def solve(self, pulse, **kwargs):
        return self.graph.solve(pulse, **kwargs)

    def coherent_one_electron(self, *, photon_numbers):
        """Resolve one-electron magnetic states for explicit coherent driving.

        Photon ladders are supplied explicitly; averaged multielectron
        configurations cannot supply a phase-resolved Hamiltonian.
        """
        from atomx.xfel.coherent import NativeOneElectronCoherentModel

        return NativeOneElectronCoherentModel(self, photon_numbers=photon_numbers)

    def scattering_bank(
        self,
        q_a0_inv,
        *,
        dispersion_grid_eV,
        spectral_order=32,
        bound_bound=True,
        cache=True,
    ):
        from atomx.xfel.observables import build_native_scattering_bank

        q = np.array(q_a0_inv, dtype=float, copy=True)
        energy_grid = np.array(dispersion_grid_eV, dtype=float, copy=True)
        key = (
            q.shape,
            q.tobytes(),
            energy_grid.shape,
            energy_grid.tobytes(),
            spectral_order,
            bound_bound,
        )
        if cache and key in self._observable_cache:
            return self._observable_cache[key]
        bank = build_native_scattering_bank(
            self,
            q,
            dispersion_grid_eV=energy_grid,
            spectral_order=spectral_order,
            bound_bound=bound_bound,
        )
        if cache:
            if len(self._observable_cache) >= 4:
                self._observable_cache.pop(next(iter(self._observable_cache)))
            self._observable_cache[key] = bank
        return bank

    def clear_observable_cache(self):
        """Release cached spectral moments; a prepared atomic model is fixed."""
        self._observable_cache.clear()

    def run(
        self,
        pulse,
        *,
        q_a0_inv=None,
        dispersion_grid_eV=None,
        spectral_order=32,
        bound_bound=True,
        **solve_options,
    ):
        """Run populations and optional state/spectrum averaged atomic imaging."""
        from atomx.xfel.observables import XFELSimulation

        if (q_a0_inv is None) != (dispersion_grid_eV is None):
            raise ValueError(
                "imaging requires q and a continuum dispersion energy grid together"
            )
        dynamics = self.solve(pulse, **solve_options)
        bank = (
            None
            if q_a0_inv is None
            else self.scattering_bank(
                q_a0_inv,
                dispersion_grid_eV=dispersion_grid_eV,
                spectral_order=spectral_order,
                bound_bound=bound_bound,
            )
        )
        return XFELSimulation(
            dynamics,
            None if bank is None else bank.observe(dynamics, pulse),
            bank,
            dict(self.graph.provenance),
        )
