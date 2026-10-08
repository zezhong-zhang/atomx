"""Native fluorescence and sudden-overlap shakeoff on a solved jj graph."""

import copy

import numpy as np

from atomx.constants import HARTREE_EV
from atomx.numerics.bspline.evaluate import eval_P, eval_Q
from atomx.properties.transitions.angular import angular_eL
from atomx.properties.transitions.rates import einstein_A_eL
from atomx.xfel.cascade import _table_key


def _config(row):
    return {tuple(map(int, k.split(","))): q for k, q in row["occupations"].items()}


def add_fluorescence(table, result_loader):
    """E1 native Delta-SCF channels with source q and jj vacancy/g weights."""
    out = copy.deepcopy(table)
    energies = {
        _table_key(_config(r)): r["energy_hartree"] for r in out["configs"].values()
    }
    by_config = {_table_key(_config(r)): r for r in out["configs"].values()}
    for row in out["configs"].values():
        if row.get("radiative"):
            raise ValueError("fluorescence table already has radiative channels")
        c = _config(row)
        result = result_loader(c)
        if not np.isclose(
            result.total_energy, row["energy_hartree"], rtol=0, atol=1e-9
        ):
            raise ValueError("SCF result does not match rate-table energy")
        stored = {
            (s.n, s.kappa): s.occupation for s in result.subshells if s.occupation
        }
        if stored != c:
            raise ValueError("SCF occupations do not match rate-table state")
        bd = result._bsp_data
        bound = result.orbitals_by_subshell()
        waves = {
            s: (eval_P(v[0], bd).ravel(), eval_Q(v[1], bd).ravel())
            for s, v in bound.items()
        }
        w, r = bd["grw"].ravel(), bd["gr"].ravel()
        channels = []
        for h in bound:
            vacancies = 2 * abs(h[1]) - c.get(h, 0)
            if vacancies <= 0:
                continue
            for a, q in c.items():
                angular = angular_eL(a[1], h[1], 1)
                if a == h or angular == 0:
                    continue
                d = dict(c)
                d[h] = d.get(h, 0) + 1
                d[a] -= 1
                key = _table_key(d)
                if key not in by_config:
                    raise ValueError("missing fluorescence daughter")
                energy = (row["energy_hartree"] - energies[key]) * HARTREE_EV
                if energy <= 0:
                    continue
                radial = np.sum(
                    w * r * (waves[a][0] * waves[h][0] + waves[a][1] * waves[h][1])
                )
                rate = einstein_A_eL(
                    energy / HARTREE_EV, (angular * radial) ** 2, 1, 2 * abs(a[1]) - 1
                )
                rate *= q * vacancies / (2 * abs(h[1]))
                if not np.isfinite(rate) or rate < 0:
                    raise RuntimeError("invalid fluorescence rate")
                channels.append(
                    dict(
                        hole=list(h),
                        source=list(a),
                        rate=float(rate),
                        energy_eV=float(energy),
                    )
                )
        row["radiative"] = channels
    out["provenance"]["radiation"] = True
    out["provenance"]["radiation_model"] = "native E1; q_source * jj vacancy / g_hole"
    return out


def shakeoff_probabilities(
    occupations, overlaps, *, available_energy_eV, thresholds_eV, provenance
):
    """XATOM shakeoff.f90: single spectator q*(1-|overlap|^2), energy gated.

    First-order exclusive single-shake model. If summed branching exceeds
    unity this approximation is invalid and raises, instead of clipping.
    Bound excitation is not subtracted from 1-overlap^2, matching XATOM's
    specific sudden-overlap approximation, not a full continuum projection.
    """
    if (
        not provenance
        or not np.isfinite(available_energy_eV)
        or available_energy_eV < 0
    ):
        raise ValueError("shakeoff needs nonnegative available energy and provenance")
    out = {}
    for s, q in occupations.items():
        if q < 0 or q != int(q) or q > 2 * abs(s[1]):
            raise ValueError("invalid spectator occupation")
        if q == 0:
            continue
        if s not in overlaps or s not in thresholds_eV:
            raise ValueError("missing spectator overlap or threshold")
        overlap = overlaps[s]
        threshold = thresholds_eV[s]
        if (
            not np.isfinite(overlap)
            or abs(overlap) > 1 + 1e-10
            or not np.isfinite(threshold)
            or threshold <= 0
        ):
            raise ValueError("invalid overlap norm or shakeoff threshold")
        if available_energy_eV >= threshold:
            out[s] = float(q * max(0, 1 - abs(overlap) ** 2))
    if sum(out.values()) > 1 + 1e-10:
        raise ValueError("single-shake branching sum exceeds unity")
    return out


def add_photo_shakeoff(table, result_loader):
    """Split each PI cross section into exclusive single/double removals.

    Uses initial and primary-daughter radial overlaps; residual electron
    energy is gated by the native double-removal energy threshold. All
    double daughters must already be present; total absorption is conserved.
    """
    if table.get("provenance", {}).get("continuum_bandwidth_fwhm_eV", 0) != 0:
        raise ValueError(
            "broadband shakeoff requires retabulate_native_pi(shakeoff=True)"
        )
    out = copy.deepcopy(table)
    rows = {_table_key(_config(r)): r for r in out["configs"].values()}
    energies = {k: r["energy_hartree"] for k, r in rows.items()}
    energies[""] = 0.0
    results = {}

    def load(c):
        key = _table_key(c)
        if key not in results:
            r = result_loader(c)
            if not np.isclose(r.total_energy, energies[key], rtol=0, atol=1e-9):
                raise ValueError("shakeoff SCF energy differs from rate table")
            stored = {(s.n, s.kappa): s.occupation for s in r.subshells if s.occupation}
            if stored != c:
                raise ValueError("shakeoff SCF occupations differ from rate table")
            results[key] = r
        return results[key]

    for key, row in rows.items():
        if row.get("double_pi"):
            raise ValueError(
                "shakeoff must not be applied twice or mixed with direct double PI"
            )
        c = _config(row)
        initial = load(c)
        bd = initial._bsp_data
        initial_waves = initial.orbitals_by_subshell()
        doubles = []
        for text, sigma in list(row["sigma_cm2"].items()):
            if sigma == 0:
                continue
            a = tuple(map(int, text.split(",")))
            d = dict(c)
            d[a] -= 1
            d = {s: q for s, q in d.items() if q}
            if not d:
                continue
            final = load(d)
            bdf = final._bsp_data
            if not np.array_equal(bd["t"], bdf["t"]):
                raise ValueError("shakeoff overlap requires a shared radial basis")
            overlaps, thresholds = {}, {}
            fw = final.orbitals_by_subshell()
            for s in d:
                overlaps[s] = np.sum(
                    bd["grw"]
                    * (
                        eval_P(initial_waves[s][0], bd) * eval_P(fw[s][0], bdf)
                        + eval_Q(initial_waves[s][1], bd) * eval_Q(fw[s][1], bdf)
                    )
                )
                dd = dict(d)
                dd[s] -= 1
                dk = _table_key(dd)
                if dk not in energies:
                    raise ValueError("missing shakeoff daughter")
                thresholds[s] = (energies[dk] - energies[_table_key(d)]) * HARTREE_EV
            available = (
                table["photon_energy_eV"]
                - (energies[_table_key(d)] - energies[key]) * HARTREE_EV
            )
            prob = shakeoff_probabilities(
                d,
                overlaps,
                available_energy_eV=available,
                thresholds_eV=thresholds,
                provenance="XATOM sudden orbital-overlap transcription; native Gu radials",
            )
            row["sigma_cm2"][text] = sigma * (1 - sum(prob.values()))
            for s, p in prob.items():
                if p:
                    doubles.append(
                        dict(
                            removed=[list(a), list(s)],
                            sigma_cm2=sigma * p,
                            probability=p,
                            provenance="XATOM shakeoff.f90 single-overlap; native Delta-SCF gate",
                        )
                    )
        row["double_pi"] = doubles
    out["provenance"]["shakeoff_model"] = (
        "exclusive single shake from sudden Dirac orbital overlap"
    )
    return out
