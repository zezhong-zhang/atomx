"""Strict ingestion of externally sourced direct-double PI cross sections.

No invented fit is bundled. Missing source data, extrapolation, or
coexistence with shakeoff for the same state is an explicit error.
"""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from atomx.constants import HARTREE_EV
from atomx.xfel.cascade import _table_key
from atomx.xfel.table_mc import _daughter, _occupation_key, _parse_shell


def add_direct_double_ionization(table, data_path):
    if table.get("provenance", {}).get("continuum_bandwidth_fwhm_eV", 0) != 0:
        raise ValueError(
            "direct-double ingestion requires a monochromatic table; spectral averaging is not implicit"
        )
    path = Path(data_path)
    data = json.loads(path.read_text())
    if not data.get("source") or data.get("units") != "cm2" or not data.get("channels"):
        raise ValueError("double-ionization data needs source, cm2 units and channels")
    out = copy.deepcopy(table)
    rows = {
        _table_key({_parse_shell(s): q for s, q in r["occupations"].items()}): r
        for r in out["configs"].values()
    }
    energies = {k: r["energy_hartree"] for k, r in rows.items()}
    energies[""] = 0.0
    existing = {sid for sid, row in rows.items() if row.get("double_pi")}
    seen = set()
    for ch in data["channels"]:
        sid = ch["initial_state"]
        if sid not in rows or sid in existing:
            raise ValueError("missing state or existing double/shakeoff channel")
        if len(ch["removed"]) != 2:
            raise ValueError("direct double ionization removes exactly two electrons")
        shells = tuple(tuple(s) for s in ch["removed"])
        token = (sid, tuple(sorted(shells)))
        if token in seen:
            raise ValueError("duplicate direct-double channel")
        seen.add(token)
        c = {_parse_shell(s): q for s, q in rows[sid]["occupations"].items()}
        daughter = _daughter(_occupation_key(c), [(s, -1) for s in shells])
        target = _table_key({(n, k): q for n, k, q in daughter})
        if target not in energies:
            raise ValueError("missing direct-double daughter")
        threshold = (energies[target] - energies[sid]) * HARTREE_EV
        grid = np.asarray(ch["energies_eV"], dtype=float)
        sigma = np.asarray(ch["sigma_cm2"], dtype=float)
        if (
            grid.ndim != 1
            or len(grid) < 2
            or sigma.shape != grid.shape
            or not np.isfinite(grid).all()
            or not np.isfinite(sigma).all()
            or np.any(np.diff(grid) <= 0)
            or np.any(grid <= 0)
            or np.any(sigma < 0)
        ):
            raise ValueError("invalid direct-double cross-section table")
        photon = table["photon_energy_eV"]
        if photon < grid[0] or photon > grid[-1]:
            raise ValueError("direct-double data does not cover photon energy")
        value = float(np.interp(photon, grid, sigma))
        if photon <= threshold and value > 0:
            raise ValueError("nonzero double ionization below native energy threshold")
        rows[sid].setdefault("double_pi", []).append(
            dict(
                removed=[list(s) for s in shells],
                sigma_cm2=value,
                provenance=data["source"],
                normalization="additional direct-double cross section",
            )
        )
    out["provenance"]["direct_double_data_sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    out["provenance"]["direct_double_source"] = data["source"]
    return out
