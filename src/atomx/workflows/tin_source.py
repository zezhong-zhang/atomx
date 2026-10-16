"""Bind the native tin CI line/level evidence to the reduced plasma source.

No large CI matrices are loaded. The partition includes every producer level,
while radiative coverage follows the producer's explicit endpoint window.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from atomx.plasma.euv_source import LTELinePlasma


def load_tin_lte_plasma(run_directory, *, bin_width_eV, fwhm_eV,
                        memory_cap_bytes=2 * 1024**3):
    """Load a complete native tin result, stream all saved lines and hash inputs.

    Refuses altered counts, missing endpoints, inconsistent photon gaps or a
    grid/model exceeding its cap. Assumes the native producer's documented
    one-object-per-line JSON layout, not arbitrary external spectrum formats.
    """
    root = Path(run_directory)
    if not np.isfinite(bin_width_eV) or bin_width_eV <= 0 or not np.isfinite(fwhm_eV) or fwhm_eV < 0:
        raise ValueError("positive finite bin width and nonnegative FWHM required")
    hashes = {}

    def read(name):
        raw = (root / name).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    result = read("result.json")
    producer = read("input.json")
    expected = {"Z": 50, "charge": 12, "nucleus": "120Sn Fermi",
                "operator": "Dirac-Coulomb", "E1": "length long_wavelength"}
    if any(producer.get(key) != value for key, value in expected.items()):
        raise ValueError("tin loader requires the documented Sn12+ 120Sn bare-DC length-E1 identity")
    n_levels, n_lines = result["levels"], result["lines"]
    bound = 320 * n_lines + 4096 * n_levels
    if bound > memory_cap_bytes:
        raise MemoryError(f"loader planning bound {bound} exceeds cap {memory_cap_bytes}; no data removed")
    levels = read("levels.json")
    if len(levels) != n_levels or result.get("source_unchanged") is not True:
        raise ValueError("producer count/source certification failed")
    by_id = {level["id"]: i for i, level in enumerate(levels)}
    if len(by_id) != n_levels:
        raise ValueError("duplicate CI level identities")
    energies = np.array([r["excitation_eV"] for r in levels])
    weights = np.array([r["J2"] + 1 for r in levels])
    folds = np.array([r["dominant_fold"] for r in levels])
    upper = np.empty(n_lines, dtype=np.int32)
    lower = np.empty(n_lines, dtype=np.int32)
    rates = np.empty(n_lines)
    endpoints = np.zeros(n_levels, dtype=bool)
    digest = hashlib.sha256()
    count = 0
    maximum_gap_defect = 0.0
    with (root / "lines.json").open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            record = raw.strip().rstrip(b",")
            if record in (b"[", b"]", b""):
                continue
            if count >= n_lines:
                raise ValueError("line count exceeds certified producer count")
            line = json.loads(record)
            u, l = by_id[line["upper"]], by_id[line["lower"]]
            maximum_gap_defect = max(maximum_gap_defect,
                                    abs(energies[u] - energies[l] - line["energy_eV"]))
            upper[count], lower[count], rates[count] = u, l, line["A_per_s"]
            endpoints[[u, l]] = True
            count += 1
    if count != n_lines or maximum_gap_defect > 2e-9:
        raise ValueError("saved line count or photon gap disagrees with native CI levels")
    hashes["lines.json"] = digest.hexdigest()
    source_id = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    max_gap = float(np.max(energies[upper] - energies[lower], initial=0))
    max_energy = (np.ceil((max_gap + 7 * fwhm_eV + bin_width_eV) / bin_width_eV)
                  * bin_width_eV)
    bins = int(round(max_energy / bin_width_eV))
    if bins > 1000000:
        raise MemoryError("more than one million energy bins requested; no automatic coarsening")
    scope = "native saved E1 endpoint window; omitted endpoints/continua do not radiate in this model"
    plasma = LTELinePlasma(energies, weights, upper, lower, rates, charge=12,
                           ion_mass_amu=120, energy_edges_eV=np.arange(bins + 1) * bin_width_eV,
                           fwhm_eV=fwhm_eV, source_id=source_id, emission_scope=scope,
                           memory_cap_bytes=memory_cap_bytes)
    metadata = dict(producer_directory=str(root.resolve()), sha256=hashes, source_id=source_id,
                    levels=n_levels, lines=n_lines, radiative_endpoint_levels=int(endpoints.sum()),
                    loader_bound_bytes=bound, table_bound_bytes=plasma.construction_bound_bytes,
                    bins=bins, bin_width_eV=bin_width_eV, fwhm_eV=fwhm_eV,
                    maximum_endpoint_gap_defect_eV=maximum_gap_defect,
                    producer_input=producer, producer_physics=expected,
                    charge_state="Sn12+", population_closure="fixed-charge LTE over ALL supplied levels")
    return plasma, metadata, endpoints, folds
