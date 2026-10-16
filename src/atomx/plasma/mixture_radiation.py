"""Bind explicit charge partitions, line tables and continuum data to transfer.

No atomic level, collision strength, PI edge or line from a missing charge is
invented. Individual native line tables retain their separate source identity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable

import numpy as np

from atomx.spectrum.lte_continuum import LTEBoundFreeEdge, lte_continuum_coefficients
from atomx.spectrum.slab_transfer import layered_transfer


@dataclass(frozen=True)
class MixtureBoundFreeEdge:
    charge: int
    bound_level: int
    next_charge_level: int
    cross_section_cm2: Callable[[float], float]
    source_id: str


class MixtureRadiation:
    """A Saha EOS and one explicitly matched radiative table per charge.

    The continuum mode is mandatory: 'none', 'free-free', or
    'free-free+bound-free'. The Gaunt provider/source is explicit when needed.
    This class supplies stationary cells/profiles; it does not solve kinetics
    or infer a hydrodynamic preparation from observed spectra.
    """

    def __init__(self, eos, line_tables, *, continuum_mode, nuclear_source_id,
                 free_free_gaunt=None, gaunt_source_id=None, bound_free_edges=(),
                 bound_free_coverage=None):
        self.eos = eos
        self.tables = dict(line_tables)
        if set(self.tables) != set(eos.charges) or not nuclear_source_id:
            raise ValueError("one explicit native line table per EOS charge and common nucleus identity required")
        if continuum_mode not in ("none", "free-free", "free-free+bound-free"):
            raise ValueError("explicit supported continuum mode required")
        if continuum_mode != "none" and (not callable(free_free_gaunt) or not gaunt_source_id):
            raise ValueError("explicit Gaunt physics and source required")
        self.edgespecs = tuple(bound_free_edges)
        if continuum_mode == "free-free+bound-free" and (not self.edgespecs or not bound_free_coverage):
            raise ValueError("explicit PI edges and their physical coverage required")
        if continuum_mode != "free-free+bound-free" and self.edgespecs:
            raise ValueError("PI edges supplied but bound-free continuum disabled")
        first = self.tables[eos.charges[0]]
        for partition in eos.partitions:
            table = self.tables[partition.charge]
            if (table.charge != partition.charge or table.source_id != partition.source_id
                    or not np.array_equal(table.energy, partition.energies_eV)
                    or not np.array_equal(table.g, partition.statistical_weights)
                    or not np.array_equal(table.edges, first.edges) or table.mass_kg != first.mass_kg):
                raise ValueError("EOS/line energy, metric weight, source, grid or nuclear mass mismatch")
        self.edge_indices = []
        for edge in self.edgespecs:
            if any(not isinstance(v, (int, np.integer)) for v in
                   (edge.charge, edge.bound_level, edge.next_charge_level)):
                raise ValueError("PI charge and level indices must be integers in the declared states")
            index = int(edge.charge - eos.charges[0])
            if (index < 0 or index + 1 >= len(eos.slices) or edge.bound_level < 0
                    or edge.next_charge_level < 0 or not edge.source_id or not callable(edge.cross_section_cm2)):
                raise ValueError("PI edge must connect explicit neighboring charge states")
            lower_slice, upper_slice = eos.slices[index:index+2]
            if (edge.bound_level >= lower_slice.stop - lower_slice.start
                    or edge.next_charge_level >= upper_slice.stop - upper_slice.start):
                raise ValueError("PI edge references a level outside the declared partitions")
            lower, upper = lower_slice.start + edge.bound_level, upper_slice.start + edge.next_charge_level
            threshold = eos.energy[upper] - eos.energy[lower]
            if threshold <= 0:
                raise ValueError("nonpositive PI threshold: autoionization/resonance treatment required")
            self.edge_indices.append((lower, float(threshold)))
        self.continuum_mode = continuum_mode
        self.free_free_gaunt = free_free_gaunt
        self.gaunt_source_id = gaunt_source_id
        self.mass_kg = first.mass_kg
        self.edges, self.centers = first.edges, first.centers
        self.energy = eos.energy
        self.band_weights = first.band_weights
        identity = dict(charges=[int(q) for q in eos.charges], nuclear_source_id=nuclear_source_id,
                        eos_source_id=eos.source_id, ion_mass_kg=self.mass_kg,
                        tables={str(q): t.source_id for q, t in self.tables.items()},
                        thresholds=eos.threshold_source_id, continuum_mode=continuum_mode,
                        gaunt_source_id=gaunt_source_id, PI=[e.source_id for e in self.edgespecs],
                        bound_free_coverage=bound_free_coverage)
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
        digest.update(np.asarray(first.edges, dtype="<f8").tobytes())
        for charge in eos.charges:
            table = self.tables[charge]
            digest.update(np.asarray([table.fwhm_eV], dtype="<f8").tobytes())
            for array in (table.upper, table.lower):
                digest.update(np.asarray(array, dtype="<i8").tobytes())
            digest.update(np.asarray(table.A_per_s, dtype="<f8").tobytes())
        self.source_id = digest.hexdigest()
        self.identity = identity
        self.emission_scope = "explicit finite multi-charge line tables and declared LTE continuum coverage"

    def cell(self, temperature_eV, ion_density_per_m3):
        state = self.eos.equilibrium(temperature_eV, ion_density_per_m3)
        eps, alpha = np.zeros_like(self.centers), np.zeros_like(self.centers)
        for charge, section, fraction in zip(self.eos.charges, self.eos.slices,
                                              state["charge_fractions"], strict=True):
            if fraction == 0:  # Only an exact underflow/zero; no weak-charge screening.
                continue
            e, a = self.tables[charge].coefficients(state["populations"][section] / fraction,
                                                   ion_density_per_m3 * fraction)
            eps += e
            alpha += a
        if self.continuum_mode != "none":
            edges = [LTEBoundFreeEdge(threshold, ion_density_per_m3 * state["populations"][level],
                                     spec.cross_section_cm2, spec.source_id)
                     for spec, (level, threshold) in zip(self.edgespecs, self.edge_indices, strict=True)]
            continuum = lte_continuum_coefficients(self.edges, temperature_eV,
                state["electron_density_per_m3"], ion_density_per_m3, state["mean_charge_squared"],
                free_free_gaunt=self.free_free_gaunt, gaunt_source_id=self.gaunt_source_id,
                bound_free_edges=edges)
            eps += continuum["emissivity_W_m3_bin"]
            alpha += continuum["absorption_per_m"]
        return dict(equilibrium=state, emissivity_W_m3_bin=eps, absorption_per_m=alpha)

    def profile(self, temperature_eV, ion_density_per_m3, cell_widths_m, *, angular_order=32):
        temperatures, densities = np.asarray(temperature_eV), np.asarray(ion_density_per_m3)
        if temperatures.ndim != 1 or temperatures.shape != densities.shape:
            raise ValueError("one temperature and ion density per spatial cell required")
        cells = [self.cell(T, ni) for T, ni in zip(temperatures, densities, strict=True)]
        transfer = layered_transfer(np.array([c["emissivity_W_m3_bin"] for c in cells]),
                                     np.array([c["absorption_per_m"] for c in cells]), cell_widths_m,
                                     angular_order=angular_order)
        return dict(cells=cells, transfer=transfer, source_id=self.source_id)
