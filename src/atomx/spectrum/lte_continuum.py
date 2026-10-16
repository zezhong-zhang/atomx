"""LTE continuum closure for source transfer, using explicit bound-free data.

Free-free reuses the shared emission/absorption kernel. Bound-free uses an
actual PI cross section and LTE detailed balance; it is not a Kramers fallback.
This is not a non-LTE radiative-recombination/photopumping closure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from atomx.constants import CODATA, DEFAULT_CODATA_VERSION
from atomx.spectrum.continuum import free_free_absorption, free_free_emissivity

_C = CODATA[DEFAULT_CODATA_VERSION]


@dataclass(frozen=True)
class LTEBoundFreeEdge:
    threshold_eV: float
    bound_density_per_m3: float
    cross_section_cm2: Callable[[float], float]
    source_id: str


def lte_continuum_coefficients(energy_edges_eV, temperature_eV, ne_per_m3, ni_per_m3,
                               mean_charge_squared, *, free_free_gaunt, gaunt_source_id,
                               bound_free_edges=()):
    """4pi power W/m³/bin and net opacity/m on a declared photon-energy grid.

    The caller explicitly supplies Gaunt physics and records its source. LTE
    PI/RR reciprocity is valid only when the bound/continuum populations share
    the same Saha EOS and state degeneracies. No missing edge is synthesized.
    """
    bound_free_edges = tuple(bound_free_edges)
    edges = np.asarray(energy_edges_eV, dtype=float)
    if (edges.ndim != 1 or edges.size < 2 or edges[0] < 0
            or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0)
            or not callable(free_free_gaunt) or not gaunt_source_id
            or not all(np.isfinite(v) and v > 0 for v in
                       (temperature_eV, ne_per_m3, ni_per_m3, mean_charge_squared))):
        raise ValueError("positive grid/thermal state and explicit Gaunt source required")
    E = (edges[:-1] + edges[1:]) / 2
    T = temperature_eV
    kwargs = dict(n_e_cm3=ne_per_m3 / 1e6, n_i_cm3=ni_per_m3 / 1e6, gaunt=free_free_gaunt)
    eps_ff = free_free_emissivity(E, T, np.sqrt(mean_charge_squared), **kwargs) * _C["_e"] * 1e6 * np.diff(edges)
    alpha_ff = free_free_absorption(E, T, np.sqrt(mean_charge_squared), **kwargs) * 100
    frequency = E * _C["_e"] / _C["_hplanck"]
    width_Hz = np.diff(edges) * _C["_e"] / _C["_hplanck"]
    occupation = np.exp(-E / T) / -np.expm1(-E / T)
    planck_bin = 2 * _C["_hplanck"] * frequency**3 / _C["_c"]**2 * occupation * width_Hz
    alpha_bf = np.zeros_like(E)
    for edge in bound_free_edges:
        if (not np.isfinite(edge.threshold_eV) or edge.threshold_eV <= 0
                or not np.isfinite(edge.bound_density_per_m3) or edge.bound_density_per_m3 < 0
                or not callable(edge.cross_section_cm2) or not edge.source_id):
            raise ValueError("positive threshold, explicit PI source and nonnegative bound density required")
        active = edge.threshold_eV <= E
        sigma = np.array([edge.cross_section_cm2(float(photon)) for photon in E[active]])
        if np.any(~np.isfinite(sigma)) or np.any(sigma < 0):
            raise ValueError("PI provider returned invalid cross section")
        alpha_bf[active] += edge.bound_density_per_m3 * sigma * 1e-4 * -np.expm1(-E[active] / T)
    eps_bf = 4 * np.pi * alpha_bf * planck_bin
    if any(np.any(~np.isfinite(v)) or np.any(v < 0) for v in (eps_ff, alpha_ff, eps_bf, alpha_bf)):
        raise ArithmeticError("invalid continuum emission or opacity")
    return dict(emissivity_W_m3_bin=eps_ff + eps_bf, absorption_per_m=alpha_ff + alpha_bf,
                free_free_emissivity_W_m3_bin=eps_ff, free_bound_emissivity_W_m3_bin=eps_bf,
                gaunt_source_id=gaunt_source_id,
                bound_free_source_ids=tuple(edge.source_id for edge in bound_free_edges))
