"""Scalar native subshell PI callback shared by CRM and XFEL consumers."""

from functools import lru_cache

import numpy as np

from atomx.constants import BOHR_RADIUS_M
from atomx.scattering.photoionization.subshell import subshell_photoionization


class NativeSubshellCrossSection:
    def __init__(self, result, orbital_label, *, threshold_eV, continuum_model):
        if not np.isfinite(threshold_eV) or threshold_eV <= 0:
            raise ValueError("positive native ionization threshold required")
        if orbital_label not in {s.label for s in result.subshells if s.occupation > 0}:
            raise ValueError("PI orbital must be occupied")
        self.result = result
        self.orbital_label = orbital_label
        self.threshold_eV = float(threshold_eV)
        self.continuum_model = continuum_model
        self.provenance = "native subshell E1, explicit threshold and continuum model"
        self._cached = lru_cache(maxsize=512)(self._evaluate)

    def _evaluate(self, energy_eV):
        result = subshell_photoionization(
            self.result,
            [energy_eV],
            continuum_model=self.continuum_model,
            orbitals=[self.orbital_label],
            thresholds_eV={self.orbital_label: self.threshold_eV},
        )
        return float(result.sigma_total_cm2[0])

    def __call__(self, energy_eV):
        if not np.isfinite(energy_eV) or energy_eV <= 0:
            raise ValueError("positive finite photon energy required")
        if energy_eV <= self.threshold_eV:
            return 0.0
        return self._cached(float(energy_eV))

    def evaluate_grid(self, energies_eV):
        result = subshell_photoionization(
            self.result,
            energies_eV,
            continuum_model=self.continuum_model,
            orbitals=[self.orbital_label],
            thresholds_eV={self.orbital_label: self.threshold_eV},
        )
        return result.sigma_total_a0_sq * (BOHR_RADIUS_M * 100) ** 2
