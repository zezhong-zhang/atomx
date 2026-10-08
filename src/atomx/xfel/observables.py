"""Native, line-aware transient atomic observables, including spectral variance.

Continuum dispersion uses the explicit finite energy window. Discrete
E1 response covers the same configuration basis as the kinetics, with
signed downward strengths for excited states. Missing higher orbitals and
Fano interference are not silently represented by the isolated poles.
"""

from dataclasses import dataclass

import numpy as np

from atomx.constants import HARTREE_EV
from atomx.scattering.anomalous import f1_kramers_kronig, f2_from_sigma
from atomx.scattering.photoionization.subshell import subshell_photoionization
from atomx.xfel.imaging import ScatteringFactorBank, native_form_factor
from atomx.xfel.model import _check_result, gaussian_spectral_quadrature, row_config


@dataclass
class XFELSimulation:
    dynamics: object
    imaging: object
    scattering_bank: object
    provenance: dict


def _line_cuts(lines, energy, bandwidth):
    if bandwidth == 0:
        return []
    sigma = bandwidth / np.sqrt(8 * np.log(2))
    cuts = []
    for line, _sign in lines:
        if abs(line.energy_eV - energy) > 10 * sigma:
            continue
        cuts.append(line.energy_eV)
        width = line.half_width_eV
        # Resolve narrow line intensity tails across their full dynamic range.
        while width < 20 * sigma:
            cuts.extend([line.energy_eV - width, line.energy_eV + width])
            width *= 4
    return cuts


def build_native_scattering_bank(
    model, q_a0_inv, *, dispersion_grid_eV, spectral_order=32, bound_bound=True
):
    """Build f0, continuum KK and discrete causal E1 spectral moments.

    The mean amplitude and the mean squared amplitude are separately
    integrated over the same Gaussian photon spectrum used for kinetics.
    Line-width ladders and native PI thresholds split the quadrature; a
    smooth global Gaussian grid would miss narrow resonance intensities.
    This is single-atom spherical response in the chosen orbital basis.
    """
    grid = np.asarray(dispersion_grid_eV, dtype=float)
    if (
        grid.ndim != 1
        or len(grid) < 8
        or not np.isfinite(grid).all()
        or grid[0] <= 0
        or np.any(np.diff(grid) <= 0)
    ):
        raise ValueError(
            "an increasing positive continuum dispersion window is required"
        )
    energy = model.graph.photon_energy_eV
    bandwidth = model.table["provenance"]["continuum_bandwidth_fwhm_eV"]
    sigma = bandwidth / np.sqrt(8 * np.log(2))
    if grid[0] >= energy - 9 * sigma or grid[-1] <= energy + 9 * sigma:
        raise ValueError(
            "continuum dispersion window must cover the full probe spectrum"
        )
    state_lines = {sid: [] for sid in model.graph.state_ids}
    if bound_bound:
        for line in model.lines:
            if line.lower in state_lines:
                state_lines[line.lower].append((line, 1.0))
            if line.upper in state_lines:
                state_lines[line.upper].append((line, -line.g_lower / line.g_upper))
    means, intensities = (
        {"": np.zeros_like(q_a0_inv, dtype=complex)},
        {"": np.zeros_like(q_a0_inv, dtype=float)},
    )
    diagnostics = {}
    for sid in model.graph.state_ids:
        if sid == "":
            continue
        row = model.table["configs"][sid]
        result = model.result_loader(row_config(row))
        _check_result(result, row)
        f0 = native_form_factor(result, q_a0_inv)
        thresholds = list(row["thresholds_eV"].values())
        thresholds = [t for t in thresholds if grid[0] < t < grid[-1]]
        knots = np.unique(np.r_[grid, thresholds, np.array(thresholds) * (1 + 1e-8)])
        continuum = subshell_photoionization(
            result,
            knots,
            continuum_model=model.continuum_model,
            thresholds_eV=row["thresholds_eV"],
        )
        continuum_f2 = f2_from_sigma(knots / HARTREE_EV, continuum.sigma_total_a0_sq)
        cuts = thresholds + _line_cuts(state_lines[sid], energy, bandwidth)
        photons, weights = gaussian_spectral_quadrature(
            energy, bandwidth, cuts, order=spectral_order
        )
        if not np.isclose(weights.sum(), 1, rtol=0, atol=1e-10):
            raise RuntimeError(
                "spectral moment quadrature does not resolve photon probability"
            )
        pi = subshell_photoionization(
            result,
            photons,
            continuum_model=model.continuum_model,
            thresholds_eV=row["thresholds_eV"],
        )
        continuum_response = f1_kramers_kronig(
            knots / HARTREE_EV, continuum_f2, photons / HARTREE_EV
        ) + 1j * f2_from_sigma(photons / HARTREE_EV, pi.sigma_total_a0_sq)
        response = continuum_response.copy()
        for line, sign in state_lines[sid]:
            response += sign * np.array([line.anomalous_factor(e) for e in photons])
        factors = f0[None, :] + response[:, None]
        means[sid] = weights @ factors
        intensities[sid] = weights @ abs(factors) ** 2
        diagnostics[sid] = dict(
            continuum_points=len(knots),
            spectral_points=len(photons),
            signed_discrete_lines=len(state_lines[sid]),
            f0_zero_electrons=sum(row["occupations"].values()),
        )
    provenance = dict(
        f0="actual jj radial density",
        continuum="native Delta-SCF PI, optical theorem and finite-window KK",
        dispersion_window_eV=[float(grid[0]), float(grid[-1])],
        bound_bound="signed causal native E1 poles within graph"
        if bound_bound
        else "explicitly omitted",
        spectral_model=f"Gaussian9sigma, Legendre{spectral_order}, native threshold and line-width splits",
        intensity="mean(|f|²) over states and photon spectrum; not squared mean amplitude",
        diagnostics=diagnostics,
        known_gap="chosen configuration basis only; no Fano interference or coherent strong-field response",
    )
    return ScatteringFactorBank(
        q_a0_inv,
        means,
        photon_energy_eV=energy,
        provenance=provenance,
        intensities_by_state=intensities,
        bandwidth_fwhm_eV=bandwidth,
    )
