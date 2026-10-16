"""Explicit finite multi-charge kinetics and optically thin energy closure.

Prepared ideal plasma, fixed volume, Maxwellian electrons and a common ion/
electron temperature. Self-consistent electron number and actual atomic energy
changes are retained. No missing atomic rate, resonance width or continuum is
invented. This is not solid ablation or a large native Sn source certification.
"""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
from scipy.integrate import solve_ivp

from atomx.constants import CODATA, DEFAULT_CODATA_VERSION
from atomx.plasma.crm import (
    radiative_recombination_rate_cm3_s,
    saha_ionization_density_cm3,
)

_C = CODATA[DEFAULT_CODATA_VERSION]


class NonLTEKineticPlasma:
    """Sourced CRM states bound to charges and an excitation/ionization energy zero.

    ``excitation_mode`` is 'tabulated' or 'none'. Bound-free processes are an
    explicit subset of 'eii', 'three-body', 'rr'; three-body requires eii.
    Effective synthetic DR satellite sources cannot stand in for physical
    resonance populations. Explicit partial Auger channels use unbranched capture.
    Rate callbacks remain caller-owned physical providers; their behavior must
    be immutable and bound to the declared rate_source_id.
    The optically thin radiation ledger includes supplied lines and spontaneous
    RR only: free-free, external photo-pumping and transport are not included.
    """

    def __init__(self, crm, charges, state_ids, *, source_id, rate_source_id,
                 excitation_mode, bound_free_processes, coverage, max_states=256,
                 max_channels=100000):
        n = crm.n_levels
        edge_count = len(crm.transitions) + len(crm.bound_free_channels) + len(crm.autoionization_channels)
        if n > max_states or edge_count > max_channels:
            raise MemoryError("finite kinetic model cap exceeded before copying; no states/channels removed")
        q = np.array(charges, copy=True)
        if (n < 1 or len(crm.levels) != n or q.shape != (n,) or q.dtype.kind not in 'iu' or np.any(q < 1)
                or len(state_ids) != n or len(set(state_ids)) != n
                or not source_id or not rate_source_id or not coverage):
            raise ValueError("explicit positive charges, unique state IDs and atomic/rate provenance required")
        if excitation_mode not in ('none', 'tabulated'):
            raise ValueError("explicit excitation mode none or tabulated required")
        processes = frozenset(bound_free_processes)
        if not processes <= {'eii', 'three-body', 'rr'} or ('three-body' in processes and 'eii' not in processes):
            raise ValueError("explicit supported bound-free processes required; three-body requires eii")
        if crm.dr_satellite_sources:
            raise ValueError("effective synthetic DR satellites are not explicit physical resonances")
        E = np.array([lev.energy_eV for lev in crm.levels], dtype=float)
        g = np.array([lev.g for lev in crm.levels], dtype=float)
        if np.any(~np.isfinite(E)) or np.any(~np.isfinite(g)) or np.any(g <= 0):
            raise ValueError("finite global energies and positive statistical weights required")
        charges_present = np.unique(q)
        if np.any(np.diff(charges_present) != 1):
            raise ValueError("declared charge stages must be contiguous")
        radiative = []
        for tr in crm.transitions:
            u, l = tr.upper, tr.lower
            if (not all(isinstance(i, (int, np.integer)) and 0 <= i < n for i in (u, l))
                    or q[u] != q[l] or E[u] <= E[l]
                    or not np.isfinite(tr.A_rate) or tr.A_rate < 0):
                raise ValueError("radiative transitions require matched same-charge states and positive photon gaps")
            if excitation_mode == 'tabulated' and (tr.omega is None or tr.omega_E_eV is None):
                raise ValueError("tabulated mode requires every excitation Omega; no gf fallback")
            if excitation_mode == 'tabulated':
                grid, omega = np.asarray(tr.omega_E_eV), np.asarray(tr.omega)
                if (grid.ndim != 1 or grid.size < 2 or grid.shape != omega.shape
                        or np.any(~np.isfinite(grid)) or np.any(~np.isfinite(omega))
                        or np.any(np.diff(grid) <= 0) or grid[0] < E[u] - E[l]
                        or np.any(omega < 0)):
                    raise ValueError("finite incident-energy Omega grid above threshold required")
            radiative.append((u, l, tr.A_rate))
        auger_widths = np.zeros(n)
        resonance_states = set()
        for ch in crm.autoionization_channels:
            r, c = ch.resonance, ch.continuum
            if (not all(isinstance(i, (int, np.integer)) and 0 <= i < n for i in (r, c))
                    or q[c] != q[r] + 1 or E[r] <= E[c]
                    or not np.isfinite(ch.partial_rate_per_s) or ch.partial_rate_per_s < 0
                    or not ch.source_id):
                raise ValueError("physical Auger channel requires an explicit adjacent continuum above threshold")
            auger_widths[r] += ch.partial_rate_per_s
            resonance_states.add(r)
        supplied_widths = np.array([lev.autoionization_rate_s for lev in crm.levels])
        if (np.any(~np.isfinite(supplied_widths)) or np.any(supplied_widths < 0)
                or not np.allclose(supplied_widths, auger_widths, rtol=1e-12, atol=0)):
            raise ValueError("all reported autoionization widths must match explicit partial channels")
        for i, charge in enumerate(q):
            above = E[q == charge + 1]
            if above.size and E[i] >= above.min() and i not in resonance_states:
                raise ValueError("above-continuum state requires explicit resonance/Auger channels")
        for ch in crm.bound_free_channels:
            b, c = ch.bound, ch.continuum
            if (not all(isinstance(i, (int, np.integer)) and 0 <= i < n for i in (b, c))
                    or q[c] != q[b] + 1 or E[c] <= E[b]
                    or not np.isclose(E[c] - E[b], ch.ionization_potential_eV, rtol=1e-12, atol=0)):
                raise ValueError("bound-free threshold must match the global state energy difference and charge")
            for supplied, actual in ((ch.g_bound, g[b]), (ch.g_continuum, g[c])):
                if supplied is not None and supplied != actual:
                    raise ValueError("bound-free statistical weight disagrees with its kinetic state")
            if ch.eii_scf_result is not None or ('eii' in processes and ch.eii_rate_cm3_s is None):
                raise ValueError("explicit EII coefficients required; live SCF/fallback rates are not accepted")
            if 'rr' in processes and (ch.bound_free_model != 'native' or not callable(ch.photoionization_sigma_cm2)):
                raise ValueError("RR energy closure requires explicit native PI cross sections")
            if ch.photoionization_rate_s is not None:
                raise ValueError("external photo-pumping requires a separately coupled radiation-energy ledger")
        # Copy only after resource/data admission; this model's numeric state cannot
        # drift with edits to the caller's mutable CRM. Provider identity is explicit.
        self._crm = copy.deepcopy(crm)
        self.energy, self.g, self.charges = E, g, q
        self._relative_energy = E - E.min()
        self.state_ids, self.atomic_source_id = tuple(state_ids), source_id
        self.processes, self.excitation_mode = processes, excitation_mode
        self.coverage, self.n_states = coverage, n
        self._upper = np.array([row[0] for row in radiative], dtype=int)
        self._line_power = np.array([row[2] * (E[row[0]] - E[row[1]]) for row in radiative])
        identity = dict(atomic=source_id, rates=rate_source_id, states=self.state_ids,
                        mode=excitation_mode, processes=sorted(processes), coverage=coverage,
                        auger=[(int(ch.resonance), int(ch.continuum), float(ch.partial_rate_per_s),
                                ch.source_id) for ch in crm.autoionization_channels],
                        bound_free=[(int(ch.bound), int(ch.continuum), ch.ionization_potential_eV,
                                     None if ch.eii_rate_cm3_s is None or callable(ch.eii_rate_cm3_s)
                                     else float(ch.eii_rate_cm3_s))
                                    for ch in crm.bound_free_channels])
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
        for array in (E, g, q, np.array(radiative), auger_widths):
            digest.update(np.asarray(array, dtype='<f8').tobytes())
        for tr in self._crm.transitions:
            for array in (tr.omega_E_eV, tr.omega):
                if array is not None:
                    digest.update(np.asarray(array, dtype='<f8').tobytes())
        self.source_id = digest.hexdigest()
        for array in (self.energy, self.g, self.charges, self._relative_energy, self._upper, self._line_power):
            array.setflags(write=False)

    def evaluate(self, temperature_eV, ion_density_per_m3, populations, *, heating_W_m3=0):
        """Physical finite-model rates, photon power and Tdot; no population repair."""
        return self._evaluate(temperature_eV, ion_density_per_m3, populations,
                              heating_W_m3=heating_W_m3, physical=True)

    def _evaluate(self, temperature_eV, ion_density_per_m3, populations, *, heating_W_m3, physical):
        p = np.asarray(populations, dtype=float)
        T, ni = temperature_eV, ion_density_per_m3
        if (p.shape != self.energy.shape or np.any(~np.isfinite(p))
                or not all(np.isfinite(v) and v > 0 for v in (T, ni))
                or not np.isfinite(heating_W_m3)
                or (physical and (np.any(p < 0) or abs(p.sum() - 1) > 1e-12))):
            raise ValueError("positive thermal state and normalized nonnegative populations required")
        qbar = float(p @ self.charges)
        if qbar <= 0:
            raise ValueError("prepared source requires positive self-consistent electron density")
        ne_cm3 = ni * qbar / 1e6
        if 'three-body' in self.processes:
            for ch in self._crm.bound_free_channels:
                if saha_ionization_density_cm3(ch.ionization_potential_eV, T,
                        g_bound=self.g[ch.bound], g_continuum=self.g[ch.continuum]) == 0:
                    raise ArithmeticError("inverse EII underflow needs a reduced-rate representation; no TBR set to zero")
        R = self._crm.rate_matrix(ne_cm3, T, sparse=True,
            include_eii='eii' in self.processes, include_three_body='three-body' in self.processes,
            include_radiative_recombination='rr' in self.processes, include_photoionization=False,
            include_collisional_excitation=self.excitation_mode == 'tabulated')
        pdot = R @ p
        line_power = float(self._line_power @ p[self._upper])
        rr_power = 0.
        if 'rr' in self.processes:
            for ch in self._crm.bound_free_channels:
                moment = radiative_recombination_rate_cm3_s(ch.ionization_potential_eV, T,
                    g_bound=self.g[ch.bound], g_continuum=self.g[ch.continuum],
                    cross_section_cm2=ch.photoionization_sigma_cm2, photon_energy_moment=1)
                rr_power += ne_cm3 * p[ch.continuum] * moment
        charge_rate, atomic_rate = float(pdot @ self.charges), float(pdot @ self._relative_energy)
        thermal_cv = 1.5 * (1 + qbar)
        Tdot = (heating_W_m3 / (ni * _C['_e']) - line_power - rr_power
                - atomic_rate - 1.5 * T * charge_rate) / thermal_cv
        return dict(generator=R, population_rate_per_s=pdot, temperature_rate_eV_per_s=Tdot,
                    electron_density_per_m3=ne_cm3 * 1e6, mean_charge=qbar,
                    line_power_eV_per_ion_s=line_power, rr_power_eV_per_ion_s=rr_power,
                    charge_rate_per_s=charge_rate, atomic_energy_rate_eV_per_ion_s=atomic_rate,
                    internal_energy_eV_per_ion=thermal_cv * T + float(p @ self.energy))


def evolve_nonlte_isochoric(model, ion_density_per_m3, temperature_eV, populations, *,
                            duration_s, heating_W_m3, sample_times_s=None, rtol=2e-8,
                            max_rhs_evaluations=10000, jacobian_memory_cap_bytes=64 * 1024**2):
    """Finite fixed-volume source; exact electron/atomic changes and escaped photons.

    ODE coordinates eliminate one population so Jacobian probes lie in the
    number-conserving tangent space. Internal probes use the smooth signed
    continuation of R(ne,T)p, never clipping/renormalizing trial populations.
    Actual returned populations and the complete energy ledger are gated.
    """
    initial = model.evaluate(temperature_eV, ion_density_per_m3, populations)
    n = model.n_states
    dimension = n + 3
    bound = 10 * 8 * dimension**2
    if bound > jacobian_memory_cap_bytes:
        raise MemoryError("joint kinetic Jacobian planning cap exceeded; no states removed")
    if not np.isfinite(duration_s) or duration_s <= 0 or not np.isfinite(rtol) or rtol <= 0:
        raise ValueError("positive finite duration and rtol required")
    p0 = np.asarray(populations, dtype=float)
    y0 = np.concatenate(([temperature_eV], p0[1:], [0., 0., 0.]))
    evaluations = 0

    def population(y):
        return np.concatenate(([1 - np.sum(y[1:n])], y[1:n]))

    def rhs(t_scaled, y):
        nonlocal evaluations
        evaluations += 1
        if evaluations > max_rhs_evaluations:
            raise RuntimeError("kinetic RHS budget exceeded; no partial energy certificate")
        heating = float(heating_W_m3(t_scaled * duration_s))
        r = model._evaluate(y[0], ion_density_per_m3, population(y), heating_W_m3=heating, physical=False)
        return np.concatenate(([r['temperature_rate_eV_per_s']], r['population_rate_per_s'][1:],
            [heating / (ion_density_per_m3 * _C['_e']), r['line_power_eV_per_ion_s'],
             r['rr_power_eV_per_ion_s']])) * duration_s

    times = np.linspace(0, duration_s, 101) if sample_times_s is None else np.asarray(sample_times_s)
    if (times.ndim != 1 or times.size < 2 or times[0] != 0 or times[-1] != duration_s
            or np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0)):
        raise ValueError("ordered sample times must span the full duration")
    sol = solve_ivp(rhs, (0, 1), y0, method='Radau', t_eval=times / duration_s,
                    rtol=rtol, atol=np.concatenate(([1e-8], np.full(n - 1, 1e-13), [1e-9]*3)), max_step=.025)
    if not sol.success:
        raise RuntimeError("nonLTE energy integration failed: " + sol.message)
    p = np.column_stack([population(y) for y in sol.y.T])
    if p.min() < -1e-10 or np.max(abs(p.sum(axis=0) - 1)) > 1e-10 or sol.y[0].min() <= 0:
        raise ArithmeticError("kinetic population/temperature gate failed; no population repair")
    q = model.charges @ p
    relative_energy = 1.5 * (1 + q) * sol.y[0] + model._relative_energy @ p
    energy = relative_energy + model.energy.min()
    ledger = sol.y[n:]
    initial_relative = 1.5 * (1 + initial['mean_charge']) * temperature_eV + p0 @ model._relative_energy
    defect = initial_relative + ledger[0] - relative_energy - ledger[1] - ledger[2]
    # Gauge-independent conditioning: an arbitrary common atomic-energy zero
    # must not make the relative residual appear smaller.
    scale = initial_relative
    relative = float(np.max(abs(defect) / (scale + abs(ledger[0]))))
    if relative > max(1e-6, 20 * rtol):
        raise ArithmeticError(f"nonLTE matter/photon energy gate failed: {relative}")
    return dict(time_s=sol.t * duration_s, populations=p, temperature_eV=sol.y[0], mean_charge=q,
                electron_density_per_m3=ion_density_per_m3 * q, internal_energy_eV_per_ion=energy,
                ledger_eV_per_ion=ledger, energy_defect_eV_per_ion=defect,
                maximum_relative_energy_defect=relative, minimum_population=float(p.min()),
                ledger_rows=('deposited_heat', 'escaped_lines', 'escaped_rr'),
                source_id=model.source_id, rhs_evaluations=evaluations, jacobian_planning_bytes=bound,
                closure='finite multi-charge optically-thin isochoric ideal Maxwellian source; no free-free/transport/solid EOS',
                experimental_CE_accepted=False)
