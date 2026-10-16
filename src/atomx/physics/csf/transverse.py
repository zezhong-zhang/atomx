"""Real, frequency-dependent transverse one-photon exchange in Coulomb gauge.

T(k) = -alpha1.alpha2 cos(k R)/R
       + (alpha1.grad1)(alpha2.grad2) [cos(k R)-1]/(k² R).

Integrating the second term by parts gives the transition-current
divergences used by the static Breit operator. Its angular prefactor is
unchanged if its distance kernel R is replaced by -2(cos(k R)-1)/(k² R).
This works for nonlocal HF spinors without a continuity substitution
based on local orbital equations.

For <ab|T|cd> we average the operators at |epsilon_a-epsilon_c| and
|epsilon_b-epsilon_d|, as in FAC BreitRK/BreitSK. The supplied energies
are fixed throughout CI. This is an effective one-photon prescription,
not a full energy-dependent bound-state QED calculation.
"""

from functools import lru_cache

import numpy as np

from atomx.constants import ALPHA
from atomx.numerics.helmholtz import HelmholtzKernel
from atomx.physics.csf.retardation import transition_current_fields


def validate_orbital_energies(orbitals, orbital_energies):
    """Copy a complete, finite Hartree energy map for the supplied orbitals."""
    if orbital_energies is None:
        raise ValueError("finite-frequency Breit needs explicit orbital energies")
    energies = {key: float(orbital_energies[key]) for key in orbitals
                if key in orbital_energies}
    if energies.keys() != set(orbitals):
        raise ValueError("finite-frequency Breit needs energies for every orbital")
    if not all(np.isfinite(e) for e in energies.values()):
        raise ValueError("finite-frequency Breit orbital energies must be finite")
    return energies


def frequency_provenance(energies):
    """One description shared by CI and post-SCF expectations."""
    return dict(
        frequency_prescription="symmetric-two-vertex-orbital-differences",
        orbital_energies_Ha=[dict(n=n, kappa=k, energy=float(energies[n, k]))
                            for n, k in sorted(energies)],
        note="Real Coulomb-gauge one-photon exchange on fixed spinors; replaces "
             "static Breit. No finite-frequency orbital relaxation, absorptive "
             "interaction or higher-order QED terms.")


def radial_evaluator(orbitals, workspace, orbital_energies, *, cache_size=128):
    """Evaluate existing Gaunt/retardation keys at the two vertex frequencies.

    ``orbital_energies`` maps every ``(n, kappa)`` to an energy in Hartree.
    Explicit input prevents assigning fictitious energies to correlation
    orbitals. Caches include frequency, rank and the transition density.
    The real dispersive interaction preserves Hermiticity and electron
    exchange symmetry; absorptive photon terms are outside this model.
    """
    if (isinstance(cache_size, (bool, np.bool_))
            or not isinstance(cache_size, (int, np.integer)) or cache_size < 1):
        raise ValueError("transverse cache capacity must be a positive integer")
    cache_size = int(cache_size)
    energies = validate_orbital_energies(orbitals, orbital_energies)
    r, weights, wave, divergence = transition_current_fields(
        orbitals, workspace, cache_size=cache_size)

    @lru_cache(maxsize=cache_size)
    def gaunt_density(a, c):
        return (wave(a[1:])[0 if a[0] == "P" else 1]
                * wave(c[1:])[0 if c[0] == "P" else 1])

    def density(kind, a, c):
        if kind == "t":
            return divergence(a, c)
        return gaunt_density(*sorted((a, c)))

    @lru_cache(maxsize=cache_size)
    def kernel(rank, omega):
        return HelmholtzKernel(r, weights, rank, ALPHA * omega)

    @lru_cache(maxsize=cache_size)
    def cached_potential(kind, rank, a, c, omega):
        return kernel(rank, omega).potential(
            density(kind, a, c),
            component="retardation" if kind == "t" else "full")

    def potential(kind, rank, a, c, omega):
        if kind == "g":
            a, c = sorted((a, c))
        return cached_potential(kind, rank, a, c, omega)

    @lru_cache(maxsize=8192)
    def evaluate(key):
        kind, rank, a, c, b, d = key
        if kind not in {"g", "t"}:
            raise ValueError("transverse interaction expects Gaunt/retardation keys")
        orbital_keys = [s[1:] if kind == "g" else s for s in (a, c, b, d)]
        ea, ec, eb, ed = (energies[s] for s in orbital_keys)
        frequencies = abs(ea - ec), abs(eb - ed)
        if frequencies[0] == frequencies[1]:
            frequencies = frequencies[:1]
        value = 0.0
        for omega in frequencies:
            # Average radial integration directions to remove quadrature
            # asymmetry, separately from the physical frequency average.
            value += 0.5 * np.sum(weights * (
                density(kind, a, c) * potential(kind, rank, b, d, omega)
                + density(kind, b, d) * potential(kind, rank, a, c, omega)))
        value = float(value / len(frequencies))
        if not np.isfinite(value):
            raise FloatingPointError(f"non-finite transverse integral {key}")
        return value

    def diagnostics():
        caches = dict(waves=wave.cache_info(), divergence=divergence.cache_info(),
                      densities=gaunt_density.cache_info(), kernels=kernel.cache_info(),
                      potentials=cached_potential.cache_info(), values=evaluate.cache_info())
        return dict(caches={name: info._asdict() for name, info in caches.items()},
                    retained_array_bound_bytes=12 * cache_size * r.size * 8,
                    scope="Cached FP64 array payload only; excludes inputs, Python keys, "
                          "scalar cache, active temporaries and peak RSS")

    evaluate.diagnostics = diagnostics

    return evaluate
