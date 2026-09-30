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


def radial_evaluator(orbitals, workspace, orbital_energies):
    """Evaluate existing Gaunt/retardation keys at the two vertex frequencies.

    ``orbital_energies`` maps every ``(n, kappa)`` to an energy in Hartree.
    Explicit input prevents assigning fictitious energies to correlation
    orbitals. Caches include frequency, rank and the transition density.
    The real dispersive interaction preserves Hermiticity and electron
    exchange symmetry; absorptive photon terms are outside this model.
    """
    energies = validate_orbital_energies(orbitals, orbital_energies)
    r, weights, wave, divergence = transition_current_fields(orbitals, workspace)
    densities, potentials, values, kernels = {}, {}, {}, {}

    def density(kind, a, c):
        if kind == "t":
            return divergence(a, c)
        pair = tuple(sorted((a, c)))
        if pair not in densities:
            densities[pair] = (wave(a[1:])[0 if a[0] == "P" else 1]
                               * wave(c[1:])[0 if c[0] == "P" else 1])
        return densities[pair]

    def potential(kind, rank, a, c, omega):
        if kind == "g":
            a, c = sorted((a, c))
        key = kind, rank, a, c, omega
        if key not in potentials:
            geometry = rank, omega
            if geometry not in kernels:
                kernels[geometry] = HelmholtzKernel(r, weights, rank, ALPHA * omega)
            potentials[key] = kernels[geometry].potential(
                density(kind, a, c),
                component="retardation" if kind == "t" else "full")
        return potentials[key]

    def evaluate(key):
        if key not in values:
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
            values[key] = float(value / len(frequencies))
            if not np.isfinite(values[key]):
                raise FloatingPointError(f"non-finite transverse integral {key}")
        return values[key]

    return evaluate
