"""CI+MBPT: effective Hamiltonian on an exact-CI reference.

The Dzuba–Flambaum–Kozlov hybrid for many-valence atoms: keep the CI
space small enough to diagonalise *exactly* (the reference layer), and
let correlation beyond that space enter through MBPT operators instead
of through CSF proliferation::

    H_eff = H_CI + Σ⁽¹⁾ + Σ⁽²⁾
              │       │      └ two-body screening (NOT IMPLEMENTED)
              │       └ one-body self-energy (implemented, see below)
              │         over a closed-shell core reference
              └ exact CI on the chosen valence CSF space
                (``ci_solve`` / ``build_ci_matrix``)

Why this module exists
----------------------
Measured on Eu³⁺ (2026-09-30 probe series, clean HEAD): the full
valence SD space of ``[Xe]4f⁶`` needs 60 720 determinants for the 2J=0
block against the exact-CI default cap of 2000 — a dense all-pairs
engine cannot go there.  The physically correct route at this scale is
perturbative: ``Σ⁽¹⁾`` and ``Σ⁽²⁾`` computed once from a closed core,
applied to a compact CI space.

Status
------
* **Σ⁽¹⁾ (implemented)** — diagonal second-order self-energy
  ``Δε_a = <a|Σ⁽¹⁾|a>`` of each valence orbital, supplied as a plain
  ``{(n, kappa): shift_in_hartree}`` mapping.  Compute the shifts from
  a solved closed-shell core reference with
  :mod:`atomx.methods.mbpt_transitions`
  (:func:`mbpt2_orbital_energy_shift`) for occupied valence orbitals,
  or extend that helper for virtual-valence sums.  Injection is exact
  at the one-body level: the shift rides the ``("h", n, κ, n)`` radial
  key, so the determinant→CSF projection weights it by the true
  one-body density of every CSF pair, on- and off-diagonal.

* **Σ⁽²⁾ (NOT implemented)** — the two-body core-polarisation
  screening of the Coulomb integrals ``R^k``.  This is the term that
  moves multiplet splittings (e.g. the Eu³⁺ ⁵D−⁷F gap, which sits
  ~3000 cm⁻¹ high without it), so until it lands, CI+Σ⁽¹⁾ alone must
  not be sold as CI+MBPT.  Requesting it raises.  Planned shape: wrap
  the ``("r", k, ...)`` radial key with the second-order bubble sum
  over core orbitals built on :mod:`atomx.methods.mbpt`'s J-coupled
  ``V_J`` machinery, mirroring the ``one_body_shifts`` hook.

Notes
-----
* Σ⁽¹⁾ is largely common-mode within a single configuration (all CSFs
  share the valence orbitals): expect it to matter for term
  *positions* between configurations, not for intra-term splittings.
  This is a property of the physics, not a bug in the injection.
* Orbital shifts are energy-like quantities in Hartree; they must come
  from the same nuclear model and radial grid as the CI reference.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from atomx.methods.ci import ci_solve

__all__ = ["CIMBPTResult", "ci_mbpt_solve"]


def ci_mbpt_solve(
    csfs,
    orbitals_dict: Dict[Tuple[int, int], tuple],
    orbital_basis,
    bsp_data,
    Z: int,
    ns: int,
    *,
    sigma1: Optional[Dict[Tuple[int, int], float]] = None,
    sigma2: Optional[dict] = None,
    **ci_kwargs,
):
    """Diagonalise ``H_CI + Σ⁽¹⁾`` (and later ``Σ⁽²⁾``) on a valence space.

    Parameters
    ----------
    sigma1
        ``{(n, kappa): Δε in Hartree}`` — diagonal one-body self-energy
        of each valence orbital over the closed-shell core reference.
        ``None`` (default) reduces to a pure exact-CI solve.
    sigma2
        Two-body screening operators.  Not implemented; passing any
        non-``None`` value raises instead of silently degrading to a
        Σ⁽¹⁾-only result.
    **ci_kwargs
        Forwarded verbatim to :func:`atomx.methods.ci.ci_solve`
        (``n_states``, ``solver``, ``max_determinants``,
        ``include_breit``, ...).

    Returns
    -------
    :class:`atomx.methods.ci.CIResult`
        with an added ``mbpt_provenance`` attribute recording what was
        (and was not) applied, so downstream consumers cannot mistake a
        Σ⁽¹⁾-only result for full CI+MBPT.
    """
    if sigma2 is not None:
        raise NotImplementedError(
            "Σ⁽²⁾ (two-body core-polarisation screening) is not implemented. "
            "Until it is, CI+Σ⁽¹⁾ alone must not be reported as CI+MBPT: "
            "multiplet splittings (e.g. Eu³⁺ ⁵D−⁷F) are dominated by the "
            "two-body screening term.")
    result = ci_solve(csfs, orbitals_dict, orbital_basis, bsp_data, Z, ns,
                      one_body_shifts=sigma1, **ci_kwargs)
    result.mbpt_provenance = {
        "sigma1": dict(sigma1 or {}),
        "sigma2": "not_implemented",
    }
    return result
