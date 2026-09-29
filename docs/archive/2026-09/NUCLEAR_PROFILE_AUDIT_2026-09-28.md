# Nuclear charge profile versus recoil isotope

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

The v2 ledger selects an isotope for recoil, but its Dirac–Coulomb nuclear
potential uses the historical element default: a reference light-element
radius or an empirical radius with A=round(2.5Z). The audit finds different
radius and recoil mass-number parameters in 102 of 118 element entries.
Equality of these parameters alone would not establish measured-radius
accuracy.

`audit_nuclear_profile_provenance.py` records the actual v2 assumptions in
`nuclear_profile_assumptions` and `nuclear_profile_provenance.json`. It does
not change calculated energies. The website now displays recoil isotope,
radius mass-number parameter, RMS radius, source, and the historical 2 fm
Fermi radius clamp. These values are inferred from the fixed v2 generator
settings and labelled accordingly.

`DiracFock(nuclear_mass_number=A)` now passes an explicit mass number to
the finite-nucleus Hamiltonian. The legacy default is preserved. Every new
DHF result records `_nuclear_profile`; changing A does not alter the recoil
mass. Light-element radii remain element references, and heavy-element
radii remain empirical. Five focused nuclear-profile, eigenvector, and DIIS
tests pass. The nuclear test checks the actual penetrating 1s energy change
and verifies that a point nucleus is independent of this mass-number input.

## Sensitivity on matched grids

All displayed differences are selected-isotope profile minus legacy profile,
with the same recoil mass and model correction stack in both calculations.

| Observable | Energy change |
| --- | ---: |
| Ar II 3s-hole excitation, A=45 to 40 | +0.04117 cm^-1 |
| Xe I ionization, A=135 to 132 | -0.02515 cm^-1 |
| Hydrogen-like Hg ionization, A=200 to 202 | -0.26231 eV |

The Ar and Xe effects are small compared with their DHF correlation errors,
but the heavy one-electron result demonstrates that the mismatch cannot be
ignored generally. Neutral/first-ion Hg sensitivity is still running.
Raw results and convergence flags are in `nuclear_isotope_sensitivity_*`.

## Required follow-up

Prepare a new physics version that supplies the recoil isotope's A to the
charge model consistently. Preserve v2 as a reference and revalidate affected
energies; do not mix different nuclear models in a single unlabeled ledger.
No migration has yet been applied. This input correction does not replace
the need for measured radii and their uncertainties.

The QED limitations are separate: v2 uses a point-nucleus Uehling kernel and
model self-energy, not precision isotope-dependent QED. Supplying A to the
Dirac–Coulomb potential does not automatically correct these terms or the
historical light-nucleus clamp. The new interface and website state these
limits explicitly.
