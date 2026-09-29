# Sc X: independent reference resolves a false node rejection

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

Configuration: `1s2 2s2 2p6 3p1 4p1`, statistical configuration average.

The previous node guard required `n-l-1` radial nodes in occupied 3p
orbitals. That local-potential rule incorrectly rejected the current
stationary nonlocal DHF solution for this configuration.

## Controlled Atomx checks

`scripts/probe_sc_x_stationarity.py` uses the production grid, selected
Sc-45 mass and Fermi profile, static DBSR Breit, model QED, and recoil.
The diagnostic compares Dirac-Coulomb energies and finite differences under
allowed orthonormal 3p/4p rotations; corrections are reported separately.

| Rotation rule / initial field | DC energy (Ha) | Largest rotation gradient (Ha/rad) |
|---|---:|---:|
| Frozen / TFDA | -729.1719340192707 | 0.0985782 |
| Stationary / TFDA | -729.2225475600890 | 5.7e-10 |
| Stationary / GSZ | -729.2225475600889 | 5.7e-10 |
| Stationary / hydrogenic | -729.2225475600889 | 1.2e-9 |

The frozen branch has nominal node counts but is not stationary. The three
stationary starts agree and each gives two substantial nodes in both 3p
orbitals. Counts are unchanged at amplitude cutoffs 0.1%, 1%, 3%, and 10%.

## Independent DBSR_HF calculation

Reference input, saved solve log, exported radial functions, and comparison:
`output/nist_spectra/dbsr_sc_x_reference/`.

The independently built DBSR_HF executable uses `term=LS`, statistical
weights `eal=5`, orbital rotations enabled, and the same Fermi parameters:
`c=7.22551248849225e-5` bohr, `a=9.890591372451268e-6` bohr.
Its DC energy is **-729.222547382401558 Ha**, differing from Atomx by
**0.038998 cm^-1**. Exported DBSR_HF radial functions independently have
two nodes in each occupied 3p and 4p orbital.

This establishes agreement for the configuration-average DC solution, not
agreement with a resolved NIST level or validation of the QED corrections.

## Guard correction

For this exact Sc X configuration with the current stationary-rotation
provenance, the guard expects two 3p nodes and also checks both 4p orbitals.
Frozen-rotation historical checks and all other configurations retain their
existing rules. The new regression compares the DC energy to the independent
reference and checks orbital ordering and the corrected node guard.

No frozen-rotation result is substituted into production. The existing
pending job was reserved for a worker loading the corrected guard.

Both Sc X regression tests passed (40.62 seconds). The fresh production
worker completed the reserved record with corrected total energy
**-728.840620462 Ha** and the full current correction stack. Its line
comparisons remain waiting for other required configurations; this job alone
does not complete the Sc X spectrum.
