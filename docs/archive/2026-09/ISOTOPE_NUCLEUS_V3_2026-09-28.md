# Versioned isotope-consistent nuclear charge model

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

The separate `full_dhf_corrected_v3.sqlite` ledger uses
`dbsr_static_breit_model_qed_grasp_recoil_isotope_nucleus_v3`.
Every newly calculated endpoint supplies its stored isotope A to the
finite-nucleus Hamiltonian while retaining the same recoil mass and
correction stack. All initial-field, damping, Slater-seed, and shooting-seed
final DHF paths forward the isotope parameter.

## Migration and validation

`prepare_isotope_nucleus_generation.py` takes a committed SQLite backup of
v2 and publishes a self-contained database only after the integrity audit
passes. Source job rows remain in `nuclear_profile_migration_history`.

- 1,416 completed states have exactly identical old/new Fermi shape
  parameters and are reused with explicit equivalent-profile provenance.
- 3,418 completed states have changed charge profiles and are requeued;
  their old energies and predictions are not presented as v3 results.
- 10,861 unfinished source states are requeued with no inherited worker
  ownership. Successful diffuse Mg Slater-seed routes retain their retry
  queue assignment.

The v3 ledger audit checks each completed state's recorded charge profile
against its selected isotope, in addition to component sums, recoil,
configuration identity, state provenance, and transition consistency.
Fourteen focused migration, profile, audit, and comparison tests pass.

Pilot v3 generation reproduces the independent Ar II ground-state probe to
6.1e-10 Ha and the hydrogen-like Hg probe exactly at stored precision.
The v2 workers were then stopped after their identities were verified and
their running rows preserved. v2 remains available with 4,995 completed
configurations; its metadata prevents accidental further generation.

Eight v3 workers were launched: four for affected H–Ca configurations,
three for Z>20, and one for the diffuse Mg retry queue. The website selects
the new ledger after atomic publication, reads its own nuclear-profile
table, and labels v2 diagnostic panels as historical. Incomplete v3
endpoints have no predictions.

## Scope and remaining limitations

This fixes the mass-number mismatch in the Dirac–Coulomb charge model. It
does not supply measured isotope radii, remove the historical 2 fm clamp
for small Fermi nuclei, add finite-size recoil, or replace model QED with
precision isotope-dependent QED. Those distinctions remain visible in the
website and recorded profile metadata. The electronic correlation gap
remains a separate issue.
