# Comparing Atomx with NIST

Compare the same physical quantity and state before interpreting a discrepancy.
Numerical convergence, physical-model accuracy and reference-data coverage are
separate questions.

## Matching rules

| Atomx quantity | Suitable reference | Required distinction |
|---|---|---|
| Resolved level difference | Identified NIST levels or their transition | Configuration, J, parity and level identity |
| Configuration-average excitation | Complete configuration centroid | Include the required manifold and degeneracy weights |
| Ionization energy | Adjacent charge-state total-energy difference | Same isotope/model; ground term versus configuration average |
| Predicted line wavelength | Observed or explicitly selected Ritz wavelength | Common medium and units; preserve which reference was selected |
| Transition probability | Classified transition with matching multipole | Same states, operator and units |

Ritz wavelengths are derived from reference level differences. Agreement with
Ritz values is not an independent ab-initio calculation. A DHF configuration
average cannot be scored as every resolved line within its multiplet.

Incomplete manifolds, ambiguous assignments, failed calculations and unmatched
records remain visible with reasons. They must not become zero errors or be
silently excluded from coverage denominators. Evaluated, theoretical and
semiempirical source classifications remain distinct.

## Numerical and physical checks

1. Verify state identity and occupied orbital ordering, including core-hole and
   diffuse excited configurations.
2. Require convergence of the energy and the applicable orbital/variational
   residuals. Inspect failed or stagnating iterations.
3. Check the declared retained equations at fixed space against an independent
   signed operator and low-order limit. Separate implementation defects from
   omitted correlation ranks and Hamiltonian/operator terms. Expanding virtuals
   cannot establish that either class is absent.
4. Estimate differential physical responses before selecting another scan:
   upper minus lower state, complete doublet centroid and splitting. Separate
   measured same-model increments, literature scales and unknown contributions.
5. Distinguish numerical grid/box refinement from restoring excluded correlation
   channels. Additional partial waves and virtual-energy bands can supply missing
   physics; an old narrow-space increment does not bound the current tail.
6. Compare matched correlation ranks and correction stages, nuclear profiles and
   isotope masses. Use the smallest control distinguishing model truncation from
   representation error; do not prioritize more digits solely by test count.

The [Yb physics-scale assessment](atomic_physics_budget_2026_10_11/README.md)
recomputes all five saved states and their common absolute threshold. The
[current correctness-first audit](atomic_physics_correctness_2026_10_11/README.md)
supersedes its execution priority: validate retained equations and their low-order
limit, then discriminate nonlinear/rank and balanced core-hole model omissions
at matched fixed space. The g-band/energy-tail screen is deferred until these
gates pass; historical corrections are not transferred to the current roots.

The [native roadmap and recovered successes](atomic_native_roadmap_2026_10_11/README.md)
supersede the earlier incomplete inventory. Native Ne7+ q/r positions differ from
Togawa 2024 experiments by -8.896190/-13.881082 meV; these are not NIST rows.
Native calibrated Eu3+ 4f6 has historical target RMS 8.2 cm-1 and full-M
3003-determinant assembly. Its aquo-ion targets and the current local NIST
bracketed semiempirical rows are distinct references; their agreement is not
independent ab-initio experimental certification. Preserve successful numbers,
declared spaces, fit status and open gates together. Develop physics in Atomx;
external programs supply reference equations and independent benchmarks.

DHF includes mean-field exchange; it does not include general electron
correlation. MCDHF and CI include correlation only within their declared spaces.
Model QED corrections do not establish precision-QED accuracy for arbitrary
many-electron spectra. Read the calculation specification and correction ledger
instead of inferring the Hamiltonian from a short plot label.

## Supported entry points

The current [native MCDHF→RCI Yb⁺ example](yb_mcdhf_rci_2026_10_11/README.md)
keeps the correlated orbital-stationarity gate beside each reported comparison.
An unconverged optimization can supply diagnostic energies, but cannot replace
the previous model's certified numerical status or become an accepted spectrum.

Shared parsing and matching live in `atomx.reference_data`; native-result
comparison services live in `atomx.workflows.nist_comparison`. Their tests call
those packages directly. The archive builder, viewer and campaign comparison
scripts have been retired. A new read-only periodic exporter presents selected
saved records and nominal Yb branch comparisons; see
[NIST reference data](NIST_SPECTRUM_ATLAS.md). Historical DHF coverage does not
certify current-source convergence or experimental accuracy.

`completed_pair` checks calculation identity across two radial grids; `interval`
checks shared endpoint Hamiltonians and returns the energy difference and grid
change. `collect(campaign, archive)` reads caller-selected SQLite databases
containing the campaign, line queue and NIST schemas. It does not create those
databases or calculate new states. Retained historical ledgers must carry their
original model identity; they are not upgraded by changing today's defaults.

## Reporting

Report coverage, matched-state counts, residuals, reference uncertainties and
numerical refinement separately. Retain individual records behind every summary.
Do not call a numerical refinement shift an experimental uncertainty or combine
correlated reference uncertainties as if they were independent.

Previous run counts, investigations and benchmark artifacts are in the
[historical archive](archive/README.md). Current capabilities are listed in
[the capability table](ARCHITECTURE_CAPABILITIES.md).
