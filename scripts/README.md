# Repository commands

Run commands from the repository root after installing Atomx, or set
`PYTHONPATH=src:.` for a source checkout. Each command documents its arguments
with `--help`. Paths below remain stable for callers and retained job ledgers.

## Ownership

Commands select inputs, manage files or databases, and report results. Shared
parsing and physical matching belong to `atomx.reference_data`; comparison
services belong to `atomx.workflows`. Numerical algorithms belong to
`atomx.numerics`. Tests of shared behavior import those package APIs directly;
command tests cover arguments, archive schemas, job state and output.

A repair command is a ledger operation for a particular calculation history.
Its retry policy is not a default solver algorithm. Keep campaign evidence in
`docs/archive/`, and consult the current [comparison guide](../docs/THEORY_NIST_COMPARISON.md)
for which experimental quantities a model can predict.

## Archive and comparison

| Task | Entry points |
|---|---|
| Build the retained line/level archive | `build_nist_spectra.py` |
| Add source fields and holdings | `import_nist_*.py` |
| Check source integrity and interpretation | `audit_nist_*.py`, `verify_nist_live_samples.py` |
| Serve the interactive explorer | `view_nist_spectra.py` with `nist_spectra_viewer.html` and `theory_comparison_viewer.html` |
| Compare line positions | `compare_theory_nist.py`, `compare_multielectron_nist.py`, `compare_full_dhf_nist.py` |
| Compare matched averages and ionization energies | `compare_configuration_centroids_nist.py`, `compare_ionization_dhf_nist.py` |
| Inspect isotope records | `compare_isotope_records_nist.py` |

The [atlas guide](../docs/NIST_SPECTRUM_ATLAS.md) explains archive construction
and the viewer. Configuration centroids require a complete evaluated manifold;
they are not resolved experimental lines. Source fields and uncertainty remain
separate from theoretical residuals.

## Calculation ledgers and maintenance

- `generate_full_dhf_nist.py` and `generate_multielectron_cache.py` create and
  populate calculation records. `maintain_full_dhf_workers.py` manages workers.
- `audit_dhf_*.py`, `audit_first20_dhf.py`, `audit_inverted_dhf_lines.py` and
  `summarize_dhf_*.py` inspect convergence, branches and retained repair evidence.
- `probe_dhf_damping_release.py` tests an alternative iteration history;
  `apply_dhf_damping_probe.py` accepts a checked result into a ledger.
- `repair_dhf_*.py`, `repair_cu_ii_4d_branch.py`,
  `add_nearby_ionic_ground_candidates.py`, `refresh_dhf_state_counts.py` and
  `supervise_*.py` modify or maintain campaign records. Select the intended
  archive and ledger explicitly when revisiting an older campaign.

## Examples and developer tools

- `demo.ipynb`: an unexecuted silver ground-state example.
- `density_gs.py`, `energies_gs.py`: ground-state comparisons with external data.
- `spectra_gallery.py`: spectral synthesis examples and a generated gallery.
- `check_determinism.py`, `perf_dashboard.py`, `fac_branch_inventory.py`:
  numerical reproducibility, performance reports and angular coverage.
- `import_complete_bethe_log.py`: reproduce the retained Bethe-log table.

Reusable benchmarks and their external fixtures live in `benchmarks/`.
Generated calculation data belong in `output/`; scientific test fixtures stay
with their tests. See [testing](../docs/TESTING.md) for the verification boundaries.
