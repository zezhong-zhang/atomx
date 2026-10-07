# Validation and remaining limits

All numerical runs used the existing `.venv`, this checkout's `src` and one BLAS,
OMP and Numba thread. Review-only validation used an isolated HEAD archive plus
selected review changes; the separate XFEL/state/complex-CI development stack
was excluded. The source manifest gives **delivered selected content hashes**
against base commit `9deec40`, rather than mislabel the dirty source as HEAD.
Other snapshot inputs came from that base. Earlier broad/profile runs predate
follow-up edits and are not a final full-suite pass.

| Run | Observed result | Evidence |
|---|---|---|
| Broad non-slow, before final follow-ups | 4026 passed, 12 failed, 6 skipped, 33 deselected, 19 strict xfails; 1169 s | [raw log](evidence/broad-before-followups.txt) |
| Original hotspot modules | 59 passed, 204.72 s | [profile](evidence/hotspots-before.txt) |
| Final small numerical + failure gates | 74 passed, 2.40 s | [raw log](evidence/small-final.txt) |
| Live CI/convolution/DR/plasma + retained transfers | 150 passed, 147.96 s | [raw log](evidence/live-followups.txt) |
| Angular/rates/CI/DR/QED/IO contracts | 141 passed, 1 strict xfail, 54.05 s | [raw log](evidence/operator-contract-final.txt) |
| Final quadrature consumers/architecture/solver integration | 103 passed, 1 deselected, 4 strict xfails, 24.73 s | [raw log](evidence/integration-final.txt) |
| Final sampled-nucleus/cache + FAC + original Breit conventions | 65 passed, 2.04 s | [raw log](evidence/nuclear-cache-final.txt) |
| Current full working-tree new-gate smoke | 81 passed, 3.09 s | [raw log](evidence/main-smoke-final.txt) |

Counts overlap; do not sum them into a unique suite total. The broad run's 12
failures were all omitted test migrations from `ConvolCache.clear` to
`release_shared_banks`; the complete affected module subsequently passed in the
150-test follow-up. No failure was hidden by loosening its physics gate. Final
code and consuming entry points passed focused checks; the final snapshot was
not rerun as one complete suite. Collection, Ruff and documentation checks are
recorded separately from numerical validation.

Replay from the resulting commits, in a clean checkout, with `.venv/bin/python`:

```bash
python -m pytest -q -m 'not slow' --tb=short --durations=30
python -m pytest -q tests/test_gauss_legendre_rule.py tests/test_slater_scaling.py tests/test_slater_kernel_crosscheck.py tests/test_review_fail_closed.py
python -m pytest -q tests/test_convol.py tests/test_rkb_bank_sharing.py tests/test_two_electron_replacement.py tests/test_review_structure_numerics.py tests/test_slater_scaling_transfer.py tests/test_dr_manifold_e2e.py tests/test_dr_public_heavy.py tests/test_plasma.py tests/test_plasma_physics.py
python -m pytest -q tests/test_ci_memo_reuse.py tests/test_ci_matrix_vs_mscheme_reference.py tests/test_radiative_normalization.py tests/test_angular.py tests/test_two_electron_angular.py tests/test_coupled_coulomb.py tests/test_dr_fac_reference.py tests/test_dr_radiative_ci.py tests/test_qedmod.py tests/test_scfresult_save_load.py tests/test_review_fail_closed.py
python -m pytest -q tests/test_energy_quadrature.py tests/test_eii_quadrature.py tests/test_two_electron_states.py tests/test_pi_rr_vs_fac.py tests/test_architecture_boundaries.py tests/test_numerical_design_contracts.py tests/test_ne_xfel_cascade_live.py tests/test_review_structure_numerics.py tests/test_correction_stack_injection.py tests/test_shooting_corrections.py tests/test_mcdhf.py -m 'not slow'
python benchmarks/bench_gauss_legendre.py --order 2000 --pairs 6
```

Set `PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1`
for matched numerical conditions. The fixture inputs are the retained base
`tests/data/` files. Frozen Ne cascade survival uses seed 20261005, 20000
trajectories, 16 bins and an independent Bernoulli survival probability at five
binomial standard deviations, plus probability conservation. It tests current
frozen-table dynamics, not new native rate generation or experimental XFEL data.

## Six-kernel findings retained

The independent normalized hydrogenic density is `P=2r exp(-r)`, Q=0. Scalars
agree with analytic/high-precision definition integrals under unchanged
per-backend gates. `cc_potentials` and standalone two-electron kernels retain a
half-node partial-rule bias (about 1e-6–1e-4 at the recorded native settings)
that decays under refinement. `compute_yk` has a k=2 first-cell pointwise error
about 6.4e-2, while its density contraction agrees at about 4.9e-12. A scalar
pass does not certify that pointwise consumer. Tests preserve these distinct
limitations; fixing the diagnosed partial rule is a separate physics increment.

Internal CI transcription parity does not provide independent phase proof.
Existing FAC/analytic/reference gates remain separate. Large B-spline solver
separation, additional independent references and Ne discrepancy attribution
remain outside this review delivery.

The final cache increment samples the nuclear operator using its **scalar**
callback contract at the integration nodes. An initial vectorized sampling
attempt failed 12 fixture setups because the Fermi callback accepts scalar
radii; it was corrected to the existing nuclear-matrix input contract, with
no replacement potential or fallback. The final cache/FAC/Breit run passed
65 tests. Replay: `python -m pytest -q tests/test_ci_memo_reuse.py
tests/test_fac_ci_operator.py tests/test_breit_operator.py`.

Final static validation: Ruff over `src tests benchmarks scripts` passed;
60 local documentation links resolved; 4119 tests collected from the delivery
snapshot in 5.66 s; diff whitespace checks passed. Collection validates imports,
not numerical correctness. Staged project navigation is checked per commit.
No unrelated scientific-development modules are included in this source manifest.

On 2026-10-08, recovery verified unchanged physical source and added the required
commit-time navigation prerequisite: generator, installed-hook configuration and
its six independent-repository tests (6 passed in 0.80 s). Every grouped commit
rebuilds navigation from its staged snapshot; unrelated development remains
unstaged. Commit-time EOF normalization changes no numerical implementation.
