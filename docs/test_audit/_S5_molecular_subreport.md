# S5 subreport — molecular Auger test slice

Slice: `tests/test_molecular_auger_spectrum.py` (846 lines, **22 test functions**, 86 collected
cases), `tests/test_molecular_auger_initial_state.py` (553 lines, **28 test functions**, 54
collected cases), `tests/test_molecular_auger_hyp1066_regression.py` (303 lines, **7 test
functions**, 8 collected cases). Total **57 test functions / 148 collected cases**; measured run
`146 passed, 2 skipped in 1.53 s` (the 2 skips are `pytest.importorskip("pyscf")`). Every test
function body was read, plus `src/atomx/scattering/autoionization/molecular.py` (1393 lines),
`src/atomx/providers/contract.py`, `docs/adr/ADR-0001-electronic-structure-provider.md`, the
fixture `tests/data/molecular_auger_hyp1066_cleanroom.json`, and `git log -S` provenance for
`804.456`, `0.73`, `spin-forbidden-non-soc-v1`, `19.5`, `cleanroom`.

## 1. Verdict

This slice is a large, carefully written **provider-contract / fail-closed suite** (40 of 57
functions are F-class: argument validation, capability gates, budget guards, provenance strings)
that contains almost no gate on the Auger *rate*. Only a handful of assertions in the whole slice
can fail for a physics reason: the synthetic direct/exchange interference test (3.0/1.0 → 4.0 vs
10.0), the S-metric orthonormality identity, the two-hole CSF count, the Koopmans final-state
energy, the single-global-scale sum-rule test, and the non-positive-kinetic-energy drop guard. The one file
that claims external physics, `test_molecular_auger_hyp1066_regression.py`, is
**D-class masquerading as B**: its ERIs are injected as `sqrt(raw_intensity)` so the only real
kernel step (the antisymmetrized `direct − exchange` amplitude) is a no-op — measured
`exchange = 0.000000` on every one of the 11 fixture lines — and the acceptance numbers are the
fixture's own inputs reproduced by construction. I measured 804.455999999798 eV for the "Ne KLL
anchor" against the fixture's acceptance of 804.456 eV (the fixture's `ε_2p =
-2.7183936399306443` is the exact solution of `804.456/HARTREE_EV = 29.563212720146137 Ha`), and
498.49999999987483 eV for the "H2O O KLL envelope centroid", whose branch in fact survives as a
**single stick** after the other two declared hole pairs are dropped. The branch ratios asserted
as `[0.73, 0.23, 0.04] ± 0.10` are exactly the fixture's injected raw intensities (measured
`{'KL1L1': 0.04000000000000001, 'KL1L23': 0.22999999999999998, 'KL23L23': 0.73}`), the Ne Σraw is
exactly 1.0 so the sum-rule normalisation is a no-op there, and the file's own provenance states
the data are "Atomx self-generated" while citing a tolerance baseline commit (`5aec6de7`) that
**does not exist in this repository** (`git cat-file -t 5aec6de7` → `fatal: Not a valid object
name`). The only test that reaches real PySCF physics asserts nothing but provenance strings (and
`all(...)` over an empty line tuple is vacuously true), and it silently skips here. Two physics
claims are actively enshrined that I believe are wrong: triplet two-hole states declared
"spin-forbidden" (`spin-forbidden-non-soc-v1`) and the same-spatial-orbital channel silently
zeroed (its fixture avoids this only by splitting 2s into two labels with occupation 2.0 each, so
the Ne payload carries 11 electrons for Z=10).

## 2. File table

| file | tests (collected) | class mix | worst sev. | note |
|---|---|---|---|---|
| `tests/test_molecular_auger_spectrum.py` | 22 (86) | `A3 D1 E1 F17` | **S1** | kernel-level interference test is genuine, but the sum-rule test is a 1-line tautology, the non-finite ERI guard never reaches the module, and the triplet-zero rule is enshrined as physics. |
| `tests/test_molecular_auger_initial_state.py` | 28 (54) | `A4 E1 F23` | **S3** | construction/selection contracts; three good first-principles checks (S-metric orthonormality, CSF counting, Koopmans energy) but no rate or line-strength assertion at all. |
| `tests/test_molecular_auger_hyp1066_regression.py` | 7 (8) | `A1 D2 E4` | **S1** | "external" anchors are the fixture's own inputs; branch ratios/energies reproduce by construction; m-independence is fixture symmetry; PySCF path asserts nothing and skips. |
| **total** | **57 (148)** | **`A8 D3 E6 F40`** | **S1** | 70 % of the functions (and a measured 102 of 148 collected cases, `pytest -k` on the validation names) are contract/argument validation. |

## 3. Findings

**F1 (S1) — `tests/test_molecular_auger_hyp1066_regression.py:158-171` — the "registered" Ne KLL
energy and branch ratios are the fixture's own inputs.** Asserted: `kinetic_energy*HARTREE_EV ==
approx(804.456, abs=0.05)` and branch sums `== approx([0.73, 0.23, 0.04],
abs=acceptance["branch_tolerance_absolute"]=0.10)`. The literals come from
`tests/data/molecular_auger_hyp1066_cleanroom.json:96-100`, and the fixture's provider block
(lines 44-52) sets `initial_state_energy_hartree = 35.0`, `ε_core = -35.0`, `ε_2p =
-2.7183936399306443`, so the code's own Koopmans arithmetic (`molecular.py:1013-1019`, `1236`)
gives `29.56321272013871 Ha × 27.211386245981167 = 804.455999999798 eV` — the target to 2e-10 eV
— and `804.456/HARTREE_EV = 29.563212720146137 Ha`, i.e. `ε_2p` was solved *from* the target.
The branch ratios are equally constructed: the fixture injects `eri[core, ε, i, j] =
math.sqrt(raw)` (test lines 55-57), `exchange = eri[core, ε, j, i] = 0` for every injected pair,
so `|direct − exchange|² = raw` exactly; measured `{'KL1L1': 0.04000000000000001, 'KL1L23':
0.22999999999999998, 'KL23L23': 0.73}` with 10 lines / 15 dropped. **Class D** (would-be B: the
JSON cites "Koerber and Mehlhorn 1966" and "Tenorio et al., JCTC 2022" but quotes no number, and
the cited baseline commit `5aec6de7` is absent from the repo). Remediation: mark the test
`@pytest.mark.snapshot`, rename it `test_ne_kll_fixture_reproduces_fixture_inputs`, and add a
real B-test that quotes Koerber & Mehlhorn's KL23L23 ¹D₂ / ³P line energies in the test body and
compares them against a spectrum built from a real `build_core_hole_delta_scf_payload` (the
adapter already exists, see `tests/test_pyscf_adapter.py:364`), with the per-line source recorded
per column in the fixture.

**F2 (S1) — `tests/test_molecular_auger_hyp1066_regression.py:223-236` — "m-independent
degenerate sublevel sum" is fixture symmetry, not a symmetry check.** Asserted:
`set(pp_sums) == {-1, 0, 1}` and `len({round(v, 12) for v in pp_sums.values()}) == 1`. The
fixture gives all three 2p–2p pairs the *same* injected raw intensity `0.24333333333333335`
(JSON lines 89-91), so each m-bin receives two equal lines: measured 0.486666… each (2 ×
0.243333…). `mo_metadata` is read only by this test (grep: no occurrence in `src/`), and the
kernel has no m-dependence to break, so any wrong-but-symmetric intensity assignment passes.
**Class E.** Principle being claimed: m-degeneracy / rotational invariance of a closed-shell
manifold. Remediation: perturb the fixture (`MO3,MO4 → 0.30`, `MO4,MO5 → 0.18`) and assert that
the *sum over a degenerate manifold* is what stays invariant when the quantisation axis is
rotated in the provider payload; failing that, delete the test (it gates nothing).

**F3 (S1) — `tests/test_molecular_auger_spectrum.py:319-321` —
`test_triplet_two_hole_state_is_spin_forbidden_zero_amplitude` enshrines a false selection
rule.** Asserted: `amplitude.amplitude == approx(0.0)`, `coherent_intensity == approx(0.0)`,
`selection_rule == "spin-forbidden-non-soc-v1"` (re-asserted as a required dropped reason at
`..._hyp1066_regression.py:204`). The initial state is the ²S core-hole state (occupation_after
= 1.0 in every payload); a two-hole triplet (S=1) couples with the ε(1/2) continuum spin to
total S = 1/2, so it is *allowed*, and the ³P KLL lines are among the strongest in the Ne KLL
spectrum the fixture claims to reproduce ([Koerber & Mehlhorn, *Das K-Auger-Spektrum von Neon*,
Z. Phys. 1966](https://www.semanticscholar.org/paper/Das-K-Auger-Spektrum-von-Neon-K%C3%B6rber-Mehlhorn/a91829e60aa5c29d1c91ccd9401619a051238599);
see also the Ne 1s-ionisation Auger chapter in [Schmidt, *Electron Spectrometry of Atoms using
Synchrotron Radiation*](http://core-varnish-new.prod.aop.cambridge.org/core/books/abs/electron-spectrometry-of-atoms-using-synchrotron-radiation/auger-spectrum-of-neon-following-1s-ionization/148F44BC2F759693E09465E6C2CF1E07)).
Measured consequence in the fixture: all 10 distinct-pair triplets are dropped
(`Counter({'spin-forbidden-non-soc-v1': 10, 'zero_coherent_intensity': 5})`), i.e. the entire
triplet third of the spectrum is zero by fiat. **Class D** (a frozen v1 decision presented as
physics). Remediation: rename to
`test_triplet_two_hole_state_amplitude_not_yet_implemented`, keep the fail-closed behaviour but
document it as a capability gap, and add the physics gate for the implemented branch: for
distinct holes `singlet ∝ (d − e)`, `triplet ∝ (d + e)`, with the Pauli zero applying only to the
same spatial orbital.

**F4 (S1) — `tests/test_molecular_auger_spectrum.py:426-444` — the sum-rule normalisation is
asserted on a one-line spectrum, and the same-orbital channel is silently dropped.** Asserted:
`sum(spectrum.intensities) == approx(1.0)`, `len(spectrum.lines) == 1`,
`lines[0].raw_intensity == approx(4.0)`, `lines[0].intensity == approx(1.0)`,
`all(line.kinetic_energy > 0.0)`. Measured: exactly one line `(('MO1','MO2'), 'singlet', KE=18.5,
raw=4.0, I=1.0)` and three drops `MO1^-2 singlet` / `MO1^-1 MO2^-1 triplet` / `MO2^-2 singlet`.
The deliberately injected `eri[0,3,1,1] = 2.0` (line 426) can only ever produce a
`zero_coherent_intensity` drop because the kernel computes `direct = exchange = 2.0` for
`hole_i == hole_j` (`molecular.py:1211-1216`), and the test never asserts that drop. A single
line normalised by its own total is 1.0 for *any* amplitude, so the "sum-rule normalization"
claim (A-class) is not gated here; the same is true at line 522
(`test_compute_molecular_auger_accepts_prepared_payload`, 1 line). **Class A (assertion),
severity S1 (vacuous).** Remediation: assert on the ≥3-line 5-MO payload already used at line
458 that `ΣI = 1` and `I_i = raw_i/Σraw` with `raw_i = (d_i − e_i)²` computed by hand from the
injected ERIs; add an explicit assertion on the `zero_coherent_intensity` drop of the
same-orbital pair, which is precisely the physically suspicious case (see F5).

**F5 (S1) — the same-spatial-orbital hole channel is identically zero in the kernel and no test
says so; the fixture hides it by construction.** `molecular.py:1211-1216` computes `direct =
Σ_e c_e·ERI[core,e,i,j]`, `exchange = Σ_e c_e·ERI[core,e,j,i]`, `amplitude = direct − exchange`;
for `i == j` these are the same array element, so every same-orbital two-hole singlet is dropped
as `zero_coherent_intensity` (measured: 5 of 15 Ne pairs, 2 of 3 H2O pairs; the file-1 synthetic
case drops `MO1^-2` and `MO2^-2`). Physically the equivalent-hole channel is not forbidden — the
spin-adapted combination for an equivalent pair is the `2J − K` form (direct appears twice), so
the KL1L1 (2s⁻²) line survives in the observed Ne KLL spectrum. The fixture avoids the issue by
labelling the two 2s spin-orbitals as two *different* MOs (`2s_a`, `2s_b`, JSON lines 37-38, 75-76)
each with `occupation = 2.0`, giving a payload with **11 electrons for Z = 10** (Σocc = 1+2+2+2+2+2
= 11; the H2O case sums to 5 electrons for a 10-electron molecule) — `ElectronicStructurePayload.
validate()` has no electron-count check, so this passes. **Class D/E (the tests never touch it);
S1.** Remediation: add a test that a same-spatial-orbital hole pair with `J = K` yields the
`2J − K` amplitude (non-zero), and add a payload sanity assertion `Σocc == Z − 1` (or
`electron_count() == nelectron − 1`) to the fixture tests.

**F6 (S1) — `tests/test_molecular_auger_spectrum.py:525-537` — the "non-finite ERI" guard fires
in the *test helper*, before any molecular kernel.** `pytest.raises(ProviderContractError,
match="ERI values must be finite")` wraps `_payload_with_eri(eri)`, whose `TwoElectronIntegrals`
constructor validates at `src/atomx/providers/contract.py:624-677`; the monkeypatched
`molecular.AugerStick` (line 534) is unreachable, so the test passes even if
`molecular_auger_spectrum` performed no finiteness check at all. The name
(`..._rejects_nonfinite_mo_eri_before_stick`) and the monkeypatch both claim a module-level guard
that the test does not exercise. **Class E.** Remediation: construct the payload with finite
ERIs, then poison the array in place (`eri[0,3,1,2] = np.nan` after construction) and call
`molecular_auger_spectrum`, asserting the failure comes from `_assert_eri_finite`
(`molecular.py:1108-1110`); drop the inert monkeypatch and fix the name.

**F7 (S1) — `tests/test_molecular_auger_hyp1066_regression.py:259-303` — the only real-physics
path asserts nothing physical and silently skips.** Asserted:
`provider.calls == [(core_hole_spec, True)]`, `spectrum.axis == "kinetic_energy"`,
`all(line.provenance.provider == "pyscf" for line in spectrum.lines)`. The `all(...)` is
vacuously true for an empty spectrum, so an all-dropped result passes; no intensity, energy,
branch ratio or line count is checked even though this is the one test built on a genuine RHF +
ΔSCF core-hole payload. It also uses `DiscreteMOContinuumChannel(mo_index=4)` for Ne/sto-3g,
which is an **occupied** 2p MO (the channel only checks occupancy when `mo_index is None`,
`molecular.py:246-255`) and nothing flags it. Measured: `SKIPPED [2] ...
could not import 'pyscf': No module named 'pyscf'` — the gate is currently zero. **Class E,
S1.** Remediation: assert physical content on this path (line count = number of singlet
distinct-pair CSFs, KE of the strongest line vs the ¹D₂ anchor, `ΣI = 1`), assert the continuum
MO is virtual (`occupations[mo_index] <= 1e-8`) or that an occupied index is rejected, and make
pyscf a required dependency of the physics gate instead of `importorskip`.

**F8 (S2) — `tests/test_molecular_auger_hyp1066_regression.py:175-196` — the H2O "main
envelope" is a single stick inside a 30 eV window.** Asserted: `480.0 <= centroid <= 510.0`
(`envelope_window_eV`), `centroid == approx(498.5, abs=1.0)`, `ΣI = 1`. Measured: the H2O case
has **1 line** — `(('MO1','MO2'), 'singlet', KE = 498.49999999987483 eV, raw = 0.6, I = 1.0)` —
and 3 drops, because the fixture's other two declared pairs (`MO1,MO1` and `MO2,MO2`, raw 0.2
each) are same-orbital and therefore cancel (F5). A "centroid" and an "envelope window" are
checked against a single position, so envelope width/shape — the physics that the Tenorio et al.
comparison is about — is not tested. **Class D.** Remediation: build the H2O case from ≥3
surviving lines with distinct final-state energies, assert the envelope width/FWHM as well as the
centroid, and quote the reference envelope value from the paper in the test body.

**F9 (S3) — `tests/test_molecular_auger_hyp1066_regression.py:133-141` — the provenance test is
a JSON string lint that cannot detect the fixture's real defect.** Asserted:
`provenance["issue"] == "HYP-1066"`, `"no NIST SRD" in data_policy`, `energy_unit == "eV"`,
`intensity_convention.startswith("global sum-rule")`, `len(references) >= 2`,
`"sum-rule-total-stick-normalization" in approximations` — i.e. it verifies that the file
describes itself as external while JSON line 7 says the values are "Atomx self-generated".
**Class E.** Remediation: replace with a machine-checkable provenance record (one row per
acceptance number with `source`, `table/figure`, `value`, `unit`, and the code+version that
produced it), and fail if a row's `source == "atomx"` is used to justify a B-class assertion.

**F10 (S3) — `tests/test_molecular_auger_hyp1066_regression.py:208-210` — the "energy
conservation" assertion restates the line's own definition.** `line.kinetic_energy ==
approx(line.initial_state.initial_energy - line.final_state.final_energy)` is the exact
expression that produced it (`molecular.py:1236` and `:1347`), on the same stored objects, so it
cannot fail. **Class E.** Remediation: recompute the expected kinetic energy independently —
e.g. `KE = ε_core − ε_i − ε_j` from the raw orbital energies plus the tracked core index, as done
in `test_molecular_auger_initial_state.py:468` — and additionally assert that the continuum
channel was evaluated at that energy (the `ContinuumChannel.coefficients(payload,
kinetic_energy)` contract currently has no test that the argument is the line's KE).

**F11 (S2) — `tests/test_molecular_auger_spectrum.py:324-370, 382-421` — the AO→MO ERI
transform is only ever tested for *rejection*, never numerically.** The three AO tests assert
`ProviderContractError` messages and that `np.einsum`/`np.isfinite` were not reached; nothing in
this slice (or anywhere: `_mo_eri` appears in tests only in this file, and only as a monkeypatch
target) checks the transform value `np.einsum("pa,qb,rc,sd,pqrs->abcd", C, C, C, C, eri,
optimize=True)` (`molecular.py:1130`). A sign/index/ordering error in the AO→MO transform, which
is the only numerical kernel on the AO path, is invisible. **Class F coverage gap → S2.**
Remediation: add a hand-computed case (e.g. 2 AO/2 MO with a 45° rotation and a single analytic
`(pq|rs)` element) asserting the transformed `(ab|cd)` matrix element equals the value derived by
hand; or an AO→MO→AO round-trip identity.

**F12 (S4) — `tests/test_molecular_auger_spectrum.py:302` — assertion on deliberately wrong
code.** `amplitude.rate_sum_intensity == approx(10.0)` pins
`AugerTransitionAmplitude.rate_sum_intensity`, documented at `molecular.py:273-275` as "Incorrect
incoherent reference, exposed only for tests/audits" and used by no production path. Asserting it
counts as coverage without gating anything. **Class E.** Remediation: delete the property and the
assertion (the ADR-0001 I3 rule is already gated by the `coherent_intensity == 4.0` line), or
keep it behind an explicit `# ADR-0001 I3 counter-example` comment in a test named
`test_incoherent_rate_sum_is_not_used`.

**F13 (S4) — `tests/test_molecular_auger_initial_state.py:531-553` — constructor round-trip.**
`test_two_hole_final_state_accepts_non_koopmans_energy_model_stamp` asserts `state.energy_model
== "cluster-ci"`, i.e. the argument echoes back; the only code path is the non-emptiness check at
`molecular.py:167`. **Class E.** Remediation: delete, or assert the behaviour that `energy_model`
is supposed to change (e.g. that a non-Koopmans stamp suppresses the `independent-hole-koopmans`
approximation string, which currently is *always* added at `molecular.py:1027` — a real,
untested inconsistency).

**F14 (S4) — duplicated validation coverage dominates the slice.** File 1 lines 555-805
(11 functions, 72 collected cases: 555, 580, 599, 620, 641, 668, 703, 750, 776, 793, 801) and
file 2 lines 146 and 276-350 (5 functions, 30 cases) re-test the same
`_validate_active_space_parameters` helper through four entry points
(`compute_molecular_auger`, `molecular_auger_spectrum`, `local_valence_active_space`,
`molecular_auger_final_states`). The coverage is legitimate (it is the fail-closed contract), but
a measured 102 of the 148 collected cases in this slice (`pytest -k` selection of the validation
names) say nothing about Auger physics; the file table must not be read as 148 physics tests.
Remediation: parametrise the shared validator once and keep one entry-point-specific ordering test
per public function.

## 4. Model tests (worth imitating)

1. **`tests/test_molecular_auger_spectrum.py:285` `test_direct_exchange_combined_at_amplitude_level_not_rate_level`** —
   the only test in the slice that separates the coherent amplitude from the incoherent rate
   (`|3−1|² = 4` vs `3²+1² = 10`) with hand-chosen ERIs whose mapping to `direct`/`exchange` is
   asserted explicitly; an ADR-0001 I3 regression fails it. (Its weakness is the sign convention,
   see F5 — but the *structure* is exactly what the other 56 tests lack.)
2. **`tests/test_molecular_auger_initial_state.py:400`
   `test_local_valence_active_space_accepts_nonidentity_overlap_contract`** — constructs
   `S = diag(1,4,9,1)` and `C = diag(1, 1/2, 1/3, 1)` so that `Cᵀ S C = I` holds *by hand*, which
   proves the kernel uses the S metric rather than the Euclidean one; it fails the moment the
   metric is dropped. The reason is named in the test's own setup, not read from atomx.
3. **`tests/test_molecular_auger_initial_state.py:412`
   `test_two_hole_csf_count_matches_closed_shell_spin_coupling_formula`** — the expected count
   `n + 2·(n(n−1)/2)` is an independent combinatorial result (4 CSFs for n=2) and catches a
   missing or duplicated spin coupling; only the enumeration-order list `["singlet","singlet",
   "triplet","singlet"]` is an implementation literal.
4. **`tests/test_molecular_auger_initial_state.py:456`
   `test_two_hole_final_states_stamp_i2_i4_i5_provenance_and_energy`** — writes the named
   independent-hole Koopmans model out in full, with signs:
   `19.5 + (−20.0) − (−1.0) − (−0.5) + 0.1`, so a sign error in `molecular.py:1013-1019` fails.
5. **`tests/test_molecular_auger_hyp1066_regression.py:239`
   `test_spectrum_weight_sum_rule_uses_no_per_peak_scaling`** — asserts the *property* that all
   lines share one global scale (`I_i/raw_i` identical to 12 decimals) plus `ΣI = 1`, which is
   the correct form of the sum-rule claim and would fail on per-peak scaling or a `normalize`
   slip (it is the only assertion in file 3 that atomx could plausibly break for a physics
   reason).
6. **`tests/test_molecular_auger_spectrum.py:808`
   `test_non_positive_kinetic_energy_line_is_dropped_and_reported`** — enforces the kinematic
   domain (`E_i − E_f ≤ 0` ⇒ no line) and asserts the machine-readable drop record, catching a
   sign flip in the energy model rather than a literal.
