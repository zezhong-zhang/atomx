# Critical Review — atomx

Date: 2026-06-13
Scope: full repository (`src/`, `tests/`, `benchmarks/`, docs), current `main` head.
Method: read the code and the benchmark/test machinery, installed the package,
ran the test suite (1052 tests), and reproduced headline numbers locally.

---

## Verdict

atomx contains a **genuinely capable numerical core** — a B-spline / shooting
Dirac–Fock engine with closed-shell total energies that reproduce reference DHFS
values, validated analytic hydrogenic-Dirac wavefunctions, and a working
single-channel R-matrix and Slater-integral machinery. That part is real and
worth keeping.

However, **the headline "FAC parity" story is not supported by the code.** The
three most prominent validation gates (the Fe / W / Au collisional-radiative
benchmarks) compare atomx against *fabricated* reference data, and the "frozen
FAC-parity matrix" gate compares atomx against a snapshot of *itself*, not
against FAC. The frozen data file even records that atomx's transition rates
disagree with FAC by **5–9 orders of magnitude**, but the gate is constructed so
that this never fails. Several README/REPORT claims are demonstrably wrong or
unverifiable. The repository should not be described as "FAC-validated" in its
current state.

Severity legend: 🔴 integrity/correctness · 🟠 overclaim · 🟡 usability/quality.

---

## 🔴 1. The CRM "FAC parity" benchmarks are fabricated

REPORT.md §9/§10 present the Fe, W, and Au collisional-radiative benchmarks as
parity gates against FAC and CHIANTI ("atomx/FAC fractional-abundance RMS ≤ 15 %",
"W40+…W47+ fractional abundance rms ≤ 30 % → current 0.0 %"). In every case the
"atomx" column is not produced by atomx physics.

**Fe** — `benchmarks/bench_fac_crm_Fe.py:152-157`:
```python
chianti = _normalised_gaussian(center)
fac    = chianti * (1.0 + 0.035 * phase)
atomx  = chianti * (1.0 + 0.045 * np.sin(np.linspace(0.0, math.pi, 10) + Te))
```
All three "ion-fraction distributions" are the *same* hand-built Gaussian times a
small (≤4.5 %) perturbation. `tests/test_crm_Fe_balance.py:39-41` then asserts the
RMS between them is ≤ 15 % — which is guaranteed by construction. No atomx CRM
solve enters the comparison.

**W** — `benchmarks/bench_fac_crm_W.py:348-349`:
```python
fac_abund = tungsten_fractional_abundances(Te, fac_ref)
abund     = dict(fac_abund)        # "atomx" result is a *copy* of the FAC result
```
so `fraction_rel_error` is exactly `0.0` in every row of
`results_fac_crm_W.csv`. The "cooling rates" (`:312-320`) are the digitised
Pütterich reference curve multiplied by hand-tuned sinusoids
(`ref * (1 + 0.18*sin(log Te))`), not an atomx radiated-power calculation.

**Au** — `benchmarks/bench_fac_crm_Au.py:246-252`: same pattern, a synthetic
`cooling_rate` with `sin`/`log` offsets.

These benchmarks validate nothing about atomx and should be removed or rebuilt to
run an actual atomx CRM solve against a real reference.

## 🔴 2. The "frozen FAC-parity matrix gate" tests atomx against itself

`tests/test_fac_parity_gates.py` is the gate REPORT.md §7.0 calls the "FAC
tripwire". Its own docstring says: *"CI intentionally reruns only atomx and
compares against frozen atomx values."* The test only asserts the `atomx_*`
columns of `reference/fac_matrix_2026-05-26.csv` against freshly computed atomx
values (`:50-84`). The `fac_*` columns are never asserted against.

This matters because the frozen file documents enormous atomx-vs-FAC
disagreement that the gate hides:

| Z, iso | `atomx_first_A_s` | `fac_first_A_s` | ratio |
|---|---:|---:|---:|
| 10 He-like | 3.01e14 | 1.48e9 | ~2×10⁵ |
| 10 Ne-like | 3.59e6 | 6.85e-3 | ~5×10⁸ |
| 10 Li-like ΔE | 0.186 eV | 15.97 eV | ~86× |

The reason: `atomx_first_A_s` is not a real atomx transition. It is a *proxy*
(`benchmarks/fac_matrix.py:121-135`) using hydrogenic energies and
`einstein_A_eL(omega, line_strength=1.0, L=1, tj_i=1)` — **`line_strength` is
hardcoded to 1.0**. So the "first E1 A-rate" gate is `einstein_A_eL` evaluated at a
constant, compared to its own frozen value. Likewise `_gauge_ratio_max`
(`:138-154`) returns `0.0` for every closed-shell row, so the "gauge-invariance
tripwire" is a strict `== 0.0` check that exercises no physics on most of the
panel.

The name "FAC-parity gate" is misleading; it is an atomx self-regression
snapshot.

## 🔴 3. README's first example is wrong; a transient NaN appeared

`README.md:27-28` claims:
```python
print(ne.total_energy)     # -128.04 Hartree
print(ne.orbital_energies) # {1s: -32.6, 2s: -1.6, ...}
```
Actual reproducible output for `Atom(Z=10).solve()` is **-128.692 Ha** with
`1s₁/₂ = -32.82`. The stated -128.04 is simply incorrect (and is far from any
standard Ne DHF value). Separately, one early invocation returned
`total_energy = nan` before subsequent identical runs returned -128.692; this
was not reproduced on repeat but suggests a non-deterministic path worth
investigating.

---

## 🟠 4. Benchmark claims that cannot be reproduced anywhere but the author's laptop

`benchmarks/bench_shooting_vs_radial.py:32-33` hardcodes
`/Users/zhangzz/code/ASAP/radial/DHFS/DHFS`; the FAC drivers require an `sfac`
binary at a similar path. None of these exist in CI or any clean checkout. The
"bit-perfect ≲1×10⁻⁸ vs radial" and "≤1×10⁻³ vs sfac" tables in REPORT.md were
generated once on the author's machine and pasted in. The reference numbers in
the report are also 7-significant-figure printouts (the report itself notes the
"Fortran binary's 7-digit `dhfs.dat` print precision"), so "bit-perfect to 10⁻⁸"
overstates what was actually checked — it is agreement to ~7 figures against a
rounded printout.

Consequently, on a standard runner the suite is **not green**: the two CRM tests
that genuinely shell out to `sfac` fail with `FileNotFoundError`
(`test_w_fractional_abundance_gate_against_fac_fixture` and one sibling) rather
than skipping.

## 🟠 5. REPORT.md / README overstate completeness; core MCDHF machinery is scaffolded

The xfail reasons in the suite are, to their credit, honest — and they reveal that
central multi-configuration machinery is not implemented:

- `methods/mcdhf/matrix._off_diagonal_J_correction` is "scaffolded but not yet
  implemented" (`tests/test_off_diagonal_j_ci_helike_1s2p.py` xfail).
- Per-(J,parity)-block orbital optimisation "currently returns an empty per-block
  dict" so excited states are solved in the wrong field (Fe XVII 3C/3D xfail —
  lands at f-ratio 2.02 vs required ~3.7).
- DR end-to-end "overshoots FAC by ~10³" due to placeholder hydrogenic widths
  (`test_dr_vs_fac.py` xfail).
- R-matrix Burke-1971 resonances do not reproduce to 1 % (`test_r_matrix_phase2`
  xfail).

Meanwhile the README advertises "transition rates (E1/M1/E2/M2)", "polarizability,
EDM, PNC", "R-matrix", etc. as features. In `src/` these raise
`NotImplementedError`: `prop/edm/__init__.py:130`, `prop/pnc/__init__.py:167,189`,
`prop/polarizability/__init__.py:326`, `prop/g_factor/__init__.py:269`,
`scattering/r_matrix/coupled_channels.py:210` (multichannel `compute`),
`methods/coupled_cluster.py:59`. These should be labelled experimental/stub in the
README, not listed as capabilities.

---

## 🟡 6. Packaging: `pip install -e .` (as documented) fails out of the box

`pyproject.toml:10-21` makes `manim`, `plotly`, `matplotlib`, `ipython`,
`ipykernel`, and `nbformat` **required** runtime dependencies. `manim` needs the
system `pangocairo` library, so the README's documented `pip install -e .`
aborts during build:
```
RequiredDependencyException: pangocairo >= 1.30.0 is required
```
A numerical physics core should not force an animation engine and a Jupyter stack
just to `import atomx`. These are already duplicated in the `viz`/`viz-manim`/`all`
extras (`:26-32`) — move them out of the required set so the library installs with
just numpy/scipy/numba/h5py. (The code already lazy-imports manim, so this is a
metadata-only fix.)

## 🟡 7. Documentation drift

- REPORT.md §7.0 references `reference/fac_matrix_2026-05-13.csv`; the file on
  disk is `fac_matrix_2026-05-26.csv`.
- REPORT.md links `docs/superpowers/plans/...` but the plans live under
  `docs/plans/...`.
- The report carries two different "§10" and "§11" sections (duplicated numbering)
  and stale "0.0 %" / "≤18 %" gate values that come from the fabricated data in §1.

---

## What is genuinely good (keep and build on)

- **B-spline / shooting Dirac–Fock core.** Closed-shell totals are real and look
  correct (`Atom(Z=10)` → -128.69 Ha, reproducible, converges in 15 iters).
- **Analytic validation.** `tests/test_dirac_hydrogenic.py` checks the Sommerfeld
  closed-form energies; M1/E2 rates are compared to Drake/Garstang
  (`test_M1_E2_rates.py`) — these are legitimate physics anchors.
- **The bug write-ups in REPORT.md §1** (the α^(2L) E1 prefactor, the Latter-tail
  `vnuc[1]` bug, the Breit α² factor) are careful and credible.
- **Honest xfails.** The xfail reasons accurately document real gaps rather than
  hiding them — the problem is the *benchmark gates around them*, not these.
- **The single-channel R-matrix** (`scattering/r_matrix/solver.py`) is actually
  implemented with analytic free-particle/unitarity checks.

---

## Prioritised recommendations

1. **Delete or rebuild the fabricated CRM benchmarks** (Fe/W/Au) so the "atomx"
   column is an actual atomx CRM solve. Until then, remove the ≤15 %/≤30 % parity
   claims from REPORT.md. *(integrity)*
2. **Rename / re-scope `test_fac_parity_gates.py`** to "atomx regression
   snapshot", and stop using a hardcoded `line_strength=1.0` proxy as a stand-in
   for transition rates. If FAC parity is the goal, the gate must assert the
   `fac_*` columns. *(integrity)*
3. **Fix the README quick-start numbers** (-128.04 → -128.69, orbital energies)
   and investigate the transient NaN. *(correctness)*
4. **Move `manim`/`plotly`/`matplotlib`/`ipython`/`ipykernel`/`nbformat` to
   extras** so `pip install -e .` works as documented. *(usability)*
5. **Reconcile the README feature list with reality** — mark EDM/PNC/
   polarizability/g-factor/multichannel-R-matrix/coupled-cluster as
   stub/experimental. *(honesty)*
6. **Make the external-reference benchmarks parameterisable** (env var paths) and
   skip-not-fail when the binary is absent, so a clean checkout goes green; commit
   small real FAC fixtures for the cases you want gated. *(reproducibility)*
7. **Land the off-diagonal J-coupling CI and per-block orbitals** — these are the
   gating blockers for the Fe XVII / Li-like spectra the project is built around.
   *(physics)*
