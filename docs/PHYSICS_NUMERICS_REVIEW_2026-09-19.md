# Critical review — atomx: physics and high-performance numerics

Date: 2026-09-19
Scope: `src/atomx` (214 files / 57.2k LOC), with emphasis on **physics correctness**
and **high-performance numerics**. `tests/`, `benchmarks/`, `scripts/`, `docs/`.
Method: adversarial read **plus execution**. Every finding below carries a `file:line`,
a runnable command and the observed output. Independent references were built from
scratch (published analytic expansions, closed-form hydrogenic values, exact
Wigner symbols via sympy, high-accuracy quadrature) rather than from the repository's
own recorded fixtures.

Baseline established first:

```
$ .venv/bin/python -m pytest -m "not slow" -q
1926 passed, 53 skipped, 17 deselected, 4 xfailed in 385.60s
```

**The suite is green.** Every finding below is therefore something the test suite does
not cover — most of them because the relevant gate is self-referential (it recomputes
the implementation's own formula) or because the recorded reference is stale.

Environment: macOS/arm64, Python 3.13.2, numpy 2.4.4, scipy 1.17.1, numba 0.65.0,
editable install of `src/`. No existing file was modified; this document is the only
file added.

---

## Verdict

The **mean-field core is sound**: closed-shell DHF energies reproduce the literature
(Ne `-128.691934` Ha, i.e. the classic Desclaux value, to 7 digits), the B-spline
in-cell Gauss quadrature is exact to 2.4e-15 relative, the 3j/6j/9j machinery matches
sympy over ~140k cases with zero sign flips, and the length/velocity-gauge transition
rates agree on the same wavefunctions. The problems are elsewhere:

1. **The QED layer is wrong twice over** — the vacuum-polarisation kernel is the wrong
   special function (3.9x too large), and the tabulated self-energy used for every
   default `Z <= 36` calculation is not the function the formula requires (2.5x–14.7x
   too large, with a hard discontinuity at `Z = 37`). Both are anchored on measured
   hydrogen/uranium Lamb shifts.
2. **The high-Z B-spline grid cannot represent the K shell.** The hydrogen-like U^91+
   1s eigenvalue is **+97.7 eV** too high on the default grid (the shooting engine gets
   it right to 3e-5 eV), and the public API exposes no knob to fix it. This is the
   engine behind *all* CI/MCDHF/transition/scattering work.
3. **Several failure modes return a plausible number with `converged=True`** rather
   than raising: negative-energy-continuum collapse for unbound d/f orbitals
   (errors of ~10^4 Ha), an empty 0.0-Ha result, and the bracket ceiling of the
   shooting solver.
4. **The distorted-wave high-energy completion never runs** — the Born blend weight is
   zero at *every* energy, so collision strengths above ~5× threshold are 1–3 orders of
   magnitude too small and fall with energy where they should rise.
5. **The plasma/CRM layer has no radiative recombination at all** and a DR prefactor
   50x too small, so its ionisation balance is not physical in the coronal regime.
6. **The validation layer is thinner than the documentation implies**: the
   "7 keV / 1.6 % at Z=82" limitation advertised in README/VERIFICATION is an artifact
   of a stale committed benchmark (the real gap is 873 eV, and the sign is reversed),
   and the QED and DR "FAC parity" gates compare the implementation against itself.

Severity legend: 🔴 correctness · 🟠 numerical accuracy / performance · 🟡 hygiene.

---

## 🔴 Critical

### C1. The Uehling vacuum-polarisation potential uses the wrong kernel (3.9x–5.3x too large)

`src/atomx/corrections/qed.py:38-72`

```python
# qed.py:59-68
prefactor = -2.0 * Z * ALPHA / (3.0 * np.pi)
...
return prefactor / r_arr * bessel_k1(float(x))     # x = 2 c r
```

The point-nucleus Uehling potential is

```
V_VP(r) = -(2 Z α / 3π) (1/r) * I(2cr),   I(x) = ∫₁^∞ e^{-x t} (1 + 1/(2t²)) √(t²-1)/t² dt
```

`I(x)` is **not** the modified Bessel function `K₁(x)`. Measured with `scipy.quad`:

| x = 2cr | I(x) exact | K₁(x) | I/K₁ |
|---:|---:|---:|---:|
| 0.01 | 3.8995 | 99.974 | **0.039** |
| 0.5 | 0.4843 | 1.6564 | **0.292** |
| 1 | 0.1779 | 0.6019 | **0.296** |
| 5 | 6.902e-4 | 4.045e-3 | **0.171** |
| 20 | 3.696e-11 | 5.883e-10 | **0.063** |

The ratio never crosses 1, so `K₁` overestimates `V_VP` at every radius, by 3.3x to
160x depending on `r`. A closed-form equivalent (verified to 1.7e-12 by the delegated
audit) is `I(x) = (1+x²/12)K₀(x) − x(5/6+x²/12)K₁(x) + x(3/4+x²/12)Ki₁(x)`, whose
small-`x` limit is `K₀(x) − 5/6`, not `K₁`.

**End-to-end, anchored on the hydrogen Lamb shift** (integrating each potential against
the exact 1s density):

```
H 1s vacuum polarisation
  leading-order analytic  mc²(α/π)(Zα)⁴(-4/15) = -217.03 MHz
  numerical, exact kernel I(x)                  = -217.51 MHz
  package  atomx.corrections.qed.uehling_potential = -844.42 MHz   <-- ratio 3.88
measured H 1S Lamb shift = 8172.87 MHz
  reconstruction with the exact kernel : 8396.5 (SE) - 217.5 (VP) - 6.6 (higher order) = 8172 MHz  ✓
  reconstruction with the package's   : 8396.5 (SE) - 844.4 (VP)                      = 7552 MHz  ✗ (-621 MHz)
```

An independent literature gate confirms it at high Z: the exact Uehling factor
`F = -4/15 + (5π/48)Zα - (2/15)(Zα)²L + …` (Yerokhin & Shabaev,
[J. Phys. Chem. Ref. Data 44, 033103 (2015)](https://arxiv.org/abs/1506.01885), Eqs. 20–21)
gives `F = -0.264348`; the correct kernel reproduces `-0.264381`, the shipped `K₁`
kernel gives `-1.039` — it violates the exact `-4/15` limit by 3.9x. At Z=92 the 1s VP
is `-517.5 eV` instead of `-98.0 eV`.

**Impact.** Every default `Atom(...)` is affected (`qed=True` is the default), as is
`CorrectionStack(include_vp=True)`. The error is small in absolute terms at low Z
(Ne: -0.064 eV vs -0.016 eV) but is 3.9x on the whole vacuum-polarisation budget, which
is the second-largest QED term.

**Suggested fix.** Replace `bessel_k1` with the Uehling integral (the closed form above,
or a Bickley–Naylor evaluation), keep the `x<0.01` branch with the correct
`K₀ - 5/6` asymptote, and add the two gates `E_VP(1s, Z=1) ≈ -3.27e-8 Ha` and
`E_VP(1s, Z=92) ≈ -98.0 eV`. The docstrings at `qed.py:7,41,87-90` describe the wrong
kernel ("1/r²") and need the same correction.

---

### C2. The tabulated self-energy used for every `Z ≤ 36` calculation is the wrong quantity (2.5x–14.7x too large)

`src/atomx/data/mohr_se.py` (table), `src/atomx/corrections/qedmod/mohr.py:36-76`
(interpolation), `src/atomx/corrections/qed.py:175-181` (default policy)

```
# qed.py:177-178
if method == "auto":
    return "mohr" if Z <= 36 else "indelicato"
```

The module docstring defines `E_SE = (α/π n³)(Zα)⁴ F(Zα) m_e c²`, which in Hartree is
`(Zα)⁴F/(π α n³)` — exactly what `mohr.py:105` computes. That is also exactly FAC's
formula (`faclib/coulomb.c`: `c = (Zα)⁴/(π n³ α); r0 = r*c`), so FAC's `_qed_se0`
table is the same `F`. The two tables agree at Z=1 and then diverge without bound:

| Z | FAC `_qed_se0` → `indelicato.py` | `data/mohr_se.py` | ratio | 1s SE (FAC) | 1s SE (mohr) |
|---:|---:|---:|---:|---:|---:|
| 1 | 10.31679 | 10.3169 | 1.00 | 0.0347 eV | 0.0347 eV |
| 2 | 8.52833 | 10.6027 | 1.24 | 0.0014 eV | 0.0017 eV |
| 10 | 4.65416 | 11.6804 | **2.51** | 0.16 eV | 0.39 eV |
| 26 | 2.78194 | 13.1567 | **4.73** | 4.28 eV | 20.24 eV |
| 36 | 2.27891 | 14.0012 | **6.14** | 12.88 eV | 79.15 eV |
| 54 | 1.78193 | 15.6152 | **8.76** | 51.0 eV | 446.9 eV |
| 92 | 1.47281 | 21.6412 | **14.69** | 355.1 eV | 5218.3 eV |

**Independent verification (three ways).**

1. *Published Zα expansion.* Implementing Eqs. (7)–(19) of
   [Yerokhin & Shabaev (2015)](https://arxiv.org/abs/1506.01885) with a **single**
   calibrated remainder `G ≈ -30` reproduces the FAC table and rejects the shipped one:

   ```
   Z=1  expansion F=10.31682   (exact 10.3169)
   Z=2  expansion F= 8.52825   (FAC 8.52833)     <- shipped table would need G = +9705
   Z=10 expansion F= 4.6361    (FAC 4.65416)
   Z=18 expansion F= 3.3433    (FAC 3.4552)
   ```

   `G` drifts smoothly from −29.8 (Z=1) to ≈ −27 (Z=10), as a higher-order remainder
   should. The shipped table would require `G` to swing to `+1.3e3`.

2. *FAC source.* `faclib/coulomb.c` uses the identical prefactor and reads
   `_qed_se0[m][id] + _qed_se1[m][id]` — the numbers transcribed into
   `qedmod/indelicato.py`.

3. *Measured Lamb shift.* The total 1s Lamb shift of U^91+ is
   **460.2 ± 4.6 eV** (Gumberidze et al., PRL 94, 223001 (2005)). With the FAC table the
   budget closes (`355.1 (SE) − 22.9 (VP) + 1.7 (2-loop) + finite-size/two-loop ≈ 434–460 eV`);
   the shipped table's self-energy *alone* is **5218 eV = 11.3x the measured total**.

**End-to-end consequence** (`Atom(..., numerics="bspline")`, default `qed_backend="auto"`):

| system | default `auto` | `indelicato` | Δ total energy |
|---|---:|---:|---:|
| Ne (Z=10) | -128.613614 Ha | -128.631713 Ha | **+0.492 eV** |
| Ar (Z=18) | -527.943053 Ha | -528.200574 Ha | **+7.008 eV** |
| Fe16+ (Z=26) | -1145.518755 Ha | -1146.815551 Ha | **+35.29 eV** |
| Kr (Z=36) | -2777.783797 Ha | -2783.442199 Ha | **+153.97 eV** |

And because `auto` switches tables at Z=37, the default self-energy is **discontinuous
in Z**:

```
Z  auto backend   1s SE (eV)
34          mohr       62.225
35          mohr       70.297   jump vs Z-1: +8.07
36          mohr       79.154   jump vs Z-1: +8.86
37    indelicato       14.132   jump vs Z-1: -65.02   <-- 6.3x drop
38    indelicato       15.465
```

Two further defects ride along:

* `shabaev_local` — the default of `CorrectionStack` — is **calibrated against the wrong
  table** (`qedmod/shabaev.py:55-62`: `⟨ψ_ref|V_loc|ψ_ref⟩ = E_SE^Mohr`). Its docstring
  even records the contradiction ("Shabaev's raw tabulated `F` values differ from Mohr's
  by a factor of ≈ 2.4 for s-states at Z=10 … Because Mohr's tabulation is what atomx
  already ships, we calibrate `A_κ(Z)` directly against it") and resolves it in the wrong
  direction — the FAC/Indelicato table, the QEDMOD lineage and the published expansion all
  agree on ≈4.65 at Z=10.
* The `n⁻³` fallback (`mohr.py:74-76`, `indelicato.py:65-66`) discards all `l` and `j`
  dependence: a 4f5/2 and a 4d3/2 orbital receive *identical* self-energies, and the sign
  is wrong for high-`l` states (FAC's own `_sd` table has 3d± ≈ ±0.04, not −0.002).

**Why the suite is green.** `tests/test_qed_indelicato_vs_fac.py:33-38` builds its own
expected value from the implementation's formula *and the implementation's table*:

```python
def _fac_self_energy(Z, n, kappa):
    f_val = FAC_F[(Z, n, kappa)]
    return (ALPHA * Z) ** 4 / (math.pi * ALPHA * n ** 3) * f_val
```

so the test is named "vs FAC" but cannot detect a wrong table. `tests/test_qedmod.py:54`
(`test_hydrogenic_1s_matches_reference`) is likewise anchored on the same constant
`10.3169`. `tests/test_qed.py::TestQEDTotalShiftPin` pins `Ne +0.027863 Ha` — it pins the
wrong number.

**Suggested fix.** Either replace `data/mohr_se.py` with the point-nucleus `F` values
(cross-checked against Yerokhin–Shabaev) or delete the `mohr` backend and make
`indelicato` the single low- and high-Z source, with a pchip interpolation in Z; remove
the `auto` discontinuity in either case. Recalibrate `shabaev_local` against the
corrected table, and replace the `n⁻³` fallback with an `l`-aware form or an explicit
"not tabulated" error. Add the external gates: H 1s SE = 8396.5 MHz, and the
U^91+ 1s SE + VP budget against 460.2 eV.

---

### C3. The default B-spline grid cannot represent the high-Z K shell (+97.7 eV at Z=92)

`src/atomx/numerics/bspline/knots.py:44-100` (`make_physics_knots`), reached from
`src/atomx/atom.py` when `ns is None` (the default)

The DBSR_HF-style grid starts at `hs = hi/Z` with `hi = 0.25`, i.e. the first breakpoint
is `0.25/Z` (measured: Z=10 → 2.5e-2, Z=92 → 2.717e-3), and the docstring states
`ns` is ignored. For a relativistic 1s the small-`r` behaviour is `r^γ` with
`γ = √(1-(Zα)²)`; at Z=92 `γ = 0.741`, so the cusp below the first knot is
unrepresentable and the error scales as `r₁^(2γ)`.

Measured against the exact Dirac–Coulomb eigenvalue, same model (point nucleus),
`Atom(Z, charge=Z-1, config="1s1", nuclear="point", breit=False, qed=False)`:

```
U^91+ 1s1/2 exact = -4861.197904 Ha
  bspline default (physics knots)   -4857.606126   dE = +3.592   Ha = +97.74 eV
  bspline ns=200                    -4861.165848   dE = +3.21e-2 Ha = +0.872 eV
  bspline ns=400                    -4861.169240   dE = +2.87e-2 Ha = +0.780 eV
  bspline ks=13 ns=300              -4861.186452   dE = +1.15e-2 Ha = +0.312 eV
  shooting (the default engine)     -4861.197903   dE = +1.17e-6 Ha = +0.00003 eV
```

Same-model error across Z (default grid):

| Z | 1 | 2 | 10 | 18 | 26 | 36 | 54 | 74 | 92 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ΔE (eV) | -1e-11 | +6e-9 | +8.0e-5 | +2.6e-3 | +2.2e-2 | +0.14 | +1.63 | +14.6 | **+97.7** |
| rel | 1e-12 | 1e-10 | 5.9e-8 | 5.8e-7 | 2.3e-6 | 7.9e-6 | 3.9e-5 | 1.8e-4 | **7.4e-4** |

The error is *positive* (the level is too high) and it **does not converge with `ns`**:
the shipped exponential family plateaus at ~0.78 eV even at ns=400, because `ns` is
ignored on the default physics-knot path and both families share the `0.25/Z` inner
start. Only raising the B-spline order (ks=13) moves it, to 0.31 eV.

**Impact.** This is the engine behind the entire CI/MCDHF/transition/scattering stack —
`numerics="shooting"` is the *default* for `Atom(numerics=None)` but is unavailable for
`configs=[...]` and for everything downstream. Concretely, the same Ar-like Pb^64+
(point nucleus, no Breit/QED) differs between engines by **79.7 eV**:

```
bspline default:  -16358.7158 Ha = -445143.3 eV
shooting       :  -16361.6453 Ha = -445223.0 eV
```

For comparison, the entire 1s self-energy of U is 355 eV — a 97.7 eV basis error is the
same order as the physics the package is built to compute. Users cannot see or fix it:
`hi`/`he`/`hmax` are not exposed by `Atom`.

**Suggested fix.** Make `hi` (and/or an explicit `r_min`) a public parameter and default
it so that `hs` resolves the K shell — e.g. `hs ≈ 0.25/Z × (something like γ)` or a
Z-dependent `hi` — and add a gate on the hydrogen-like 1s eigenvalue for
Z ∈ {1, 10, 26, 54, 92} at the default grid. A convergence warning when the 1s
eigenvalue error exceeds ~10⁻⁶ relative would have caught this.

---

### C4. Silent wrong answers: negative-energy collapse, empty results, and the bracket ceiling

Three distinct paths return a plausible-looking number with `converged=True`.

**(a) Negative-energy-continuum collapse for unbound d/f orbitals.**
When a requested d/f subshell has no bound state in the mean field, the B-spline Dirac
eigensolver returns a state at `-2m_ec²` (≈ `-37558` Ha) instead of reporting failure:

```
$ Atom(Z=10, charge=-1, config="1s2 2s2 2p6 3d1", numerics="bspline").solve()
  E = -37686.422067 Ha   converged=True
  orbital_energies: 3d3/2 = -37557.7301   3d5/2 = -0.0005
$ Atom(Z=2, charge=-1, config="1s2 4f1", ...)      E = -37560.591955 Ha   4f5/2 = -37557.7301
$ Atom(Z=18, config="1s2 2s2 2p6 3s2 3p6 3d1", ...) E = -38086.413857 Ha   3d3/2 = -37557.7301
```

The trigger is an orbital that is unbound (virtual) in that ion; occupied bound 3d
orbitals (Sc, Fe, Kr, Zn) are fine. `prepare_bspline_data` already carries the cure —
`kinetic_balance=True` is documented as "eliminates spurious states"
(`numerics/bspline/basis.py:377-381`) — but the production path does not use it.

**(b) An empty result reported as converged.**

```
$ Atom(Z=1, config="7s1", numerics="bspline").solve()
  total_energy = 0.0000000000 Ha   orbital_energies = {}   converged = True
```

5s is 9.5 % off and 6s 78 % off on the same path, with `converged=True` throughout.

**(c) The shooting engine returns its own bracket ceiling.** `dbound`'s `ier` is
unpacked and never tested (`numerics/shooting/solver.py:659,666,730,737`) and
`ShootingResult.converged` mirrors only the potential-mixing flag:

```
$ Atom(Z=1, config="6s1", numerics="shooting", exchange="slater").solve()
  E = -0.013333333334 Ha   exact = -0.013888996749   rel = 4.0e-2   converged=True
$ ... config="7s1" ...      E = -0.013333333334   exact = -0.010204150943   rel = -3.1e-1
```

`-0.0133333 = -Z/r_max` with `r_max = 75.0` hard-coded (`atom.py:562,579,588`). Even the
`exchange="hf"` path silently returns the 4 %-wrong 6s value. A realistic instance: the
Na I Rydberg series 9s…13s all collapse to the same eigenvalue (`-0.0133333` Ha), with
total energies identical to 10 digits. Enlarging `r_max` to 150 fixes it
(Z=1 n=6: 4.0e-2 → -1.2e-11), so it is domain truncation, not the integrator.

Related, same class: `DFSolver` reports `converged=True, residual=0.0` when a requested
subshell has no bound state at all — the `continue` at `methods/dirac_fock.py:1590`
leaves the hydrogenic seed untouched, and `max_orb_change` stays 0, so the check at
`:1824` fires. H⁻ comes out at `E = -0.375001` Ha (HF limit −0.4879), Li⁻ and Na⁻ the
same, and an 8-electron Z=6 system returns `-150271` Ha with `converged=True`.

**Suggested fix.** Reject negative-energy eigenpairs explicitly (an eigenvalue below,
say, `-0.9 m_ec²` is never a bound atomic orbital), treat an empty orbital set as a hard
error, propagate `ier` into `ShootingResult.converged` and raise on non-zero codes, make
`r_max` adaptive (`r_max ≳ 2.5 n²/Z`) or at least warn when the returned eigenvalue
equals the bracket ceiling, and validate the returned eigenvalue's sign/energy against
the requested `n`.

---

### C5. The CRM has no radiative recombination, and the DR prefactor is 50x too small

`src/atomx/plasma/crm.py:286-410`, `src/atomx/scattering/dielectronic_recombination/solver.py:70`

`_build_rate_matrix` contains collisional ionisation, three-body recombination,
photoionisation, DR satellites, bound-bound collisions and spontaneous emission — but no
spontaneous radiative recombination. The only recombination is three-body, whose rate is
`≈ n_e²β = n_e S/Saha`, which vanishes in the coronal limit. Consequences measured
against the code's own Saha function:

```
Planck radiation field at T_rad = T_e, 2-level system, χ = 100 eV, T_e = 100 eV:
  n_c/n_b = Saha*(1 + PI/(n_e S))
  n_e = 1e18 : 98.3x above Saha        (PI/(n_e S) = 97.3)
  n_e = 1e22 : 1.0097x
  n_e = 1e24 : 1.0001x
```

True LTE requires the Planck-driven photoionisation to be cancelled exactly by
spontaneous RR at Saha populations; without the RR channel the LTE limit is never
recovered, and no coronal ionisation balance is reachable at any density.

Separately, the dielectronic-recombination prefactor is in the wrong temperature unit:

```
scattering/dielectronic_recombination/solver.py:70
_DR_PREFACTOR_CM3_S_EV_32 = 6.60305e-24     # (h²/2πmkT)^{3/2} with kT in RYDBERG
                                            # with T in eV it must be 3.312830e-22
ratio = 13.605693^{3/2} = 50.186
```

and the "FAC parity" test that appears to cover it is circular:
`benchmarks/bench_fac_dr.py::load_fac_reference` computes `alpha_FAC` with the *same*
`dr_rate_coefficient` kernel. Feeds `CRMDRSatelliteSource`, so DR satellite intensities
are 50x low.

Two more in the same layer: `crm.py:329-352` silently zeroes *both* the excitation and
the de-excitation rate when `exp(-dE/kT)` underflows (dE/kT ≳ 745), so a 2-level system
with E=[0,600] eV at T_e=0.8 eV returns populations `[0.5, 0.5]` instead of Boltzmann
(the correct `C_deexc = 1.176e-6 cm³/s`); and `plasma/average_atom.py` never iterates
the chemical potential, returning `Z̄ = 0` with `Σocc = 769.4` electrons for Fe (Z=26) at
T=0.01 eV, n_e=1e20 — 30x the nuclear charge.

**Suggested fix.** Add spontaneous RR with `α_RR(T) = PI_Planck(T)/Saha(T)` (the code's
own Kramers photoionisation cross-section already implies it via Milne), guard the
underflow by computing `C_deexc` independently, fix the DR prefactor to the eV convention
used by `saha_ionization_density_cm3`, and either make the average atom self-consistent
or raise when `Σocc > Z`. The Saha and Boltzmann *conditional* limits are exact
(2e-16 and 5.6e-16 respectively), so the framework is right — the channel list is not.

---

### C6. The Born high-energy completion in the distorted-wave excitation chain never engages

`src/atomx/scattering/distorted_wave/solver.py:302-303, 590-617`,
`src/atomx/scattering/distorted_wave/born.py:270-293`

`_geometric_tail` returns exactly `0.0` whenever the partial-wave profile has not yet
started to decay at `kappa_max` (`q1 >= q0`):

```python
# solver.py:302-303
if q0 <= 0.0 or q1 <= 0.0 or q1 >= q0:
    return 0.0
```

and `born_blend_weights` maps `tail == 0` to `w = 0` *and then propagates it downward*:
the loop runs from the highest energy to the lowest, capping each weight at the value
found for the next-higher energy (`born.py:284-292`). One zero at the top of the energy
grid therefore zeroes the completion weight **at every energy**.

Instrumented on the repository's own benchmark case (He-like Ne, `1s2 → 1s2p`,
`kappa_max=10`, `L_max=6`, the settings of `bench_fac_dw_excitation.py`):

```
recorded Born blend weights  w = [0. 0. 0. 0. 0. 0.]
recorded geometric tails     T = [1.25e-12, 2.70e-04, 1.85e-02, 0, 0, 0]
              E/threshold      Omega (shipped)   Omega with w=1 (Born)   ratio
                     1.05            0.03026                0.00737     4.11
                     2.00            0.04487                0.04012     1.12
                     5.00            0.08090                0.08228     0.983
                    50.0             0.01049                0.21238     0.0494
                   100.0             0.00412                0.28432     0.0145
                   200.0             0.00159                0.41671     0.0038
```

At 50/100/200× threshold the shipped collision strength is **20x / 69x / 263x too small**
relative to the plane-wave Born limit (which is where it should converge), and it *falls*
with energy while the correct value rises. Note the tail at 2× threshold is a healthy
2.7e-4 — the weight there is zeroed only by the monotone cap, not by `_geometric_tail`
itself.

**Why the suite is green.** The FAC parity gate samples only
`ENERGY_FACTORS = [1.05, 2.0, 5.0]` (`bench_fac_dw_excitation.py:37`), where the defect is
under 1 %. The failure is entirely in the high-energy tail that nothing samples.

**Suggested fix.** Do not let a single zero propagate: cap only across *finite* weights,
or use `max(w_i, w_{i+1})` semantics rather than `min`. Additionally, when
`_geometric_tail` bails out because the profile is still rising, that is a signal that
`kappa_max` is too small — surface it (the `0.85` cap at `solver.py:308` currently hides
it). Add a gate on `Ω(E)/Ω_Born(E)` at 50–200× threshold for one He-like ion.

---

## 🟠 High

### H0. `collision_strength_to_rate` is dimensionally wrong (~10^8 too small)

`src/atomx/scattering/distorted_wave/solver.py:784-816`

> **Correction (2026-09-19).** The first version of this finding printed an "exact"
> column of `1.837e+03 cm³/s`. That number was wrong: the formula I wrote,
> `a₀²√(2π/m)(kT)^{-3/2}∫Ωe^{-E/kT}dE`, is only dimensionally a rate when read in
> **atomic** units (a₀³/τ). Read with CGS inputs it has units `cm·s/g`, off from a rate
> by `1/E_h = 2.294e10`. The correct value is **8.011e-8 cm³/s** at T = 1 eV for
> Ω = 1, g_i = 1, confirmed independently by `plasma/crm.py`'s own
> `8.629e-6·Υ/(g_l√T_K)` and by `plasma/rates.py::maxwellian_rate`. The *defect* and its
> diagnosis stand; only the magnitude was overstated, by a factor `E_h`.

The documented prefactor `2πa₀²/(g_i√(m_e kT))` is missing both the `(kT)^{-3/2}`
Maxwellian normalisation and the electron velocity, and drops `g_i` in the
implementation. Measured against the correct Maxwellian average (atomic-unit form
`√(2π)/g_i·(kT_au)^{-3/2}∫Ωe^{-E/kT}dE_au`, converted with `a₀³/τ`):

```
T = 1 eV    : code 9.178199e-16   exact 8.011e-08 cm^3/s   ratio 1.15e-08
T = 10 eV   : code 2.895207e-15   exact 2.533e-08          ratio 1.14e-07
T = 100 eV  : code 4.141097e-15   exact 8.011e-09          ratio 5.17e-07
T = 1000 eV : code 1.690227e-15   exact 2.533e-09          ratio 6.67e-07
```

The residual scales as `T^{+1}`, i.e. `(kT)^{-1/2}` where `(kT)^{-3/2}` is
required. No in-repo caller, so this is a public-API defect only — but any user building
a rate coefficient from `dw_excitation_cross_section` gets a number that is not a rate.

---

### H1. The advertised "7 keV at Z=82" limitation is an artifact of a stale benchmark

`benchmarks/results_fac_structure.csv` (last touched 2026-04-20; HEAD is 2026-09-18),
quoted in `README.md:184`, `docs/VERIFICATION.md:41,74`

Running the repository's *own driver function* (`benchmarks/bench_fac_structure.py::run_atomx`)
with today's code and comparing to the committed atomx column:

| Z | iso | recorded atomx (eV) | FAC in the CSV (eV) | current atomx (eV) | now − FAC | doc − FAC |
|---:|---|---:|---:|---:|---:|---:|
| 26 | Ar-like | -34 007.8 | -34 028.5 | -34 007.8 | +20.7 | +20.7 |
| 54 | Ar-like | -173 801.5 | -174 596.1 | -174 782.2 | **-186.1** | +794.6 |
| 74 | Ar-like | -347 789.4 | -351 765.9 | -352 349.8 | **-583.9** | +3 976.5 |
| 82 | Ar-like | -437 028.1 | -444 036.2 | -444 909.2 | **-873.0** | **+7 008.1** |

For Z ≤ 36 the recorded column still reproduces exactly; above it, the recorded column
is 250–295 Ha stale (it corresponds to an under-converged basis, `ns ≈ 25`). The
documented limitation is therefore wrong **in the pessimistic direction**: the current
code agrees with FAC to 873 eV (0.20 %) at Z=82, not 7 008 eV (1.6 %), and the sign of
the residual is reversed. `docs/VERIFICATION.md` correctly notes this row has "No CI
gate", which is exactly why it drifted.

**Suggested fix.** Regenerate the CSV, replace the README/VERIFICATION numbers with the
measured ones (or state the envelope over the panel), and add a cheap CI gate that
recomputes the atomx column and compares against the committed values at a loose
tolerance so staleness is detected.

---

### H2. Single-configuration Einstein A uses the lower state's degeneracy

`src/atomx/properties/transitions/calculator.py:474`
(`A = einstein_A_eL(omega, S, L, tj_i)` inside `_build_single_orbital_result`)

The loop keeps `dE = E_f - E_i > 0`, so `i` is the **lower** state, but the spontaneous
rate needs `(2j_upper+1)`. The other two engines do it correctly
(`calculator.py:593`, `ci_lines.py:619`). Exact hydrogenic orbitals through the real
code path:

```
1s1/2 -> 2p3/2   S=2.21963   A(code)=1.253677e+09   A(correct)=6.268384e+08   ratio 2.0000
1s1/2 -> 2p1/2   S=1.10982   A(code)=6.268313e+08   A(correct)=6.268313e+08   ratio 1.0000
```

Both Ly-α fine-structure components have `A = 6.2649e8 s⁻¹`; the code's own `gf` in the
same row is correct, so `A` and `gf` violate `A = 6.6703e15·gf/(g_u λ²[Å])` within one
`TransitionResult`. This is the default (UTA) path for any atom solved without
`configs=[...]`. Error factor `(2j_u+1)/(2j_l+1)` — up to 4 or more, in either
direction, so it does not cancel in branching ratios.

---

### H3. The normal mass shift has the wrong sign — isotope shifts come out reversed

`src/atomx/properties/isotope_shift/mass_shift.py:136`: `E_NMS_Ha = -kinetic / mass_me`

`H_NMS = +(m_e/M)T_e` raises a bound level, so the shift must be positive; the module
negates it, while its own `specific_mass_shift` (`:215`) uses `+K_SMS/M` and
`IsotopeShiftResult`'s docstring documents the standard positive convention. Measured:

```
E_NMS(H 1s) = -1.791704 THz   (must be +1.7917)
package H-D 1s-2s isotope shift = -671.5554 GHz
measured                        = +670.994 GHz
exact reduced-mass value        = +671.5554 GHz
```

The magnitude is right to 0.1 %; only the sign is inverted, and
`tests/test_isotope.py:55-58` enshrines the wrong sign.

---

### H4. No off-diagonal Lagrange multipliers: the SCF fixed point is not the HF/DF one

`src/atomx/methods/dirac_fock.py:1563-1637`

Same-κ orbitals come out non-orthogonal and the Fock operator is not variationally
consistent with the reported energy:

```
<1s|2s> : Li 5.69e-4   Na 2.7e-4   Ne 7.0e-6   Kr 4.7e-5   (stable in tol, ns, ks, kinetic balance)
shooting engine Schmidt-orthogonalises (numerics/shooting/hf_shooting.py:658) -> 5e-17
E(Li) bspline -7.433531653  vs  shooting -7.433712066   (1.8e-4 Ha apart)
sum_a q_a<a|V^a|a> - 2 E_2e =  -1.5e-6 (Be)  -3.7e-6 (Ne)  +8.1e-4 (Ar)  +2.1e-2 Ha (Kr)
   -> total_energy != sum q*eps - E_2e, which an exact SCF must satisfy identically
   (shooting sides with the identity: Ar -528.683678 vs reported -528.684450)
```

For Be/Ne the two engines agree to 1e-9/4e-8, so the gap is specific to same-κ pairs
with unequal occupations — i.e. exactly the open-shell cases where the multiplier term
does not vanish. The `converged=True` flag is set on a residual that plateaus near 1e-10
for Li/Fe (a limit cycle of the non-orthogonal fixed point).

---

### H5. The `compute_yk` rectangle-rule cumulative biases the `slater`/`gu` exchange models

`src/atomx/numerics/bspline/evaluate.py:66-112` (`_compute_yk_flat`), used by
`methods/dirac_fock.py:721-760` (`_build_direct_potential_yk`, taken whenever
`exchange != "hf"`), `numerics/bspline/local_potential.py:159` (the `gu` model),
`scattering/distorted_wave/solver.py:248` and `scattering/autoionization/__init__.py:240`

```python
for i in range(n):                 # Outward integral
    s += w_flat[i] * rho_flat[i] * rk[i]
    i_out[i] = s
```

The cumulative includes the *full* Gauss weight of the current node on both the outward
and the inward pass, so the diagonal cell is counted twice:
`Y_code(r_i) = Y_exact(r_i) + w_i ρ_i / r_i`. This is the same O(h) bias that
`corrections/breit.py:22-31` documents and fixed for the Breit term ("rectangle-rule
cumulatives double-count the diagonal cell"), but the Hartree path was not fixed with it.
Measured against the analytic hydrogenic Y⁰ (`scipy.quad` at every node):

```
compute_yk vs exact analytic Y^0 :  max |rel err| = 2.9e-02   median = 2.5e-16
int rho * Y^0_code dr = 6.3609386658     int rho * Y^0_exact dr = 6.2516824434
exact F0(1s,1s) = 5Z/8 = 6.2500000000    -> +1.75 % error
```

Replacing `compute_yk` with an accurate reference (10x-refined grid, exact `r_<`/`r_>`
split) and re-running the SCF:

```
exchange=slater  Ne    : -128.63936139 -> -128.62336312   (-0.435 eV)
exchange=slater  Fe16+ : -1148.38270327 -> -1148.35860252 (-0.656 eV)
exchange=slater  Kr    : -2788.70767347 -> -2788.54300535 (-4.481 eV)
exchange=gu      Ne    : -128.63190495 -> -128.65827311   (+0.718 eV)
exchange=gu      Fe16+ : -1148.38603880 -> -1148.41156357 (+0.695 eV)
exchange=gu      Kr    : -2788.72159751 -> -2788.76731421 (+1.244 eV)
```

The `hf` path is unaffected (it uses the cell kernels, for which the monkeypatch changed
the Ne energy by exactly 0). The documented FAC agreement for `gu` is 4.1e-3 worst case
at Z=36 — a 1.24 eV shift at Kr is the same order, so part of the "local-exchange model
gap" is a quadrature artifact, not physics.

**Suggested fix.** Split the diagonal cell (½ weight on each side), or reuse the
trapezoidal cumulatives from `numerics/quadrature.py::yk_on_grid`, on this path too.

---

### H6. Two incompatible Wigner–Eckart phase conventions

`src/atomx/angular/coefficients.py:35-62` (`reduced_C`, carries Grant's extra
`(-1)^{j_a+1/2}`) vs `:259-277` (`ckm`, standard WE theorem), and
`scattering/r_matrix/angular_cc.py:54-64` (the non-relativistic counterpart, `(-1)^l`).

Compared against an independent exact m-scheme reduction built from sympy 3j symbols:

```
ckm vs exact m-resolved: n=1416, mismatches=292
  ckm(-1,-1,-1,-1,0,0) = -1.0   (the identity operator must give +1)
  reduced_C(-1,-1,0)   = -1.4142135624   (must be +sqrt(2j+1))
magnitudes agree to <= 1.4e-15 in all 320 (kappa,kappa',k) families
26 of 48 (kappa,kappa',k) families carry a flipped overall sign
```

So the *physics content* (magnitudes) is right and only the phase bookkeeping is not —
but `ckm` is the shared m-resolved factor for DFS/CI, distorted-wave and autoionization
two-electron elements, and the spurious sign depends on the ket's `j`, so relative signs
between direct and exchange vertices can flip. No end-to-end wrong observable was
exhibited within this review, hence High rather than Critical.

---

## 🟡 Medium

| # | finding | location | measured |
|---|---|---|---|
| M1 | Self-energy `n⁻³` fallback discards `l`/`j`: 4f5/2 and 4d3/2 get *identical* values; sign wrong for high-l | `corrections/qedmod/mohr.py:74-76`, `indelicato.py:65-66` | 4f5/2 = 4d3/2 = 0.33814 at Z=92 vs FAC `_sd` ±0.04 |
| M2 | Config parser cannot express implicit occupation 1 or `n ≥ 10` | `methods/mcdhf/config.py:247`, `config/jj_coupling.py:63` | `Atom(Z=3, config="1s2 2s")` → `ValueError: invalid literal for int() with base 10: ''`; `config="10s1"` → `KeyError: '0'` |
| M3 | `grid_slater_rk` argument packing differs from `compute_slater_rk_cell` while the docstring claims they match | `numerics/quadrature.py:343` | `grid(A,A,B,B)` = 0.8042 where F⁰(1s,2p3/2) = 2.4297 (factor 3.02); all 7 call sites happen to use the grid packing, so no wrong observable today |
| M4 | DR/UTA/Stewart-Pyatt: `stewart_pyatt_potential` is discontinuous at the branch switch (λ_eff→0 instead of (2/3)R) | `plasma/screening.py:141-188` | Z=6, Z_i=1, n_e=1e24: V(0.5 R_IS) jumps -3.98e-8 → -9.063 Ha (247 eV) |
| M5 | `uta_simple_from_levels` g_u·g_l proxy vs the real FAC Fe XVII array | `uta/` | centroid −1.20 %, σ **+68.6 %**, merged strength 93x off. **Correction (2026-09-19): this is not a weighting bug.** All three FAC `2p5 3d → 2p6` lines share `J_u = 1, J_l = 0`, so no `(2J+1)`-based weight can reproduce a *strength*-weighted moment when the true strengths differ by 240x — the proxy is a degeneracy-weighted (Unsöld-limit) UTA moment, which is the standard assumption. The numbers above are the Unsöld deviation against a strength-weighted reference, and the real defects are the contract (an optional `strengths=` argument is now offered, and `total_strength` no longer describes a degeneracy product as a line strength). `uta_moments` itself is exact |
| M6 | MBPT(2) over-correlates He by 19 % and dips below the exact ground state | `methods/mbpt.py:578-580,655` | E2 = −0.0499 Ha vs exact −0.042044 Ha; `mbpt2` and `pair_ladder` agree exactly → shared overcount |
| M7 | Missing orbital in the CI build silently yields the uncorrelated energy | `methods/mcdhf/matrix.py:673-692` | Be: E(2 CSF) == E(1 CSF) |
| M8 | Same-subshell doubles dropped in SD excitation generation | `methods/mcdhf/excitations.py:123-132` | `generate_sd_excitations('1s2', …)` returns zero doubles → `correlation="sd"` under-recovers silently |
| M9 | Hyperfine nuclear moments off; `hyperfine/__init__.py:5` has a spurious factor 2 in the A formula | `properties/hyperfine/nuclear_moments.py:116,120` | ¹³³Cs +3.1e-4, ⁶³Cu +1.8e-3 vs accepted; H 1s A = 1421.27 MHz vs 1420.4058 (+610 ppm). **Correction (2026-09-19):** the missing reduced-mass term is **(1+m_e/m_p)⁻³ = −1632 ppm**, not the −0.054 % this row first quoted (the Fermi energy goes as `m_r³`; `E_F(H) = 1418.840082(9) kHz`, Antognini/Lin/Meißner arXiv:2208.04025 Eqs. 2, 7). The ladder then closes: 1421.27 → 1418.95 → 1420.60 → 1420.40 vs experiment 1420.4058 |
| M10 | `dipole_polarizability` crashes on the default engine | `properties/polarizability/__init__.py:160` → `methods/mbpt.py:322` | `Atom(Z=1, config='1s1')` (shooting) → `TypeError: 'NoneType' object is not subscriptable`; bspline gives 4.49975 a₀³ vs exact 4.5 |
| M11 | ~~`ls_jj_transformation` returns non-square/non-unitary matrices (no parity guard)~~ — **RETRACTED (2026-09-19)** | `angular/recoupling.py:27-87` | Not reproduced. With the documented convention (arguments are twice-J) all 33 non-empty blocks for `l1,l2 ≤ 2, J ≤ 4` are square and orthonormal to 4e-15. The 22/45 figure only appears when `J` is passed where the API expects `2J` (15 rectangular blocks) — a caller-convention trap, not a missing guard. Both suggested parity guards are *vacuous* in the doubled convention (`2j1+2j2+J2` and `2L+2S+J2` are always even), so there was nothing to fix; the docstring now states the convention explicitly and a unitarity gate covers p²/d²/mixed |
| M12 | Stale CODATA-2018 literals duplicated across the tree | `numerics/quadrature.py:37`, `numerics/shooting/dirac_continuum.py:117`, `scattering/distorted_wave/born.py:162`, `plasma/average_atom.py:152`, `properties/g_factor/__init__.py:126`, `properties/zeeman/lande.py:56` | +1.4e-9 to +6.8e-10 relative vs the CODATA-2022 values `constants.py` claims to use |
| M13 | `slater_condon_f` disagrees with both its docstring and its own comment | `angular/ls_coupling.py:138-162` | returns `(2l+1)²·3j²`; f(2,1)=1.2 where the docstring says 2/15 (no internal callers) |
| M14 | `total_sum` unbound on early exits of `dbound` | `numerics/shooting/dirac_bound.py:207` vs `:243` | raises `UnboundLocalError` in pure Python, silently returns `ier=5` under numba — py/numba divergence |
| M15 | Continuum phase-shift fit omits the Coulomb log term `−η ln(2kr)` | `numerics/shooting/dirac_continuum.py:145-152` | 12.2 rad error at Z=1, k=0.271, r_max=50 — the returned `delta` depends on the box radius; DW/RR/AI discard it but the public API and its tests assert on it |
| M16 | `coulomb_bethe_topup` is dead code and drops the `g_i = 2J+1` factor | `scattering/distorted_wave/solver.py:824-863` | returns `4f/ΔE·ln` instead of `4g_i f/ΔE·ln`; Bethe coefficient 8.879 = 4·2·0.4162/0.375 confirmed against an exact GOS integral |
| M17 | `except Exception` silently disables the Born completion and the continuum waves; `E_OUT_MIN = 1 eV` clamps the outgoing electron at threshold | `scattering/distorted_wave/solver.py:468, 483-484, 511-522` | a physics failure degrades to a wrong number with no signal |

---

## Coverage caveat for the scattering layer

The collision-physics review verified the **distorted-wave / Born** chain numerically
(that is where C6 and H0 live). The four parallel sub-reviews of
`photoionization`, `radiative_recombination`, `dielectronic_recombination`,
`autoionization`, `r_matrix`, `two_electron/angular`, `polarisation`,
`charge_exchange`, `electron_ionization` and `_jit_kernels` **died without output and
were not re-run**. Those modules should be treated as *not reviewed* here; the only
evidence about them is the plasma review's DR-prefactor finding (C5) and the fact that
their in-repo gates are shape/regression tests.

---

## 🟠 High-performance numerics

Measured on `Atom(Z=26, config="1s2 2s2 2p6 3s2 3p6 3d6", numerics="bspline").solve()`,
fresh process, median of repeated runs.

**P1 — 86 % of the wall time is inside njit kernels; the real hotspots are
`_convol_case4_t` (35 % of wall) and the Slater cell integrals.** A cProfile-based
attribution is *misleading here*: numba dispatchers are C-level callables and emit no
profile events, so every kernel's time is charged to its Python caller's tottime. The
handed-over profile in this review's first pass ("40 % of runtime in Python overhead in
`_build_hf_matrix_dbsr`") was an artifact; direct scoped wall-clock timers give:

| region | calls | total s | inside njit | Python-level rest |
|---|---:|---:|---:|---:|
| `_build_hf_matrix_dbsr` | 252 | 5.590 | 5.345 | 0.245 (2.4 % of wall) |
| `_compute_slater_rk_cell_once` | 930 | 3.619 | 3.606 | 0.013 |
| whole solve | — | 10.40 | **8.96 (86 %)** | 1–2 % |

**P2 — 13.8 % of runtime is redundant recomputation of the same RKB banks.**
Only 20 distinct `(k, itype)` keys exist, but `mcdhf/solver.py:570` constructs a **new
`DFSolver` on every MCDHF outer iteration** (5 per solve), each lazily creating its own
`_convol_cache`; all 20 banks are rebuilt 5x with bit-identical SHA1. Measured waste:
0.402 s of 3.198 s at ns=60 (12.6 %), 1.44 s of 10.40 s at ns=113 (13.8 %). Worse,
`build_ci_matrix` calls `_reset_memo()` (`mcdhf/matrix.py:429`) which drops the stashed
`_convol_cache` — directly contradicting its own docstring four lines above
(`matrix.py:425`: "sharing them avoids an O(seconds) rebuild per CI construction"). Fix:
make the 20 banks process-wide, keyed on `(nv, ks, ksp, ksq, nsp, nsq, t.tobytes())`.

**P3 — `fastmath=True` costs nothing where it is used most, and the "bit-identical"
docstrings are false.** 20 kernels carry `fastmath`; comparing each against its own
non-fastmath compilation:

| kernel | ns=113 max rel | plain/fastmath time |
|---|---:|---:|
| `_convol_case4_t` (35 % of wall) | 2.6e-12 | 1.03x (neutral) |
| `_convol_case2`, `_convol_case2_mixed` | 1.3e-13 | **0.78x (plain is faster)** |
| `_convol_case1`, `_convol_case4` | 4.0e-13, 2.6e-12 | 0.82x, 0.90x |
| `_density_sym` | 8.7e-15 | 1.03x |
| `_convol_case1_t` | 7.8e-13 | **3.1x (fastmath pays)** |

Dropping it on the first six is free and restores reproducibility; `convol.py:691`
("bit-identical to the plain band construction") and `convol.py:914-921` are contradicted
by measurement. No fastmath body contains a NaN/inf guard, so this is a reproducibility
and dead-claim issue rather than a correctness one — but `scripts/check_determinism.py`
compares hashes across processes *on the same machine*, so FMA/reassoc drift is invisible
to it by construction.

**P3b — end-to-end scaling is quadratic, which is the right shape.** Interleaved
measurements over ns = 40…120 give 1.45 / 3.03 / 5.27 / 8.23 / 11.72 s, an empirical
exponent of **1.907** (local 1.82–2.00), while `eigh` grows at 1.76. Call counts are
fixed and the kernels are `O(ns² ks²)`, so nothing is accidentally `O(N⁴)`; the dominant
term at every size is the matrix builder (44–54 %), then the Slater/MCDHF path (24–35 %).
No algorithmic-complexity defect was found — the cost is in the constant factors (P1, P2).

**P3c — no benefit from BLAS threading, and nothing sets thread limits.** `scipy` is
built against Apple **Accelerate**, not OpenBLAS. Interleaved 3× runs give 9.786 s
(default) / 9.266 s (`OMP_NUM_THREADS=1`) / 9.237 s (`=4`), with cumulative `eigh` time
identical (0.552–0.568 s) at every thread count; 86 % of the wall is serial njit, which is
a documented design choice (`convol.py:734-742`, crossover at ns ≥ 290). The only
`parallel=True` kernel (`numerics/quadrature.py:232`) is off the hot path. Since nothing
in the package pins thread counts, a batch job will oversubscribe by default —
`OMP_NUM_THREADS=4` costs ≤1 % and removes that.

**P4 — the default grid is ~3x more expensive than an explicit `ns`.** Fe-like Fe at
ns=60 costs 3.16 s and at the automatic physics grid 9.17–9.90 s, for a total-energy
difference of 1.5e-3 Ha; re-running the identical call costs 8.94–9.34 s across processes.
That is a legitimate accuracy/cost trade, but it is undocumented and the exponential and
physics knot families are not nested (`ns=40…80` moves the energy by 2.3e-5 Ha *upward*,
while the default is 1.5e-3 Ha lower), so "use a larger `ns`" is not a reliable
convergence knob.

**P5 — `scripts/perf_dashboard.py --check` is not a performance gate.** It re-renders
markdown from the committed CSVs and compares text (`perf_dashboard.py:153-160`); it runs
no benchmark and compares nothing against a budget. Exit code 0 proves only that
`benchmarks/PERFORMANCE.md` is in sync with `benchmarks/*.csv`. The 34 recorded wall times
are unverified and can drift arbitrarily.

---

## Validation-integrity notes

* **Self-referential QED gates.** `tests/test_qed_indelicato_vs_fac.py`,
  `tests/test_qedmod.py::test_hydrogenic_1s_matches_reference` and
  `tests/test_qed.py::TestQEDTotalShiftPin` all recompute the implementation's own
  expression (formula *and* table constant). They detect drift, not error — which is how
  C1 and C2 survived the remediation pass that rebuilt the other gates.
* **The DR "FAC parity" test is circular** (`bench_fac_dr.py::load_fac_reference` builds
  `alpha_FAC` with the same kernel under test).
* **`docs/VERIFICATION.md` is honest about most of this**, and its "not verified by
  anything" list already names the CRM curves and the high-Z total-energy row. Two of the
  three items on that list turn out to be outright defects (C5) or a stale record (H1).
* Recorded-reference staleness is a systemic risk here: `results_fac_structure.csv` drifted
  for five months of physics changes with no gate, because the row is documented as having
  none.

---

## What was checked and found correct

So the review is bounded, these were verified positively:

* **Mean field.** Ne DHF (no Breit/QED) = `-128.691934` Ha, matching the classic
  Desclaux/FAC value; closed-shell He/Be/Ne/Ar agree with the independent shooting engine
  to ≤4e-8; convergence is variational and starting-point independent.
* **Angular algebra.** 3j over 17 207 cases (max dev 7.98e-16, 0 sign flips), 6j over
  120 649 (4.37e-16), 9j over 4 710 (1.39e-16) vs sympy; relativistic reduction
  *magnitudes* to ≤1.4e-15 over 320 (κ,κ′,k) families; m-scheme CSF counts satisfy
  Σ_J(2J+1)n_J = C(2j+1,q) exactly; CSFs orthonormal to 2.2e-16.
* **Radial quadrature.** The B-spline in-cell Gauss path matches an independent
  de Boor + exact-split double integral to **2.4e-15 relative** on all 11 F/G integrals
  tested; `numerics/quadrature.py`'s composite-trapezoid path converges at the measured
  order p = 1.99–2.00 to the same values (1.6e-6…4.2e-4 on its default grid).
* **Transitions.** Length vs velocity gauge on the same exact hydrogenic orbitals:
  −4.7e-11 (2p3/2→1s), +1.1e-10 (3d5/2→1s), −7.0e-7 (4f→1s). Einstein A prefactor with
  the *correct* degeneracy gives 6.2684e8 s⁻¹ for H 2p3/2→1s1/2 vs NIST 6.2649e8.
* **Two-photon 1s–2s in H**: 8.22883 s⁻¹ vs literature 8.2290 (−2.1e-5). Static
  polarizability of H 1s: 4.49975 a₀³ vs exact 4.5. Landé g_J exactly matches the closed
  form for 58 (L,S,J) combinations. Field shift: Seltzer F(1s) = 2/3 Ha/Bohr² exact.
* **QED-adjacent**: `constants.py` CODATA-2022 values agree to ≤4e-11; the self-energy
  *formula* and its `mc² = 1/α²` Hartree conversion are correct (verified against FAC's
  `coulomb.c` and the published expansion) — only the `mohr` table is wrong; the Uehling
  prefactor, sign, `x = 2cr` scaling and unit handling are all correct — only the kernel
  is wrong.
* **Shooting integrator**: `r→0` power-series start-up exact to 2.9e-16 relative at Z=92
  (1.5e-15 at Z=137) against a 50-digit mpmath reference; node counts exactly `n−l−1` in
  all 48 orbitals tested; hydrogenic accuracy ≤7.6e-12 relative for contained states;
  bracketing needs ≤35 shots (cap 200 never binds).
* **Plasma/spectrum**: Saha and Boltzmann conditional limits exact to 2e-16 / 5.6e-16;
  Debye length, ion-sphere radius/potential, all three IPD formulas, the free–free
  prefactor (4.5e-5 vs closed form) and the Kramers total (0.07 %), Voigt/Gaussian/
  Lorentzian normalisation and limits, Doppler/natural widths, `uta_moments` arithmetic.
* **Collision strengths (DW/Born)**: `πa₀² = 8.79736e-17` and the CODATA-2022 constants;
  `σ = πa₀²Ω/(g_i k_i²)` with relativistic `p²` (exact to 5 digits at 2043 eV); the
  `BornTransition` angular factor and prefactor match an independent exact-GOS
  integration to ≤1 % over 100 eV–20 keV, with `Σ_j reduced_C(1s,2p_j,1)² = 2.000000000000001`
  exactly as the Bethe coefficient requires; the Bethe slope `4g_i f/ΔE = 8.879` is
  consistent with the code's own oscillator strength; absolute DW `σ` at 2× threshold
  0.01305 Mb vs independent Born 0.0116 Mb (+13 %); `Ω ≡ 0` strictly below threshold;
  and the `pop = (g_b−q_b)/g_b` factor is *correct* (the He-like/H-like ratio 1.09 is the
  right relation and matches the recorded FAC fixture).
* **README quick-start** reproduces exactly: `-128.61363`, `-2.8629`, `-14.6139`,
  mixing `[0.944, 0.282, 0, 0, 0, 0.171]`, `-1145.499908` for the shooting/gu Fe16+ case.
* The full non-slow suite is green (1926 passed).

---

## Found during remediation (2026-09-19)

### N2. The multi-configuration path silently ignored `exchange` · FIXED

`Atom` constructed its `MCDHFSolver` without an `exchange` argument, so a
local-exchange model requested on a multi-configuration run was dropped in
silence while `provenance` still reported it:

```
Atom(Z=10, configs=["1s2 2s2 2p6"], exchange="hf")     -> -128.691926 Ha
Atom(Z=10, configs=["1s2 2s2 2p6"], exchange="slater") -> -128.691926 Ha   <-- identical
Atom(Z=10, config="1s2 2s2 2p6",    exchange="slater") -> -128.623437 Ha   <-- the real Slater
```

The MCDHF/CI solver implements non-local Hartree-Fock exchange only.  It now
raises a `ValueError` naming the limitation instead of running a different
model than the one recorded; the single-configuration path (which implements
all three) is unchanged.  Gate:
`tests/test_exchange_model_is_honoured.py`.

This was found by the cross-code verification pass, not by the original
review.

### N1. `correlation="sd"` returns a non-variational energy for Be · OPEN

Discovered while fixing M8 (same-subshell doubles were being dropped).  With
the missing `a²→b²` doubles restored, the Be SD space grows from 19 to 39
CSFs and the energy moves *further* below the exact ground state:

| space | CSFs | E (Ha) |
|---|---:|---:|
| `1s2 2s2` alone | 1 | -14.575892 |
| `correlation="s"` | 9 | -14.659475 |
| `correlation="sd"` | 39 | **-15.053612** |

The exact non-relativistic Born-Oppenheimer ground state of Be is
**-14.6674 Ha**, so the SD result is **0.386 Ha below it** — impossible for a
variational CI in a space that contains the singles space.  The defect
predates the M8 fix: the 19-CSF set at HEAD already gave -14.9002.

Diagnosis (not root-caused): the orbital optimisation, not the CI matrix.
In the *singles* run the `2p` orbital energy is **-2.0005 Ha**, i.e. the
hydrogenic (completely unscreened) `E_Dirac(2p, Z=4) = -2.00047`, and it is
ordered below `2s` (-0.3336).  The SD run converges elsewhere (-0.2371) and
lands below the exact energy.  So the MCDHF average over a large SD space
appears to reach a collapsed/unscreened solution.

Diagnosis narrowed (2026-09-19): the defect is in the **CI/CSF layer, not the
orbital optimisation** — switching to frozen orbitals makes it worse, not
better:

| `radial_optimization` | `correlation="s"` | `correlation="sd"` |
|---|---:|---:|
| `mcdhf` (default) | -14.659 (OK) | **-15.054** |
| `fac_frozen` | **-15.400** | **-16.358** |
| `fac_frozen_local` | — | **-15.155** |

Two further observations for whoever picks this up: the 39-configuration SD
space expands to **399 CSFs with only 99 distinct occupation vectors** (up to
16 sharing one, including core-excited `2s2 2p2`), and in the
`correlation="s"` run the `l > 0` virtuals sit at their *hydrogenic* energies
(2p = -2.0005 against `E_Dirac(2p, Z=4) = -2.00047`) while `3s` is properly
screened at -0.0971.

Status: recorded, not fixed — root-causing it is an MCDHF investigation, not
a bug fix.  `tests/test_sd_space_variational.py` asserts the *required*
variational property for the healthy spaces (1 CSF and singles) and carries
the SD and unscreened-orbital cases as **strict `xfail`s with the measured
numbers**, so the defect is visible in the suite and cannot silently worsen.
This is the one review-adjacent item left open.

---

## Suggested order of work

1. **C1** (Uehling kernel) — one function, two new gates, largest single physics error.
2. **C2** (self-energy table) — decide the single source of truth and delete the other;
   recalibrate `shabaev_local`; add the H 1s and U^91+ Lamb-shift gates.
3. **C4** (silent wrong answers) — reject negative-energy eigenpairs, non-empty-result
   checks, propagate `ier`; these turn unknown-unknowns into exceptions.
4. **C3** (high-Z grid) — expose the inner step and gate the hydrogen-like 1s eigenvalue
   across Z.
5. **C5** (CRM RR channel + DR prefactor) — the plasma layer currently cannot reproduce
   a coronal balance at all.
6. **C6** (Born completion weight) — a one-line cap-semantics fix plus a high-energy gate;
   every collision strength above ~5× threshold is currently wrong by 1–3 orders of
   magnitude.
7. **H0/H1** — the dimensionally wrong rate coefficient, then regenerate the stale
   benchmark and gate it (the advertised limitation is wrong by 8x in the wrong
   direction).
8. **H2/H3** — two small, high-impact property bugs (factor-2 rates, inverted isotope
   shift) with existing tests that pin the wrong value.
9. **P2** — one cache-key change for ~14 % of runtime.
