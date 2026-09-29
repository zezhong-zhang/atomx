# Static Breit retardation and larger correlation spaces

> Historical report. Commands, paths, counts and status describe the recorded
> source snapshot; some campaign tools have since been retired. See the
> [current documentation](../../README.md) for supported usage.

## Implemented

Resolved CI/MCDHF now uses **Gaunt plus frequency-independent retardation**
when Breit is enabled. This includes off-diagonal configuration interference.
Orbitals are optimized for the Dirac–Coulomb energy; the full static Breit
operator is subsequently diagonalized in CI, as in the matched GRASP RCI
control. It is not a self-consistent Breit orbital optimization and does not
include frequency-dependent transverse-photon exchange.

Implementation: `methods/mcdhf/retardation.py`, integrated through
`breit_ci.py`, `exact_ci.py`, `matrix.py`, `ci.py`, and correction provenance.
DHF configuration-average DBSR_HF Breit remains a separate implementation.

The derivation starts from

```
B = -alpha1.alpha2/r12 - (alpha1.grad1)(alpha2.grad2) r12 / 2
```

After integration by parts on both electrons, the second term is a bilinear
in transition-current divergences. For real radial spinors, the radial factor is

```
D_ac = (Q_a P_c - P_a Q_c)' + (kappa_c-kappa_a)(Q_a P_c+P_a Q_c)/r.
```

Its angular factor is the scalar spinor-harmonic overlap. The distance kernel
has Racah-normalized multipoles

```
R_L(r,s) = r_<^(L+2) r_>^(-L-1)/(2L+3)
           - r_<^L r_>^(1-L)/(2L-1).
```

This gives linear-cost radial cumulative contractions. B-spline derivatives
are evaluated analytically, with interpolatory partial-cell quadrature;
raw-grid orbitals use cubic interpolation and interior Gauss quadrature.
No orbital-energy-difference substitution or fitted retardation factor is used.
In particular, the implementation does not assume a local DHF potential.

The static Breit operator and its scope are also documented in the
[DIRAC Hamiltonian reference](https://diracprogram.org/doc/release-12/manual/hamiltonian.html)
and the [RATIP theory paper](https://www.atomic-theory.uni-jena.de/pub/Ratip-cpc-2012b.pdf).

## Additional defect found and fixed

The old Gaunt code stopped the photon rank at `min(j_a+j_c,j_b+j_d)`.
Because alpha has rank one, the allowed upper bound is **one larger**.
An s1/2–p1/2 transition can have photon L=2; the old bound omitted it.
The B-spline Gaunt radial quadrature also used flattened-grid trapezoids.
It now uses the existing higher-order Gauss-cell partial-integral kernel.
Old tests that explicitly encoded the incorrect ceiling and omitted
retardation were updated to the actual operator identities.

## Independent validation

| Configuration | Atomx full static Breit (Ha) | DBSR_HF reference (Ha) |
|---|---:|---:|
| He 1s² | 0.0000637774340205 | 0.000063777432919 |
| Ne 1s²2s²2p⁶ | 0.0166436309421 | 0.016643630840490 |
| Ar closed shell | 0.132364566082 | 0.132364619357042 |

These calculations agree to better than 8e-7 relative. Their remaining
radial/nuclear discretization differences are not interpreted as physical
uncertainties.

**Common-orbital CI test:** import GRASP's He n=2 correlated orbitals, use the
same five J=0 CSFs, and compare RCI with/without the static transverse term.

- GRASP static Breit shift: `3.9742542039977735e-5 Ha`.
- Atomx, grid step .02: `3.974254213368056e-5 Ha`.
- Atomx, grid step .01: `3.9742542096377065e-5 Ha`.

The finer-grid shift differs by about **5.6e-14 Ha**. This is agreement on
shared orbitals and a fixed finite CI space, not agreement with experiment at
that precision. Absolute DC energies differ by ~1e-9 Ha after interpolation.
The first GRASP RCI eigenenergy table is used: its later diagnostic estimated
self-energy table is not part of the requested DC/static-Breit Hamiltonian.

Reproduce this check with `scripts/benchmark_grasp_static_breit.py`; raw
reference input/output and common-orbital results are retained under
`output/correlation_benchmark/retardation_grasp_he/`.

## Correlation expansion

The three-code driver now supports `--breit --max-n N --lmax L` and a GRASP
previous-layer orbital seed. SD destinations include unfilled reference
subshells, ensuring both endpoints can use the same active orbital set.
Both endpoints are recalculated; correlation is not added to only the neutral.
FAC uses `SetBreit(-1, 2)` (all-n static Breit), without configuration-energy
adjustments. QED, recoil and nuclear size are deliberately absent from these
controls. A no-pair/projector and radial convergence audit is still required.

Completed helium controls, ionization energies in eV:

| Space | Atomx DC + static Breit | NIST | Residual |
|---|---:|---:|---:|
| n≤2, l≤1 | 24.4266327 | 24.5873890 | -0.1607563 |
| n≤3, l≤2, independent-start refinement | 24.5401173 | 24.5873890 | -0.0472717 |

The n=3 result exposes an important optimization issue. The original Atomx
start passed its residual criterion at a higher stationary energy. Importing
GRASP's n=3 layer orbitals as a *starting guess*, projecting them into Atomx's
positive-energy B-spline space and reoptimizing all orbitals found a lower
energy, `-2.9019391893361153 Ha` including Breit. No reference energy was used
as a fitting target. The previous stationary result is preserved separately.
The refined residual is 9.87e-6, below the 1e-5 gate; this does not prove the
global minimum. See `scripts/refine_grasp_correlation_seed.py` and its saved
source/radial hashes and orbital coefficients.

GRASP's new-layer optimization completed, but subsequent all-orbital n=3
optimization failed. Its layer-only RCI candidate is shown as **provisional**,
with a failed full-optimization flag. It is not presented as a fully optimized
MCDHF solution. FAC optimizer convergence remains unchecked in the report.

The larger n=4 He (through f) and Mg valence-SD (through d) controls were also
launched, together with an independent n=3→n=4 Atomx orbital continuation.
Their completion/convergence status is recorded by the run artifacts and the
interactive report; no unconverged row is promoted to a production spectrum.

## Website and checks

The NIST site's theory-comparison navigation now links to
`/comparison/correlation-breit/`. It exposes configuration lists, operator
settings, convergence diagnostics, failed executions, and NIST residuals.
The original DC-only benchmark remains available separately.

- 89 tests pass across the Breit algebra, retardation, QED integration,
  DBSR references and comparison-driver tests.
- 9 correction-stack tests pass, including updated retardation provenance.
- The common-orbital GRASP comparison is reproducible from saved inputs.
- The static-Breit report and JSON endpoints return HTTP 200; report JavaScript
  passes `node --check`.

No full-table production regeneration or fitting to NIST was performed.

### First n=4 returns

The first He n=4 run passed its orbital residual gate but stopped at essentially
the same energy as the better n=3 solution. The independent n=3→n=4 continuation
finds a lower energy, so the first n=4 return is a higher stationary solution,
not an established ground-state minimum. It is retained to expose this failure
mode. The continuation has its own explicit residual and convergence status.

The Mg n=4 neutral run hit 1280 iterations with residual 1.52e-4 (>1e-5);
Mg+ converged. Its resulting 7.533273 eV ionization is therefore provisional.
GRASP gives 7.531422 eV for this space and FAC 7.309520 eV with its different
orbital protocol. These values are not a completed experimental-accuracy claim.

The n=4 continuation finished its 400-cycle diagnostic cap at
`-2.902287255481666 Ha` including static Breit, below the default-start
stationary solution, but its residual is **5.65e-5**, above 1e-5.
It is shown as a separate **unconverged continuation** in the website; it is
not substituted for a validated ground-state result. All benchmark processes
started for this pass have now finished. The next numerical work is orbital
stability and continuation convergence, before further production expansion.
