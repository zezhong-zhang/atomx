# Migration Notes

Changes are grouped by the version that introduced them. The package version
is `atomx.__version__` (`src/atomx/__init__.py`); see `CHANGELOG.md` for the
per-version summary.

## Unreleased

### Independent DF orbital controls and explicit-size basis correction

`Atom.solve(orbital_tolerance=...)` and `Sampling(orbital_tolerance=...)`
set a dimensionless L2 orbital-update threshold, independently of the energy
criterion. The explicit solve argument wins. Direct Hamiltonian DF supports
the control on both engines; unsupported methods reject it. Requested
controls and measured updates are recorded. See [solver controls](DF_SOLVER_CONTROLS.md).

Hamiltonian DF with an explicit `Basis(ns=...)` now uses the balanced P/Q
B-spline representation, as the automatic physics grid already did. The old
unbalanced representation could falsely converge to a spurious positive-kappa
root; affected calculations must be regenerated. Basis size remains explicit,
and basis orders are recorded in the calculation specification. B-spline CSF
results now report final optimizer sweeps as `iterations`, rather than the
initializer's outer-iteration count.

### Radiative multipole and FAC conventions

- FAC transition dictionaries now use `A_per_s` and
  `reduced_matrix_element`, replacing the incorrect `gA_per_s` and `gf_alt`
  meanings. No extra upper-state degeneracy division belongs on FAC's printed
  rate. Aggregated `MULTIP=0` rows expose `aggregate_gf` and no reduced amplitude.
- `write_tra` explicit rows require the corrected rate and amplitude fields.
  Atom-generated rows preserve A directly and export a positive amplitude
  magnitude; coherent phases cannot be recovered from a line strength.
- Higher-multipole oscillator strengths include their photon-frequency factors.
  Magnetic angular normalization is corrected, and bound transitions and
  B-spline recombination pass photon energy to `radial_mL(..., omega=...)`.
  Calling that low-level kernel without omega explicitly retains its static
  limit. `TransitionResult.photon_approximation` records the operator choice.
- Historical FAC rate tables using the old gA interpretation require
  regeneration. See [current verification evidence](VERIFICATION.md).

### Optional shooting refinement

`Sampling(radial_refinement=RadialRefinement(...))` opts into repeated grid
and automatic-box checks for single-configuration shooting. Import
`RadialRefinement` from `atomx` and pass the sampling through `Basis`.
The point count becomes the starting count; explicit radial boxes stay fixed.
`result.radial_convergence` records outcomes and limits. Ordinary fixed-grid
calls keep their existing controls. See [solver controls](DF_SOLVER_CONTROLS.md).

### Direct Slater reference integration

`numerics.bspline.slater.slater_rk_direct` now takes a two-dimensional
cell Gauss grid and weights, matching `bsp_data["gr"]` and `bsp_data["grw"]`.
Flattened grids cannot identify cell boundaries and are rejected. Density
arrays may still be flat. The public `slater_rk(..., method="direct")`
signature is unchanged. Its independent reference integration now splits
coincident cells along the Coulomb cusp; the default Poisson method is
unchanged.

### Removed forwarding modules and misleading aliases

Imports now name the implementation directly. These old modules and names
have been removed; there is no compatibility forwarding layer.

| Removed path/name under `atomx` | Use instead |
|---|---|
| `methods.hartree_fock.HartreeFock`, public `HartreeFock` | `HartreeFockSlater` (local Slater exchange, not nonlocal HF) |
| `methods.relativistic_hf.BreitPauli` | `methods.breit_pauli.BreitPauli` or public `BreitPauli` |
| `scattering.protocols` | `state_protocols` |
| `corrections.workspace` | `numerics.radial_workspace` |
| `numerics.shooting.scf_potential` | `potentials.static_field` |
| `properties.hyperfine.qed_corrections` and its constant exports | `data.nuclear_structure.TABLE[isotope]` |
| Yk imports through `numerics.bspline.evaluate` | `numerics.yk.compute_yk`, `yk_flat`, `yk_partial_matrices` |

For example, the proton Zemach radius is
`TABLE["h1s"].zemach_radius_fm`; its polarizability contribution is
`TABLE["h1s"].polarisability_fraction`. This is tabulated nuclear data.
The removed `qed_corrections` module contained no hyperfine QED operator
implementation. The existing hyperfine correction ladder retains its
documented empirical coefficients and limitations.

The `methods.mcdhf.solver` entry point remains: its distinct method/result
types select explicit multiconfiguration calculations. Package `__init__`
exports remain the public API.

### Default DHF Hamiltonian averaging

Single-configuration `Atom(..., exchange="hf")` now selects
`Correlation(averaging="hamiltonian")`: ground-state intermediate coupling
within the CSFs of the specified nonrelativistic configuration. Coefficients
come from Hamiltonian diagonalization; occupations are `sum_j |c_j|² n_ij`
(Guerra et al., *Relativistic calculations of screening parameters and atomic
radii of neutral atoms*, 2016 revision, Eqs. 5 and 10). Orbital optimization
uses the full `c†Hc` derivative, including off-diagonal CSF interference.
This does not generate excited configurations or add dynamical correlation.

`statistical` remains the distinct fixed degeneracy ensemble; `equal` gives
each integer jj distribution equal weight. `fractional` uses products of
mean occupations. `generate_jj_average` accepts only the fixed `statistical`
and `equal` choices: it cannot compute Hamiltonian eigenvector weights.

Single-CSF configurations retain the ordinary DHF solver. Other Hamiltonian
DF cases use direct, self-consistent radial source equations on both engines.
They share Hamiltonian contractions and result assembly with MCDHF, but do not
enter its finite-continuum orbital optimizer. The numerical engines follow the
same self-consistent electron branch. Explicit MCDHF retains its fixed no-pair
variational space. The orbital functional is Dirac–Coulomb;
Breit enters the subsequent state Hamiltonian, and QED/recoil retain their
recorded correction stages. No claim of self-consistent Breit/QED orbital
optimization is made. Direct DF requires radial grid/box refinement independently
of SCF convergence. `Basis(shooting_space=...)` now requires explicit MCDHF;
use `Basis(sampling=Sampling(n_grid=...))` to refine direct shooting DF.
See [DF solver controls](DF_SOLVER_CONTROLS.md) for routing and timing evidence.

An old single-configuration `Correlation(averaging="ci")` request now raises
with migration instructions. Use `averaging="hamiltonian"` for DHF. To request
MCDHF explicitly, pass a configuration list (even a one-item list), a correlation
expansion, or an explicit `MCDHF` method. No silent theory switch is retained.

The density CLI follows this default; `--correlation s/sd` selects MCDHF.
The demo begins with a source-path cell to avoid importing an editable
installation of another checkout. Restart a kernel that already loaded it.


### Shared CSF implementation

The general state machinery is no longer owned by MCDHF. Internal imports
must use the canonical modules; the removed modules have no forwarding shims.

| Responsibility | Canonical location |
|---|---|
| CSF definitions/enumeration (formerly `methods.mcdhf.config`) | `config.csf` |
| Exact angular Hamiltonian tables (formerly `methods.mcdhf.exact_ci`) | `angular.csf_hamiltonian` |
| Hamiltonian matrix elements, Breit/retardation, densities, one-body contractions | `physics.csf` |
| Shared state functional and variational driver | `methods.csf` |
| Monoconfiguration intermediate-coupling DF entry point | `methods.hamiltonian_df` |
| Expanded configuration-space entry point | `methods.mcdhf` |
| B-spline constrained orbital optimizer | `numerics.bspline.orbital_optimization` |
| Shooting constrained orbital optimizer | `numerics.shooting.orbital_optimization` |

State-consuming properties now use `Atom._csf_result` for both DF and MCDHF;
public `Atom.result`, `state_energies`, `waves` and `density` remain the accessors
for users. The DF result records that no configuration expansion was generated.

### Shooting engines and result ownership

Update downstream callers using this table. Repository source, scripts,
benchmarks and tests now use the canonical paths. No forwarding modules remain
at the removed locations. Physical contractions previously accessed through HF
solver private methods now live under `solver.functional`, for example
`solver.functional.two_electron_energy(...)`.

| Previous import (under `atomx`) | Canonical import |
|---|---|
| `methods.shooting_local.ShootingSolver`, `numerics.shooting.solver.ShootingSolver`, or the shooting package export | `atomx.numerics.shooting.local.ShootingSolver` |
| `methods.shooting_hf.ShootingHFSolver` or `numerics.shooting.hf_shooting.ShootingHFSolver` | `atomx.numerics.shooting.hartree_fock.ShootingHFSolver` |
| `ShootingResult` through a shooting solver or package | `atomx.results.ShootingResult` |
| `dfs_frozen_orbital` through the old shooting solver | `atomx.numerics.shooting.local.dfs_frozen_orbital` |
| Slater integrals and `_yk_over_r` through the shooting solver | `atomx.numerics.shooting.coulomb` |
| Inhomogeneous orbital helpers through `hf_shooting` | `atomx.numerics.shooting.dirac_orbital` |
| Homogeneous pair and Green solve through `hf_shooting` | `atomx.numerics.shooting.dirac_green` |
| `methods.mcdhf.grid_shooting` electron-space builders | `atomx.numerics.shooting.electron_space` |
| `methods.mcdhf.grid_shooting.optimize_grid_shooting` | `atomx.numerics.shooting.orbital_optimization.optimize_grid_shooting` |
| `methods.mcdhf.grid_shooting.occupied_1s2_reference_potential` | `atomx.methods.csf.grid_functional.occupied_1s2_reference_potential` |
| `methods.mcdhf.shooting_preconditioner` | `atomx.numerics.shooting.preconditioner` |
| `methods.mcdhf.shooting_solver.ShootingSpace` | `atomx.numerics.shooting.electron_space.ShootingSpace` (public `atomx.ShootingSpace` is unchanged) |
| `methods.mcdhf.shooting_solver.solve_variational_shooting` | `atomx.methods.csf.grid_solver.solve_variational_shooting` |
| `numerics.analytic[.hydrogenic_dirac]` hydrogenic helpers | `atomx.methods.hydrogenic_reference` |
| `numerics.bspline.continuum.photoionization_cross_section` | `atomx.scattering.photoionization.photoionization_cross_section` |

The old `numerics/analytic` namespace contained numerical B-spline reference
calculations, not closed-form wavefunctions. Its forwarding files are removed.
Historical design and audit documents may retain the old paths as evidence of
the tree they reviewed.

#### Saved results

`ShootingResult` retains its fields, native arrays, metric and physical metadata.
Schema-1 shooting HDF5 files use a format identifier rather than a Python module
path, so they require no rewrite. Python pickle import compatibility is not
provided by this change; the versioned archive is the supported persistence path.

Local Slater results explicitly have no jj configuration ensemble. The archive
writer and validator now accept that `None` value and preserve it as JSON null;
previously they attempted to iterate it. Nonlocal HF ensembles continue to be
validated and preserved for open-shell correction replay.


Numerical modules now reject upward imports without compatibility exceptions.
See [migration verification](archive/2026-09/NUMERICAL_BOUNDARY_MIGRATION_2026-09-29.md).


### The `Atom` constructor is ten arguments (breaking — every call site changes)

`Atom.__init__` used to take 28 flat keyword arguments mixing six unrelated
physical concerns, plus a short-lived "grouped dataclass" layer on top
(`System`, `ElectronicStructure`, `Basis`, `Interaction`, `CIProtocol`,
`Environment`).  Both are gone.  There is now **one** spelling per physical
question, and every retired keyword raises a `TypeError` naming its
replacement.

```python
# before
ne = Atom(Z=10, config="1s2 2s2 2p6", numerics="bspline", ns=200,
          grid_hi=0.02, grid_he=0.10, breit=False, qed=False,
          averaging="statistical", j_resolved_csfs=False,
          plasma_ne=1e22, plasma_Te=100.0)

# after
from atomx import Atom, Basis, Corrections, Correlation, Environment

ne = Atom(
    10,                                   # Z (positional: the atom's identity)
    config="1s2 2s2 2p6",                 # or a list for multi-configuration
    charge=0,
    nuclear="fermi",
    exchange="hf",                        # or Exchange(model=..., alpha_x=...)
    corrections="none",                   # or Corrections(...)
    correlation=Correlation(averaging="statistical"),   # or "s" / "sd"
    engine="bspline",                     # was numerics=
    basis=Basis(ns=200, grid_hi=0.02, grid_he=0.10),
    environment=Environment(plasma_ne=1e22, plasma_Te=100.0),
)
ne.solve()
```

`Z` and `config` are the only positional slots; everything after them is
keyword-only.  The old signature accepted 28 positional-or-keyword
parameters, so a stale positional call would have silently bound `charge` as
`ns`; now it raises.

#### Every retired name and its replacement

| retired | replacement | notes |
|---------|-------------|-------|
| `configs=[...]` | `config=[...]` | a *list* still selects the MCDHF/CI path; a plain string is one configuration |
| `numerics=` | `engine=` | the argument selects an engine, not "numerics" |
| `ns=` | `basis=Basis(ns=...)` | B-spline count |
| `ks=` | `basis=Basis(ks=...)` | B-spline order |
| `r_max=` | `basis=Basis(r_max=...)` | radial box (Bohr) |
| `grid_hi=` | `basis=Basis(grid_hi=...)` | DBSR_HF knot-shape override |
| `grid_he=` | `basis=Basis(grid_he=...)` | DBSR_HF knot-shape override |
| `grid_hmax=` | `basis=Basis(grid_hmax=...)` | DBSR_HF knot-shape override |
| `starting_potential=` | `basis=Basis(starting_potential=...)` | SCF initial guess |
| `sampling=` | `basis=Basis(sampling=...)` | accuracy tier; `solve(sampling=...)` still exists |
| `alpha_x=` | `exchange=Exchange(model="slater", alpha_x=...)` | only the Slater model has a coefficient |
| `slater_method=` | `exchange=Exchange(model=..., slater_method=...)` | only the local models, and only on the shooting engine |
| `breit=` | `corrections=` | `Corrections(breit=False)` or a preset |
| `qed=` | `corrections=` | `Corrections(qed=False)` or a preset |
| `qed_backend=` | `corrections=Corrections(qed_backend=...)` | now also reaches multi-configuration runs |
| `correction_stack=` | `corrections=Corrections(stack=...)` | the stack stays authoritative |
| `averaging=` | `correlation=Correlation(averaging=...)` | Guerra CSF weights |
| `radial_optimization=` | `correlation=Correlation(radial_optimization=...)` | MCDHF outer loop |
| `j_resolved_csfs=` | `correlation=Correlation(j_resolved_csfs=...)` | CSF enumeration |
| `per_block_orbital_sets=` | `correlation=Correlation(per_block_orbital_sets=...)` | DODS/EOL scaffold |
| `plasma_ne=` | `environment=Environment(plasma_ne=..., plasma_Te=...)` | both are required together, and are now applied |
| `plasma_Te=` | `environment=Environment(plasma_ne=..., plasma_Te=...)` | as above |
| `system=System(...)` | `charge=`, `nuclear=` | the interim group is gone |
| `electronic=ElectronicStructure(...)` | `config=`, `correlation=` | |
| `interaction=Interaction(...)` | `exchange=`, `corrections=` | |
| `ci=CIProtocol(...)` | `correlation=Correlation(...)` | |
| `basis=Basis(numerics=..., grid_hi=...)` | `engine=`, `basis=Basis(grid_hi=...)` | `Basis` no longer carries the engine |
| `environment=Environment(plasma_ne=...)` | unchanged spelling | but the fields now do something |

`Z`, `config`, `charge`, `nuclear`, `exchange`, `correlation` and
`environment` keep their names; `environment` and `basis` keep the keyword
but their objects changed (see the last two rows).

#### Worked examples, one per concern

**1. System — `charge`, `nuclear`**

```python
# before
he_plus = Atom(Z=2, charge=1, nuclear="point")
# after
he_plus = Atom(2, charge=1, nuclear="point")
```

**2. Electronic structure — `config` / `configs` / `correlation`**

```python
# before
be = Atom(Z=4, configs=["1s2 2s2", "1s2 2p2"])
be = Atom(Z=4, correlation="sd", averaging="statistical")
# after
be = Atom(4, config=["1s2 2s2", "1s2 2p2"])
be = Atom(4, correlation=Correlation(expansion="sd",
                                     averaging="statistical"))
# `correlation="sd"` alone is still accepted (the string form):
be = Atom(4, correlation="sd")
```

**3. Radial basis and engine — `numerics`, `ns`, `ks`, `r_max`, `grid_*`,
`starting_potential`, `sampling`**

```python
# before
ne = Atom(Z=10, numerics="bspline", ns=200, ks=9, r_max=90.0,
          grid_hi=0.02, grid_he=0.10, starting_potential="tfda",
          sampling=Sampling.high)
# after
from atomx import Atom, Basis, Sampling

ne = Atom(10, engine="bspline",
          basis=Basis(ns=200, ks=9, r_max=90.0,
                      grid_hi=0.02, grid_he=0.10,
                      starting_potential="tfda",
                      sampling=Sampling.high))
```

**4. Interaction model — `exchange`, `alpha_x`, `slater_method`, `breit`,
`qed`, `qed_backend`, `correction_stack`**

```python
# before
fe = Atom(Z=26, charge=16, config="1s2 2s2 2p6", numerics="shooting",
          exchange="slater", alpha_x=2 / 3, slater_method="slag6",
          breit=False, qed=True, qed_backend="mohr")
# after
from atomx import Atom, Corrections, Exchange

fe = Atom(26, charge=16, config="1s2 2s2 2p6", engine="shooting",
          exchange=Exchange(model="slater", alpha_x=2 / 3,
                            slater_method="slag6"),
          corrections=Corrections(breit=False, qed=True,
                                  qed_backend="mohr"))
```

**5. Correlation / CI protocol — `averaging`, `radial_optimization`,
`j_resolved_csfs`, `per_block_orbital_sets`**

```python
# before
n = Atom(Z=7, config="1s2 2s2 2p3", averaging="ci",
         radial_optimization="fac_frozen", j_resolved_csfs=True)
# after
from atomx import Atom, Correlation

n = Atom(7, config=["1s2 2s2 2p3"],
         correlation=Correlation(averaging="ci",
                                 radial_optimization="fac_frozen",
                                 j_resolved_csfs=True))
```

**6. Environment — `plasma_ne`, `plasma_Te`**

```python
# before: accepted and then ignored — the run was an ordinary vacuum atom
ne = Atom(Z=10, charge=2, plasma_ne=1e22, plasma_Te=100.0)
# after: the Debye-screened potential is added to the nucleus inside the SCF
from atomx import Atom, Environment

ne = Atom(10, charge=2,
          environment=Environment(plasma_ne=1e22, plasma_Te=100.0,
                                  screening="debye"))
```

#### Behaviour changes that come with the migration

Four of the retired keywords used to be accepted and *dropped*.  They now do
what they say (measured sizes in `CHANGELOG.md`):

* `plasma_ne`/`plasma_Te` (now `environment=`) screen the nuclear potential
  inside the SCF.  A run that passed them and relied on the numbers being
  the vacuum ones will move — by design; that is the point of the argument.
* `Sampling.r_max` and `Sampling.n_grid` (now `Basis(sampling=...)`) reach
  the solvers: a tier with a non-default `r_max` now changes the box, and a
  non-default `n_grid` changes the shooting grid.
* `qed_backend` (now `Corrections(qed_backend=...)`) reaches
  multi-configuration runs instead of being replaced by the default `auto`.
* `slater_method` on the B-spline engine now **raises**: the Galerkin path
  evaluates its Slater integrals with its own quadrature and never read it.

Two more names changed meaning:

* `atomx.Exchange` / `atomx.Correlation` are now the constructor value
  objects from `atomx.atom_inputs`.  The legacy enums used by the *method*
  API (`DiracFock(exchange=Exchange.SLATER)`, `MCDHF(correlation=...)`) are
  still available as `atomx.types.Exchange` / `atomx.types.Correlation` and
  are still accepted as values by `Atom(exchange=...)`.
* `Atom.provenance` now reports `{"engine": ..., "exchange": ...,
  "averaging": ...}` (the key was `"numerics"`), plus a `"plasma"` entry
  when the atom sits in one.

### Grouped `Atom` constructor inputs, interim design (withdrawn)

The six grouped frozen dataclasses (`System`, `ElectronicStructure`,
`Basis`, `Interaction`, `CIProtocol`, `Environment`) and the
`system=`/`electronic=`/`basis=`/`interaction=`/`ci=`/`environment=`
keywords were an additive layer on top of the 28 flat keywords, and
`tests/test_atom_input_surface.py` pinned the pair.  They were superseded by
the single canonical surface above: `Basis` and `Environment` survive with
new fields, the other four are gone.

### Package version is single-sourced

`pyproject.toml` no longer carries a literal version — it declares
`dynamic = ["version"]` and hatchling reads `src/atomx/__init__.py`. The two
values had drifted (`0.1.0` in the metadata that `pip`/`uv` reported,
`0.2.0` in `atomx.__version__`); the code value wins. Nothing to change in
downstream code, but `importlib.metadata.version("atomx")` now agrees with
`atomx.__version__`.

### Heavy visualization/notebook packages moved to extras

`pip install atomx` no longer pulls `manim`, `plotly`, `nbformat`, `ipython`
or `ipykernel`. Install the back-end you use:

```bash
pip install "atomx[viz]"          # plotly
pip install "atomx[viz-manim]"    # manim (also needs system pangocairo)
pip install "atomx[notebook]"     # nbformat/ipython/ipykernel
pip install "atomx[all]"
```

`matplotlib` stays a base dependency because `import atomx` imports
`atomx.viz.mpl` unconditionally. Code that relied on a transitive plotly/manim
import should import it explicitly.

### Test markers renamed

`pytest -m fac_parity` no longer selects anything. The marker conflated two
different things and was renamed accordingly:

| old | new | meaning |
|-----|-----|---------|
| `fac_parity` (stored external data) | `external_fac` | asserts atomx against stored external FAC reference data |
| `fac_parity` (self-regression) | `snapshot` | asserts atomx against its own recorded output |

`pytest` also runs with `--strict-markers` now, so a stale selector fails
loudly instead of silently collecting nothing.

### Benchmark scripts take their reference code from the environment

`benchmarks/fac_runner.py`, `bench_local_exchange.py`,
`bench_shooting_vs_radial.py`, `scripts/energies_gs.py`,
`scripts/density_gs.py` and the FAC calibration helper no longer contain
machine-specific absolute paths. Set `FAC_SFAC_PATH` (or `FAC_BIN` /
`FAC_BIN_DIR`), `FAC_DYLD_LIB`, `RADIAL_DHFS_DIR`, `ATOMX_MCDHF_REF_CSV`,
`ATOMX_WF_TXT_DIR` as needed; each script stops with an actionable message
when its reference is missing rather than silently skipping.

## 0.2.0 — QED backend default

`Atom(..., qed=True)` now defaults to `qed_backend="auto"`.

`auto` keeps the historic Mohr/MSS self-energy backend for `Z <= 36` and
uses the FAC/Indelicato-Mohr K/L-shell backend for `Z >= 37`.  This closes
the documented high-Z residual against FAC.

Users who need bit-stability against the old MSS-only output should pass:

```python
Atom(82, config="1s1", corrections=Corrections(qed_backend="mss"))
```

## Finite-frequency electric radiation

One-photon transition, photoionization, RR and orbital DR radiative-width
calculations now use the finite-frequency electric operator. Both electric
gauges retain the spherical Bessel dependence. Static response calls remain
static. See [operator conventions](TRANSITION_OPERATORS.md).

For direct radial calls, `radial_eL` accepts `kappa_i`, `kappa_f` and signed
`omega=E_f-E_i`; omitting omega selects its static limit. The velocity kernel
now requires both kappas and signed nonzero omega and raises when missing;
the former arbitrary fallback is removed. Its sign convention now agrees
with length gauge. For momentum/static diagnostics explicitly request
`long_wavelength=True`.

`ci_line_strengths` defaults to `photon_approximation="finite_wavelength"`.
Its `CILineSet.trace` is `None` because each line has a different operator.
Request `photon_approximation="long_wavelength"` for static lines and the
mixing-invariant determinant-trace identity.

## Direct-momentum specific mass shift

`compute_sms_abinitio(orbitals, workspace, *, Z=None)` no longer accepts
orbital energies. It evaluates direct Vinti integrals and uses occupation
fractions in the reduced-angular sum. Supply `Z` for the point-Coulomb Breit
SMS2/SMS3 terms; omit it for the leading momentum operator. Open shells now
raise instead of being silently treated as closed shells. The old neon SMS
value near -117.84 is invalid; see [recoil conventions](RECOIL_OPERATORS.md).
The unequal-j relativistic SMS2 phase is also corrected in the common
recoil implementation, affecting the SMS correction stack as well as properties.

## Finite-nucleus basis resolution and orbital accuracy

Fixed-size Fermi B-spline bases now use the shared nuclear surface scale for
the first knot, replacing `0.05/Z`, which failed to resolve light nuclei.
Increasing `ns` alone previously left this error unchanged. Energies and
properties from those under-resolved fixed bases must be regenerated.

`ShootingHFSolver.solve(orbital_tolerance=...)` can independently converge
orbital shapes for first-order properties while retaining the requested
energy stopping criterion. The default remains unchanged; this native
control is not yet an `Atom.solve`/`Sampling` option. See
[accuracy and timing evidence](DF_SOLVER_CONTROLS.md#independent-orbital-accuracy-for-properties).

## Radial collision quadrature and stable Coulomb kernels

`dw_eii_cross_section`, `eii_cross_section` (DW portion) and
`dw_excitation_cross_section` now accept `radial_refinement=1`. A positive
integer subdivides every observable integration interval, retaining the
same endpoints and original nodes. This control is separate from the
continuum propagation grid (`ContinuumModel.shooting(n_grid=...)` or
`shooting_n_grid`) and outgoing-energy quadrature. EII provenance records
the refinement, mesh size and trapezoidal rule. Refinement does not certify
convergence automatically.

The general numerical entry point is
`make_fine_grid(..., refinement=2)`. Resource caps raise instead of silently
coarsening the requested mesh. `RadialMetric.trapezoid` owns the nonuniform
weights, including the two half-weight endpoints; the CE, EII and determinant
Auger callers no longer use `np.gradient(r)` as integration weights.

The arbitrary-grid Coulomb kernels use bounded powers of radii relative to
local block anchors. They rescale cumulative moments exactly at block
boundaries and retain the same trapezoidal Green function. This avoids the
overflow of separate absolute `r**k` and `r**(-k-1)` factors. Raw `pow_tables`
and its private caches have been removed; use `yk_on_grid`, `grid_slater_rk`
or `yk_pair_dot_batch_on_grid`. Only immutable meshes are cached.

Scalar Slater integrals skip only exactly zero outer-density support,
retaining the last nonzero sample's half interval. EII integrates both spinor
components on the full mesh. The former pointwise relative-tail cutoff had
no general bound on signed matrix-element errors and is no longer applied.

## Shared Coulomb angular coefficients

The coupled two-electron Coulomb coefficients now live in
`atomx.angular.coefficients.coupled_coulomb_direct` and
`coupled_coulomb_exchange`. They use an exact Racah reduction of the former
magnetic-sublevel sums, with the same `reduced_C_jj` (Grant) phase convention. Their
arguments are `(kappa_c, kappa_d, kappa_a, kappa_b, twice_J, rank)`; the
antisymmetrized amplitude remains `D*R_direct - E*R_exchange`.

The private DW helpers `_cc_direct`, `_cc_exchange`, `_ck_m` and `_cg2` have
been removed. Import `ckm` from `atomx.angular.coefficients` and
`clebsch_gordan` from `atomx.angular.clebsch` (the canonical CG kernel;
the former `coefficients.cg` wrapper is gone).
CE, EII, Auger and CI photon callers have been migrated; no forwarding aliases
remain. The radial integrals and physical channel truncations are unchanged.

The shared Wigner implementation also now uses its precomputed log-factorial
table. Long or strongly cancelling Racah sums are evaluated with exact Python
integers before the final floating-point square root. Compiled callers reach
this bounded-cache fallback through Numba object mode; the short, well
conditioned path stays compiled. This fixes cancellation errors at high
angular momentum without changing angular conventions.

## Explicit finite-frequency transverse exchange

`ci_solve(..., include_breit=True, breit_orbital_energies=...)` now replaces
static Breit by the real frequency-dependent transverse operator. Energies
are an explicit `(n, kappa)` mapping in Hartree, fixed during diagonalization.
Without the mapping, the existing static operator is retained.
Public calculations can now request
`Corrections(breit_frequency="orbital_differences")`. Ordinary DHF uses
converged orbital energies; Hamiltonian DF/MCDHF additionally require an
explicit `breit_orbital_energies` map for every active orbital. The same
fields are available on `CorrectionStack`, `DiracFock` and `MCDHF`.
Static Breit remains the default. Finite-frequency exchange replaces the
static contribution, and is evaluated on fixed orbitals. See
[operator, numerical method and FAC evidence](TRANSVERSE_INTERACTION.md).

### Native frozen-core integrals

Bound CI levels and photons retain their original radial representation.
Closed-core contraction is `physics.csf.frozen_core.closed_core_one_body`;
the scattering-specific `build_core_folded_one_body` and
`build_bspline_core_folded_one_body` implementations were removed.
`bspline_bound_integrals` adapts a native result and its virtual coefficients
for the shared Coulomb callback. Low-level `compute_exchange_matrix_cell`
returns all four blocks `(PP, QQ, PQ, QP)`; direct and exchange matrices use
the padded `bsp_data['ns']` layout.

Shooting checkpoints optionally include `_rv_nuc`, the sampled nuclear plus
environment field actually used in SCF. Existing archives still load. A bound
CI reconstruction needing this field rejects archives that lack it and asks
for a new solve, rather than assuming a point nucleus.

### DR photons and continuum results

`ResonanceManifold.dr_levels` now uses resolved final-state CI energies and
finite-frequency E1 stabilization. `ManifoldLevel.radiative_channels` records
the final level, photon energy and rate. Low-level manifolds must supply the
bare-plus-frozen-core `one_body` Hamiltonian for photon calculations; dressed
orbital energies are insufficient. Use `manifold_from_configs` to retain the
native bound representation. `photon_approximation="long_wavelength"` requests
the static diagnostic explicitly.

Regenerate continuum-dependent results produced with the incorrect VINT tail
interval or shared Auger radial cache. The continuum solver now propagates in
the actual Coulomb tail, and Auger integrals are scoped to one continuum
energy/potential. These corrections can substantially change widths; former
slow execution was also a symptom of the wrong extrapolated potential. See
[verification evidence](VERIFICATION.md#dr-final-state-ci-and-continuum-repair-2026-09-30).

`dr_calculate` now shares the native manifold constructor. Local-potential
shooting results can be used directly; B-spline results retain their native
Fock basis. Nonlocal shooting HF virtuals are not yet supported and raise
explicitly, without substituting a local model. Continuum sampling expands
to cover a bound on the CI resonance energies. Resonance labels now contain
the hole/parity group, J and level index, avoiding collisions in thermal
rate breakdowns. The `channels['ground']` entry retains the ground Auger rate.
The computed default complex remains dr33, with the limitations documented
on `dr_calculate`; larger capture spaces are not implied by this migration.


### Exact Coulomb matching for continuum shooting

`solve_dirac_continuum_shooting` now obtains normalization and phase from the
regular/irregular Dirac–Coulomb spinors. The returned phase is relative to
Salvat DCOULF's relativistic Coulomb convention, modulo an overall real wave
sign; consumers using the former nonrelativistic phase convention must update.
`match_phase=False` still returns NaN for phase and uses the same accurate wave
normalization. Two tail points check the matching coefficients when available.
Callers must still choose a box that reaches the physical Coulomb tail; the
check does not certify that truncating a non-Coulomb field is valid.

Regenerate continuum-dependent rates where the former WKB normalization bias
matters. The new `mpmath>=1.3` runtime dependency supplies a private precision
context only when the fast Coulomb asymptotic expansion cannot meet tolerance.
The new numerical module is `numerics/shooting/dirac_coulomb.py`; the existing
`coulomb.py` still supplies Slater integrals. See the
[external normalization evidence](VERIFICATION.md#diraccoulomb-continuum-normalization-2026-09-30).


### Continuum sampling on an observable grid

Both scalar and batch continuum solvers accept `interp_grid`. Requested
samples now use the Dirac integrator's converged power series and adaptive
propagation where subdivision is needed, including the regular origin
solution. Scattering's `_continuum_on_grid` delegates this work to numerics.
Linear interpolation of the wave is removed from these paths.

The batch API's returned radius array now corresponds to the sampled waves
when `interp_grid` is supplied. Output radii must be finite, nondecreasing and
inside the solved box; endpoint clamping is no longer accepted. A failed
batch wave raises a contextual `RuntimeError` with the original exception as
its cause, instead of returning an unexplained `None` value. Global box and
point-count defaults remain unchanged pending broader adaptive validation.
See [screened continuum evidence](VERIFICATION.md#screened-continuum-and-dense-radial-sampling-2026-09-30).

### Shared mean Hartree field for collisions

The collision-specific `_distorting_potential_fn` has been removed. DW
excitation, ionization and the affected Auger paths now use
`potentials.static_field.static_hartree_potential(result, screening="mean")`.
This scales the complete target density by `(N-1)/N`, retains the solved
nuclear field, and interpolates electronic screening separately. It avoids
flattening the nuclear potential below the first B-spline quadrature point.
The one-electron point-nucleus limit is exactly `-Z/r`, without density
integration. Finite-nucleus results require their stored nuclear field.

`PotentialModel.static_hartree(screening="mean")` exposes the same choice.
Mean screening cannot be combined with a named subshell hole. EII now honors
`ContinuumModel.potential`; previously that input could be ignored while
provenance incorrectly reported local exchange. The result reports the
actual screening and residual charge used for Coulomb matching. Explicit
bare fields cannot request local exchange. This correction can change
results that previously supplied an ignored potential model.

DW excitation and ionization also evaluate bound B-spline expansions
directly on their integration meshes. `eval_P` and `eval_Q` accept optional
`radii`; omitting it retains the cached native Gauss-grid evaluation. Explicit
radii must lie inside the spline box. Excitation handles different initial
and final boxes by zero extension outside each box. This removes linear
interpolation and the artificial origin clamp in ionization.

### Full energy-sharing integration and explicit EII tail choice

DW ionization now integrates the full half-sharing interval `[0, (E-I)/2]`.
The reusable numerical rule `sinh_squared_quadrature` applies Gauss-Legendre
integration under `E1 = I*sinh(x)^2`; its Jacobian is included in the weights.
It resolves the binding-energy scale and square-root endpoint behavior
without calling the continuum solver at zero energy. The previous fixed
endpoint cutoffs and fixed gap between its two sampling regions are removed.
The default `n_outgoing_energies` is 16. This is a quadrature order, not a
convergence guarantee; refine it for the target and incident energy.

`eii_cross_section` and `dw_eii_cross_section` now accept
`partial_wave_tail="none"` (default) or `"geometric"`. The default returns
only the computed partial-wave sum. The former EII-specific tail switched
discontinuously to zero when the last terms began rising, which introduced
large false energy-quadrature sensitivity. The optional estimate now uses
the same continuous capped estimator as excitation. It remains approximate;
the result reports the computed sum, applied tail, estimated tail and the
number of nodes outside the estimator's decay regime. Increasing
`kappa_max` remains necessary to establish physical partial-wave convergence.
Despite its historical name, that control bounds orbital `l` and includes
both Dirac partners, through `kappa=-(kappa_max+1)`.

The shared estimator is `numerics.extrapolation.geometric_tail_estimate`.
The two private tail helpers in `scattering.distorted_wave.solver` were
removed, with all internal callers migrated. No import aliases remain.
Continuum cache keys now retain full floating-point energies rather than
rounding to ten decimal places, and repeated incident energies reuse solved
waves. A near-threshold case with a 1.27e-12 Ha outgoing node now remains
positive instead of rounding to zero.

The stored helium-like benchmark raises the computed partial-wave cutoff
from 10 to 16 following removal of its automatic tail. Only native values
and derived ratios were refreshed; external FAC values and agreement gates
remain unchanged. New FAC runs must state their multipole cutoff explicitly:
FAC's `IONMAXK=6` means rank 3 in its doubled-index convention.

## Ionization continuum domain independent of the bound box

`ContinuumModel.shooting(r_max=...)` in distorted-wave ionization may now
exceed the native bound B-spline box. This radius controls both continuum
propagation and the collision integrals. The fixed bound expansion is zero
outside its own box; no extrapolation or new bound-state solve is performed.
Continuum-continuum Coulomb sources span the full collision domain even when
the consuming bound-continuum density has compact support. Only exactly zero
consumer tails are skipped, with both spinor components included.

Results record the bound and continuum radii separately and state that radial
domain convergence requires an independent check. The collision operator is
explicitly recorded as `instantaneous_coulomb`, with `transverse_exchange=False`:
atomic Breit/QED settings do not add a transverse collision amplitude.
Default radii and physical models are unchanged.

## High-angular-momentum continuum dense output

Dense Dirac sampling now evaluates middle-interval power series using a
locally normalized spinor, then restores its amplitude. When the left
endpoint cannot represent the requested relative accuracy and the right
endpoint is larger, it uses the right endpoint of the same potential interval.
This prevents a tiny normalized wave from exhausting the relative residual
test. Exactly zero middle-interval initial data return the homogeneous zero
solution; the regular Frobenius origin boundary remains distinct. Adaptive
integration raises if its step can no longer advance at machine precision.

This fixes the high-partial-wave EII hang reproduced at kappa=-65. It does
not discard weak partial waves or impose an amplitude cutoff. The shared
coupled Coulomb coefficient caches are bounded to 131072 entries each;
eviction changes memoization only, not the calculated coefficients.

## Bounded continuum storage in ionization

EII now evaluates one outgoing energy pair at a time and retains only that
pair and the incident continuum waves. Stored wave arrays no longer grow
with energy-sharing quadrature order or the incident energy grid. Repeated
incident energies reuse the completed shell result, including diagnostics.
Radial solves, exact floating-point energy keys, quadrature and contraction
order are unchanged. Missing waves still raise unless diagnostic mode is
explicitly requested. The unused scalar-wave fallback was removed.

The measured He+ panel reduces peak process memory by 52%, with a roughly
4% runtime cost. This enables larger numerical convergence studies; it is
not a claimed throughput improvement. See the EII streaming section of
`VERIFICATION.md` for the matched comparison and current FAC reference limits.

## Reusable continuum numerical batches

`numerics.shooting.dirac_continuum.DiracContinuumBatch` provides a context
for successive wave batches with common numerical controls:

```python
with DiracContinuumBatch(potential, r_max=50., n_grid=1000,
                        Z_residual=17., interp_grid=radii) as solver:
    grid, waves = solver.solve_many([(-1, 10.), (1, 10.), (-2, 10.)])
    grid, waves = solver.solve_many([(-1, 20.), (1, 20.), (-2, 20.)])
```

The context reuses worker threads and checks the potential each time. It
does not retain returned waves. Closing releases workers, including after
exceptions. The standalone `solve_dirac_continuum_many` entry remains
available for individual batches. EII now uses one context per incident
energy and reuses its outgoing P/Q buffers between sharing nodes.

## Shooting propagation workspace

Public DIR0 and DIR integration calls retain their signatures. Radial sweeps
use internal kernels accepting caller-owned coefficient buffers; the buffers
are reused within that sweep and are not global state. The zero-state path
clears the active coefficients and the turning-point coefficient. Adaptive
grid checks and numerical tolerances remain independent of this allocation
change. See `DF_SOLVER_CONTROLS.md` for live Salvat RADIAL comparisons.

The Breit configuration-average evaluator copies its radial mesh before
marking that copy immutable for geometry reuse. Caller-owned grids remain
mutable and cannot silently stale the integration geometry.

## 2026-10-05 review: explicit operator conventions

Use `atomx.angular.coefficients.reduced_C_jj` for the retained Grant jj phase
and `atomx.scattering.r_matrix.angular_cc.reduced_C_LS` for orbital LS coupling.
The ambiguous `reduced_C` imports are removed. `reduced_cl_j2` retains the separate
GRASP/FAC phase used by one-body transitions; do not substitute a renamed function
without checking its convention.

CG coefficients come from `angular.clebsch.clebsch_gordan`; kappa helpers from
`angular.recoupling`; label formatters from `config.subshell`. Private DW/local
copies and `coefficients.cg` are removed. Use `properties.transitions.rates.
einstein_A_prefactor` only with explicit magnetic-state strength;
`einstein_A_eL` accepts reduced strength and applies initial-state averaging.

`ConvolCache.clear()` is replaced by `release_shared_banks()`. This releases
shared banks process-wide for the grid. The known-wrong private legacy CI builder
has been deleted; use `physics.csf.matrix.build_ci_matrix`.
