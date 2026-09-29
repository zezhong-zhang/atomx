# Migration Notes

Changes are grouped by the version that introduced them. The package version
is `atomx.__version__` (`src/atomx/__init__.py`); see `CHANGELOG.md` for the
per-version summary.

## Unreleased

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
| `methods.mcdhf.grid_shooting.occupied_1s2_reference_potential` | `atomx.methods.mcdhf.grid_functional.occupied_1s2_reference_potential` |
| `methods.mcdhf.shooting_preconditioner` | `atomx.numerics.shooting.preconditioner` |
| `methods.mcdhf.shooting_solver.ShootingSpace` | `atomx.numerics.shooting.electron_space.ShootingSpace` (public `atomx.ShootingSpace` is unchanged) |
| `methods.mcdhf.shooting_solver.solve_correlated_shooting` | `atomx.methods.mcdhf.grid_solver.solve_correlated_shooting` |
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

n = Atom(7, config="1s2 2s2 2p3",
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
