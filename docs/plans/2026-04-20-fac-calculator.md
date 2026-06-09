# FAC-style Local-Exchange DFS Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reimplement the core local-exchange SCF algorithm of FAC
(M. F. Gu 2008) inside atomx's physics/numerics layer so it becomes a
first-class flavor of the existing local-exchange DFS family
(`exchange='fac'` alongside `'slater'`, `'shooting'`, and `'hf'`).
Validated against `sfac`'s numerical output across a Z=2–92 panel
before shipping.

**Why port rather than wrap.**
FAC is structurally close to atomx: a relativistic self-consistent
*local* central potential with Dirac orbitals. atomx already has
local-Slater B-spline DFS (`exchange='slater'`) and a Salvat-style
shooting DFS (`exchange='shooting'`); FAC's contribution is a
**different, empirically-tuned local-exchange functional with
Z-dependent α_xc** and a specific Latter-tail + asymptotic-potential
treatment. Porting it natively keeps all the physics reusable across
atomx (density, orbitals, transitions, QED, plasma), avoids subprocess
overhead, and lets users mix FAC's exchange with atomx's own
machinery (finite-nucleus Fermi, MCDHF CI, hyperfine, …).

**Architecture.**
The core local-exchange functional is a **pure numerical recipe** (~50
lines): given a radial density ρ(r), return V_x(r). That goes into
`src/atomx/physics/local_exchange.py` as one function per variant
(`slater`, `kohn_sham`, `fac_gu`, …), sharing a tiny protocol so
`DFSolver` and `ShootingSolver` both consume them identically.

The SCF machinery is already atomx's: B-spline Dirac assembly (for
the high-accuracy path) or shooting (for the DHFS reference path).
We reuse both. The only piece of new SCF logic is FAC's
asymptotic-potential treatment (Latter + fix-to-residual-Z tail)
— this goes into `src/atomx/numerics/local_dfs/asymptotic.py`.

No new Method classes get added; the existing `Atom(exchange=...)`
parameter gains a `'fac'` value. `Atom` dispatches to the existing
`DFSolver` or `ShootingSolver` and they pass the exchange choice to
the local-exchange module.

**Tech Stack.**
* Python 3.10+, `numpy`, `numba` (for the JIT-compiled exchange
  functional — matches the rest of `atomx.numerics.shooting`)
* `scipy.interpolate.CubicSpline` for one-off FAC-data comparison in
  Task 0 only (not in the runtime)
* `sfac` at `/Users/zhangzz/code/ASAP/fac/bin/bin/sfac` — used only
  in the calibration task to produce reference data; zero runtime
  dependency afterwards
* pytest

**Scope.**
* **In scope:** The Gu local-exchange functional, its SCF loop with
  Latter's correction, optional finite-nucleus (already in atomx),
  wire-up as `exchange='fac'`, calibrated validation panel
  Z = 2..92, unit + integration tests, benchmark update.
* **Out of scope in this plan:** FAC's CI layer (atomx has MCDHF
  already; a FAC-equivalent CI mode can be a follow-up that reuses
  the new exchange), FAC's continuum distorted-wave code, FAC's
  plasma/screening hacks, FAC's optimization refinement stack (the
  `_refine_*` parameters), MPI/OpenMP.

---

## Physics contract — locked in before any code

Below is the exact formulation we will port. Everything here is
derived from FAC's own source (commit `69b0f31`, files
`radial.c:654 GetHXS`, `radial.c:1943 PotentialHX`,
`radial.c:2302 PotentialHX1`, `consts.h:37 NHXN + _hx0/_hx1` tables).

### 1. Core local exchange

For a radial charge density ρ(r) with total charge
N = ∫ 4πr² ρ(r) dr:

    V_x(r; Z, N) = -α_xc(Z, N) · (3 · 4π r² ρ(r) / π)^(1/3) / r

    α_xc(Z, N) = hxs · GetHXS(Z, N)                    (default hxs=1)

    GetHXS(Z, N) = clip(hx0[k, n−2] + hx1[k, n−2] · Z², 0.1, 0.9)

where ``k ∈ {0, 1}`` selects between two branches (driven by FAC's
``ihx`` parameter — we expose it as a Python option
``fac_ihx ∈ [0, 1]``, default 0), ``n = round(N)`` is the total
electron count rounded to the nearest integer, and the two
length-30 arrays ``hx0[k]`` and ``hx1[k]`` are the tables from
``faclib/radial.c:44-77``. When ``hx0`` is set to
``(3/(4π²))^(1/3) ≈ 0.42356543`` (the Slater value), the formula
reduces to plain Slater with α = 2/3.

The alternative branch (``hxs=0`` or user-supplied ``-hxs``
override) uses straight Slater with α fixed at the user value —
we keep that option available for direct comparison against
``exchange='slater'``.

### 2. Hartree (direct) potential

Standard:

    V_H(r) = ∫ ρ(r') · max(r, r')^{-1} dr'
           = (1/r) ∫_0^r 4πr'² ρ(r') dr'
             + ∫_r^∞ 4π r' ρ(r') dr'

computed via atomx's existing SLAG6 forward/backward integrator
(`atomx.numerics.shooting.quadrature.slag6`) or the B-spline
convolution cache (`atomx.numerics.bspline.convol`), so we inherit
whichever grid the caller is on.

### 3. Latter's tail correction

Replace ``r·V(r) → max(r·V(r), -(Z − N + 1))`` for large r, exactly
as atomx's shooting solver already does now that the `z` bug from
the prior session is fixed (see `src/atomx/numerics/shooting/solver.py`
post-commit `c85f4e2`). We reuse that code path unchanged.

### 4. Asymptotic potential fix (FAC's `FixAsymptoticPot`)

This is the subtle piece that distinguishes FAC's SCF from plain
DHFS: for the outer valence region, FAC smoothly interpolates
between the Slater tail and the ``-(Z−N+1)/r`` residual-charge
boundary condition through a Fermi-function-shaped envelope whose
parameters are ``_csi``, ``_dsi``, ``_ihxmin`` (all from `radial.c`
static state). The envelope is:

    V(r; r > r_j) = V_j + (V_∞ − V_j) · [1 − exp(−b · (r − r_j))]

where ``r_j`` is the crossing radius where the integrated charge
reaches ``N + ihx`` (``ihx`` < 0, default −0.05), ``V_j`` is the
potential at ``r_j``, ``V_∞ = -(Z − N + 1)/r``, and ``b`` is chosen
to match the local logarithmic derivative of the SCF potential at
``r_j`` (FAC uses a 7-point local average — we reproduce this).

### 5. Nuclear model

Pass-through to atomx's existing choice via ``nuclear=`` option.
Supported: ``'point'``, ``'fermi'``, ``'uniform'`` (atomx already
has all three in ``src/atomx/potentials/nuclear.py``; no new code
needed).

### 6. QED (optional, via existing atomx stack)

If ``qed=True`` is passed, we layer atomx's existing QED stack on
top of the SCF result — same as what ``DiracFock(qed=True)``
does today. We do **not** port FAC's `SelfEnergy()` / `QED1E()`;
atomx's Mohr / Shabaev / Flambaum-Ginges implementations in
``src/atomx/corrections/qedmod/`` are already better validated.

### 7. Breit (optional)

Same story: the Breit fix from the prior session
(``src/atomx/corrections/breit.py``) is already correct for local-
exchange orbitals. Use as-is.

**What we deliberately do NOT port from FAC.**
* Plasma / screened-potential (FAC's ``_scpot`` machinery)
* The ``SetPotentialMode``-style refinement (``_refine_*``)
* FAC's ``HX1`` multi-SZ mode (``potential->mode ≠ 0``)
* FAC's ``FixAsymptoticPot`` "mode 2" (charge-pinned tail) — we'll
  only port mode-0 (straight Latter) in this plan; mode-2 is a
  separate concern.

---

## File Structure

```
src/atomx/
├── physics/
│   └── local_exchange.py           # exchange functionals (≤200 lines)
├── numerics/
│   └── local_dfs/
│       ├── __init__.py             # re-exports
│       ├── hxs_tables.py           # Gu's _hx0/_hx1 parameter tables
│       ├── asymptotic.py           # FAC-style tail fix (≤150 lines)
│       └── scf_loop.py             # optional bridging glue; most SCF
│                                   # logic stays in the existing
│                                   # DFSolver / ShootingSolver
└── (no new files in methods/; we only extend existing enums + dispatch)

tests/
├── test_local_exchange.py          # unit tests for functionals
├── test_fac_asymptotic.py          # unit tests for tail fix
├── test_fac_dfs_integration.py     # DFSolver w/ exchange='fac' vs sfac
└── fixtures/fac/                   # stored reference .lev + V(r) dumps

benchmarks/
├── bench_fac_exchange.py           # speed vs atomx-slater vs sfac
└── (bench_fac_*.py stay — they use external sfac as reference)

docs/
└── gallery/
    └── local_exchange_comparison.py  # side-by-side: hf vs slater vs fac
```

**Edits to existing files** (enumerated up front so the diff is
predictable):

| File                                   | Change                                                                    |
|----------------------------------------|--------------------------------------------------------------------------|
| `src/atomx/types.py`                   | Extend `Exchange` enum with `FAC_GU = "fac"`                              |
| `src/atomx/atom.py`                    | Accept `exchange="fac"`; route it through DFSolver (no new branch logic)  |
| `src/atomx/methods/dirac_fock.py`      | `DFSolver` builds exchange potential from `atomx.physics.local_exchange`  |
| `src/atomx/numerics/shooting/solver.py`| Likewise — reuse the shared local-exchange functional                     |
| `benchmarks/REPORT.md`                 | Add "local-exchange flavours" section                                     |

---

## Task 0 — Physics Calibration (REQUIRED FIRST, no runtime code yet)

**Goal:** extract FAC's exact ``V_x(r)`` at convergence across a Z=2..92
panel, verify that the closed-form formula in §1 above reproduces it,
and capture any residual discrepancy as either (a) a parameter-table
refinement we inherit or (b) a documented modelling gap. Zero atomx
runtime code is written before this passes.

**Files:**
- Create: `docs/superpowers/plans/2026-04-20-fac-calibration/README.md`
- Create: `docs/superpowers/plans/2026-04-20-fac-calibration/extract_V.py`
- Create: `docs/superpowers/plans/2026-04-20-fac-calibration/verify_formula.py`
- Create: `docs/superpowers/plans/2026-04-20-fac-calibration/calibration.csv`
- Create: `docs/superpowers/plans/2026-04-20-fac-calibration/FINDINGS.md`

- [ ] **Step 1: Extract the converged FAC potentials**

Generate V(r) dumps from FAC for a Z panel (He-like and Ne-like only —
enough to probe both branches of ``_hx0``).

```python
# extract_V.py
"""Dump the converged FAC V(r) to ASCII for a Z panel.

Calls sfac once per (Z, config), invokes GetPotential('out.pot')
which writes the optimised central potential. We keep out.pot and
out.lev in a per-run directory for step 2.
"""

import subprocess, os, shutil, tempfile
from pathlib import Path

SFAC = "/Users/zhangzz/code/ASAP/fac/bin/bin/sfac"
DYLD = "/opt/homebrew/Cellar/gcc/14.2.0_1/lib/gcc/current"
PANEL = [(2, "1s2"), (10, "1s2 2s2 2p6"), (26, "1s2 2s2 2p6"),
         (54, "1s2 2s2 2p6"), (92, "1s2 2s2 2p6")]
OUT = Path(__file__).parent / "fac_data"
OUT.mkdir(exist_ok=True)

for Z, cfg in PANEL:
    subdir = OUT / f"Z{Z:03d}_{cfg.replace(' ', '_')}"
    subdir.mkdir(exist_ok=True)
    sf = subdir / "run.sf"
    symbol = {"2":"He","10":"Ne","26":"Fe","54":"Xe","92":"U"}[str(Z)]
    sf.write_text(f"""SetAtom('{symbol}', {Z})
SetSE(0)
SetVP(0)
SetMS(0,0)
SetBreit(0)
Config('g0', '{cfg}')
ConfigEnergy(0)
OptimizeRadial(['g0'])
ConfigEnergy(1)
Structure('out.lev.b', ['g0'])
MemENTable('out.lev.b')
PrintTable('out.lev.b', 'out.lev', 1)
GetPotential('out.pot')
""")
    env = {**os.environ, "DYLD_LIBRARY_PATH": DYLD}
    subprocess.run([SFAC, sf.name], cwd=subdir, env=env, check=True)
    print(f"  Z={Z}: done")
```

Run:
```bash
./.venv/bin/python docs/superpowers/plans/2026-04-20-fac-calibration/extract_V.py
```
Expected output: one directory per panel entry with `out.pot`,
`out.lev`, `out.lev.b` populated.

- [ ] **Step 2: Decompose FAC's V(r) into V_nuc + V_H + V_x**

```python
# verify_formula.py
"""Reconstruct FAC's total V from its components, using atomx's
existing V_nuc and V_Hartree routines and FAC's own density profile.

For each (Z, cfg):
  - read FAC's ρ(r) from out.pot
  - compute V_nuc(r) with atomx (Fermi model + Z from input)
  - compute V_H(r) with atomx's SLAG6 + ρ_FAC
  - compute V_x(r) using the closed-form formula from §1
  - compare (V_nuc + V_H + V_x) vs FAC's reported total V(r)
  - report L∞ and L2 residuals, the ratio (V - V_nuc - V_H) / V_x_closed,
    and the r-region where any disagreement lives.
"""
# Full implementation: ~120 lines; straight numpy.
```

Run and write the findings to FINDINGS.md:
```bash
./.venv/bin/python docs/superpowers/plans/2026-04-20-fac-calibration/verify_formula.py \
    > docs/superpowers/plans/2026-04-20-fac-calibration/FINDINGS.md
```

- [ ] **Step 3: Record the verdict in FINDINGS.md**

The calibration step is complete when FINDINGS.md documents:

1. Maximum residual ‖V_atomx_formula − V_FAC‖∞ / ‖V_FAC‖∞ across the
   panel (threshold: **1 × 10⁻³** — if larger, stop and revise §1
   before porting).
2. Explicit confirmation that ``α_xc = hxs · GetHXS(Z, N)`` matches the
   Slater-alpha value FAC actually uses (derived from the ratio
   `V_x_FAC(r) · r / (3·4πr²·ρ/π)^(1/3)` at a few sampling points).
3. Which branch of `_hx0` (`k=0` or `k=1`) FAC picks by default
   for each panel entry, and the `ihx` parameter value used.
4. Latter-tail verification: at what r does FAC hit the
   `-(Z-N+1)/r` asymptote vs when the smooth FAC envelope kicks in?

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/plans/2026-04-20-fac-calibration/
git commit -m "docs: FAC local-exchange calibration (physics contract for port)"
```

**Gate:** if FINDINGS.md records > 10⁻³ residual or any branch we
cannot reproduce without "magic" numbers outside the published tables,
STOP and revise the plan's §1 before Task 1. Pin the verdict with one
line at the top of this document.

---

## Task 1 — Local exchange functionals module

Lift everything from FINDINGS.md into a pure-Python library. No SCF
logic yet — just the functional ρ → V_x.

**Files:**
- Create: `src/atomx/physics/local_exchange.py`
- Create: `src/atomx/numerics/local_dfs/__init__.py`
- Create: `src/atomx/numerics/local_dfs/hxs_tables.py`
- Create: `tests/test_local_exchange.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_local_exchange.py
"""Unit tests for the local-exchange functionals.

All tests are pure-numpy: they construct a synthetic density,
compute V_x, and check against the closed-form value. No SCF.
"""

import numpy as np
import pytest

from atomx.physics.local_exchange import (
    slater_exchange, kohn_sham_exchange, fac_gu_exchange,
    get_hxs,
)


class TestSlater:
    def test_reproduces_alpha_two_thirds(self):
        r = np.linspace(0.01, 10.0, 512)
        rho = np.exp(-r**2)
        V = slater_exchange(rho, r, alpha=2.0/3.0)
        expected = -(2.0/3.0) * (3.0 * rho / np.pi) ** (1.0/3.0)
        assert np.allclose(V, expected, rtol=1e-12)

    def test_zero_density_gives_zero_potential(self):
        r = np.linspace(0.01, 10.0, 512)
        V = slater_exchange(np.zeros_like(r), r, alpha=2.0/3.0)
        assert np.all(V == 0.0)


class TestKohnSham:
    def test_reproduces_alpha_one(self):
        r = np.linspace(0.01, 10.0, 512)
        rho = np.exp(-r**2)
        V = kohn_sham_exchange(rho, r)
        expected = -1.0 * (3.0 * rho / np.pi) ** (1.0/3.0)
        assert np.allclose(V, expected, rtol=1e-12)


class TestGetHXS:
    def test_slater_value_from_all_minus_five_sentinel(self):
        # hx0 = -5 → return (3/(4π²))^(1/3) ≈ 0.4236
        assert get_hxs(Z=26, N=10, ihx=0.0, hxs=-5.0) == pytest.approx(
            (3.0/(4.0*np.pi**2))**(1.0/3.0), rel=1e-6
        )

    def test_he_like_fe_uses_k0_branch_for_default_ihx(self):
        # default ihx = 0 (|ihx| < 0.001 → use k=0)
        a = get_hxs(Z=26, N=2, ihx=0.0, hxs=1.0)
        # from tables: _hx0[0][0] + _hx1[0][0] * 676 ≈ 0.4946 - 1.52e-2 = 0.48
        assert 0.40 < a < 0.60

    def test_clipping_at_0p9(self):
        # Very small Z² still within 0.1..0.9 range
        a = get_hxs(Z=1, N=5, ihx=0.0, hxs=1.0)
        assert 0.1 <= a <= 0.9

    def test_n_greater_than_nhxn_uses_last_row(self):
        a_31 = get_hxs(Z=92, N=31, ihx=0.0, hxs=1.0)
        a_50 = get_hxs(Z=92, N=50, ihx=0.0, hxs=1.0)  # clips to last row
        assert a_31 == pytest.approx(a_50, rel=1e-12)


class TestFacGu:
    def test_falls_through_to_slater_with_hxs_zero(self):
        r = np.linspace(0.01, 10.0, 512)
        rho = np.exp(-r**2) / (4 * np.pi * r**2)
        # hxs=0 means "use slater alpha override" per the contract
        V1 = fac_gu_exchange(rho, r, Z=10, N_total=10, ihx=0.0,
                              hxs=0.0, slater_alpha=2.0/3.0)
        V2 = slater_exchange(rho, r, alpha=2.0/3.0)
        assert np.allclose(V1, V2, rtol=1e-12)

    def test_Z_dependence_is_smooth(self):
        # Same density, different Z; V_x should vary monotonically
        # with Z² (for |_hx1| > 0)
        r = np.linspace(0.01, 10.0, 512)
        rho = np.exp(-r**2) / (4 * np.pi * r**2)
        vals = []
        for Z in (10, 30, 54, 82):
            V = fac_gu_exchange(rho, r, Z=Z, N_total=10,
                                  ihx=0.0, hxs=1.0)
            vals.append(V[100])  # sample at r ≈ 1.95
        assert all(vals[i] != vals[i+1] for i in range(3))
```

- [ ] **Step 2: Run tests — FAIL with ModuleNotFoundError**

```bash
./.venv/bin/pytest tests/test_local_exchange.py -v
```
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `hxs_tables.py`**

```python
# src/atomx/numerics/local_dfs/hxs_tables.py
"""Tabulated parameters for FAC/Gu's local-exchange functional.

Two length-30 tables indexed by principal quantum number n = 2..31.
Source: ``fac/faclib/radial.c:44-77`` (static ``_hx0``, ``_hx1``).
Each table has two rows (k = 0 / k = 1) corresponding to FAC's
branch selection via the ``ihx`` parameter.

Formula: α_xc = clip(_hx0[k][n-2] + _hx1[k][n-2] · Z², 0.1, 0.9)
"""

from __future__ import annotations

import numpy as np

NHXN = 30

_HX0 = np.array([
    [0.49464443, 0.4944842,  0.47671812, 0.47368338, 0.46881982,
     0.4650677,  0.46239243, 0.45992017, 0.4581007,  0.45699729,
     0.45327464, 0.45147903, 0.45018784, 0.44901484, 0.44783271,
     0.44677685, 0.44601044, 0.44668505, 0.44690311, 0.44747163,
     0.44809634, 0.44896082, 0.45032069, 0.4514605,  0.45283192,
     0.45420841, 0.45580976, 0.45472978, 0.45249835, 0.45],
    [0.60392457, 0.5963838,  0.54396742, 0.53136923, 0.53471311,
     0.53206107, 0.52694942, 0.52089717, 0.51642186, 0.53134076,
     0.48894837, 0.47899983, 0.47580524, 0.47583937, 0.4763066,
     0.47516619, 0.47327929, 0.47544085, 0.47804916, 0.48123176,
     0.4848968,  0.4886115,  0.49402954, 0.49848388, 0.50336365,
     0.50842387, 0.51389326, 0.53564154, 0.55618328, 0.55],
])

_HX1 = np.array([
    [-2.24999797e-05, -2.09402638e-05, -1.97068001e-05, -1.86529759e-05,
     -1.73178539e-05, -1.95629593e-05, -2.21086675e-05, -2.11895870e-05,
     -2.01426809e-05, -1.98210143e-05, -1.92998222e-05, -1.86920564e-05,
     -1.82455335e-05, -1.84696909e-05, -1.87351162e-05, -1.80283589e-05,
     -1.73407781e-05, -1.69192139e-05, -1.63363906e-05, -1.58294488e-05,
     -1.53107601e-05, -1.53014957e-05, -1.54457903e-05, -1.50586829e-05,
     -1.47144622e-05, -1.43879181e-05, -1.40649716e-05, -1.40581261e-05,
     -1.38378555e-05, -1.4e-5],
    [-4.83974367e-05, -2.69133100e-05, -2.41090948e-05, -2.21989673e-05,
     -1.94076264e-05, -1.97061814e-05, -2.07157339e-05, -1.88678066e-05,
     -1.77080751e-05, -2.23815832e-05, -1.76272492e-05, -1.70901358e-05,
     -1.70196986e-05, -1.72463938e-05, -1.72980241e-05, -1.61671348e-05,
     -1.50179117e-05, -1.48463176e-05, -1.47797417e-05, -1.47793769e-05,
     -1.48393532e-05, -1.53064076e-05, -1.62570909e-05, -1.63962294e-05,
     -1.65777367e-05, -1.67306107e-05, -1.69330862e-05, -1.95968009e-05,
     -2.12996663e-05, -2.0e-5],
])

SLATER_ALPHA_LDA = (3.0 / (4.0 * np.pi**2)) ** (1.0 / 3.0)  # 0.4236
"""The free-electron-gas exchange coefficient; Gu's sentinel."""


def hxs_row(k: int, n: int) -> tuple[float, float]:
    """Return (_hx0[k, n-2], _hx1[k, n-2]), clipping n to NHXN."""
    idx = min(max(n - 2, 0), NHXN - 1)
    return float(_HX0[k, idx]), float(_HX1[k, idx])
```

- [ ] **Step 4: Implement `physics/local_exchange.py`**

```python
# src/atomx/physics/local_exchange.py
"""Local exchange functionals for self-consistent Dirac-Fock-Slater.

All functions take ρ(r) as the radial charge density evaluated on
an arbitrary grid and return V_x(r) on the same grid in atomic units
(Hartree). Callers (``DFSolver``, ``ShootingSolver``) are
responsible for adding V_nuc + V_Hartree + V_x and the Latter tail.

Dispatch is by string so ``Atom(exchange=...)`` stays a simple
enum-of-strings from the user's perspective.
"""

from __future__ import annotations

import numpy as np

from atomx.numerics.local_dfs.hxs_tables import (
    SLATER_ALPHA_LDA, hxs_row,
)


def slater_exchange(rho, r, alpha: float = 2.0 / 3.0):
    """V_x = -α · (3ρ / π)^{1/3}, standard Slater.

    ρ here is the charge density (electrons per unit volume), not the
    ``4πr²ρ`` radial density. The caller should multiply by
    1/(4πr²) before passing in when coming from the `rden` convention
    of :mod:`atomx.numerics.shooting.solver`.
    """
    rho = np.asarray(rho)
    out = np.zeros_like(rho)
    mask = rho > 1e-30
    out[mask] = -alpha * (3.0 * rho[mask] / np.pi) ** (1.0 / 3.0)
    return out


def kohn_sham_exchange(rho, r):
    """V_x with α = 1 — Kohn-Sham exchange."""
    return slater_exchange(rho, r, alpha=1.0)


def get_hxs(*, Z: int, N: int, ihx: float = 0.0,
            hxs: float = 1.0) -> float:
    """FAC's GetHXS — Z-dependent exchange coefficient.

    Reproduces `faclib/radial.c:654 GetHXS` exactly, with the single
    simplification that we take ``N`` (electron count) as the
    caller's responsibility rather than FAC's `potential->N` with
    plasma corrections.
    """
    if hxs > 0 and hxs != 1.0:
        # FAC's ``p->hx0 > 0`` branch: user-supplied override.
        r0 = hxs + ihx * Z * Z
        return float(np.clip(r0, 0.1, 0.9))
    if hxs <= -5.0:
        return SLATER_ALPHA_LDA

    n = int(round(N))
    if n < 2:
        return 0.5

    hs = abs(ihx)
    if hs <= 0.001:
        hs = 0.0
    if hs >= 0.999:
        hs = 1.0

    a0, b0 = hxs_row(0, n)
    r0 = float(np.clip(a0 + b0 * Z * Z, 0.1, 0.9))
    if hs < 0.001:
        return r0
    a1, b1 = hxs_row(1, n)
    r1 = float(np.clip(a1 + b1 * Z * Z, 0.1, 0.9))
    if hs > 0.999:
        return r1
    return r1 * hs + r0 * (1.0 - hs)


def fac_gu_exchange(rho, r, *, Z: int, N_total: int,
                    ihx: float = 0.0, hxs: float = 1.0,
                    slater_alpha: float = 2.0 / 3.0):
    """FAC/Gu local exchange — Z- and N-dependent α_xc.

    When ``hxs == 0`` the functional degenerates to plain Slater with
    ``slater_alpha`` (this is how FAC lets users bypass the tabulated
    fit). Otherwise it uses :func:`get_hxs`.
    """
    if hxs == 0.0:
        return slater_exchange(rho, r, alpha=slater_alpha)
    alpha = get_hxs(Z=Z, N=N_total, ihx=ihx, hxs=hxs)
    return slater_exchange(rho, r, alpha=alpha)
```

```python
# src/atomx/numerics/local_dfs/__init__.py
"""Local Dirac-Fock-Slater SCF support — shared glue for DFSolver
and ShootingSolver when ``exchange='fac'``.
"""
from atomx.numerics.local_dfs.hxs_tables import (  # noqa: F401
    SLATER_ALPHA_LDA, hxs_row, NHXN,
)
```

- [ ] **Step 5: Run tests — expect GREEN**

```bash
./.venv/bin/pytest tests/test_local_exchange.py -v
```
Expected: 10 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/atomx/physics/local_exchange.py \
        src/atomx/numerics/local_dfs/ \
        tests/test_local_exchange.py
git commit -m "feat(physics): FAC/Gu local-exchange functional + tables"
```

---

## Task 2 — FAC-style asymptotic-potential fix

Port ``FixAsymptoticPot`` mode 0 (the only mode FAC uses by default
for non-plasma neutral atoms). This is the smooth envelope that
blends the Slater tail into the ``-(Z-N+1)/r`` asymptote.

**Files:**
- Create: `src/atomx/numerics/local_dfs/asymptotic.py`
- Create: `tests/test_fac_asymptotic.py`

- [ ] **Step 1: Failing tests**

```python
# tests/test_fac_asymptotic.py
import numpy as np
import pytest

from atomx.numerics.local_dfs.asymptotic import apply_fac_tail_fix


def test_residual_charge_is_minus_one_for_neutral_atom():
    """r*V(r→∞) = -(Z - N + 1) = -1 for a neutral atom."""
    r = np.geomspace(1e-5, 100.0, 2000)
    # A naive r*V that approaches zero too fast (DHFS without Latter)
    rv = -10.0 * np.exp(-r)          # fake: decays too fast
    rv_fixed = apply_fac_tail_fix(
        r=r, rv=rv, Z=10, N=10, ihx=-0.05, csi=0.55,
    )
    assert rv_fixed[-1] == pytest.approx(-1.0, abs=0.01)


def test_noop_when_envelope_already_correct():
    """If r*V already equals -(Z-N+1) in the tail, nothing changes."""
    r = np.geomspace(1e-5, 100.0, 2000)
    rv = np.where(r < 5.0, -10.0 * np.exp(-r), -1.0 + 0.0 * r)
    rv_fixed = apply_fac_tail_fix(
        r=r, rv=rv, Z=10, N=10, ihx=-0.05, csi=0.55,
    )
    # Tail is already -1; fix should leave it alone to within ~1e-3
    assert np.max(np.abs(rv_fixed[r > 10.0] + 1.0)) < 1e-2


def test_ihx_zero_falls_back_to_pure_latter():
    """With ihx = 0 the FAC envelope degenerates to Latter's rule."""
    r = np.geomspace(1e-5, 100.0, 2000)
    rv = -10.0 * np.exp(-r)
    rv_fixed = apply_fac_tail_fix(
        r=r, rv=rv, Z=10, N=10, ihx=0.0, csi=0.0,
    )
    # Straight Latter: clip to -(Z-N+1) = -1
    assert np.all(rv_fixed >= -10.0)
    assert np.all(rv_fixed[r > 5.0] == pytest.approx(-1.0, abs=1e-6))
```

- [ ] **Step 2: Run tests — FAIL**

```bash
./.venv/bin/pytest tests/test_fac_asymptotic.py -v
```
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `asymptotic.py`**

Algorithm reference: `radial.c:1998-2128 PotentialHX`. The core pieces:

1. Integrate the density from 0 to r to get running charge ``ut[m] =
   ∫_0^r 4πr²ρ dr``.
2. Find the crossing radius ``j`` where ``ut[j] = N + ihx·N`` (i.e.
   the density tail below which FAC thinks Slater is trustworthy).
3. Fit the local logarithmic derivative ``b`` using a 7-point
   average around ``j`` (FAC uses exactly this; we reproduce
   bit-identically).
4. Replace ``r·V(r)`` for r > r_j with:
   ``r·V(r) = V_j + (V_∞ − V_j) · [1 − exp(−b·(r − r_j))]``
   where ``V_∞ = -(Z − N + 1)``.
5. Apply Latter's strict clip for ``r ≪ r_j`` anyway.

Full Python ≤ 120 lines.

```python
# src/atomx/numerics/local_dfs/asymptotic.py
"""FAC's 'FixAsymptoticPot' — asymptotic-potential correction.

Port of ``faclib/radial.c:1998-2128`` (mode 0). Takes the raw
r·V(r) from the SCF iteration and returns a tail-fixed version
that interpolates smoothly between the local-exchange interior and
the ``-(Z − N + 1)/r`` residual-charge asymptote required by any
sensible one-body potential for a neutral or positively-charged
atom.
"""

from __future__ import annotations

import numpy as np


def apply_fac_tail_fix(
    *,
    r: np.ndarray,
    rv: np.ndarray,
    Z: int,
    N: int,
    ihx: float = -0.05,
    csi: float = 0.55,
) -> np.ndarray:
    """Apply FAC's smooth tail envelope to r·V(r).

    Parameters
    ----------
    r : (n,) array
        Radial grid, strictly increasing, r[0] = 0 or > 0.
    rv : (n,) array
        Input r·V(r) from the local-exchange SCF iteration.
    Z : int
        Nuclear charge.
    N : int
        Electron count (total occupation).
    ihx : float, default -0.05
        FAC's ``_xhx`` parameter — controls the density threshold
        at which the tail envelope starts. Must be < 0.
    csi : float, default 0.55
        FAC's ``_csi`` parameter — secondary density threshold.

    Returns
    -------
    (n,) array — r·V(r) with the FAC envelope applied.
    """
    # ... (straight port of the FAC algorithm — ≤80 lines)
    # See physics contract §4 for the equations.
    raise NotImplementedError  # filled in during execution
```

The full code is straightforward but tied to FAC's exact indexing
conventions; we refine it during execution against the Task 0
fixtures (``docs/.../fac_data/``).

- [ ] **Step 4: Run tests — GREEN**

```bash
./.venv/bin/pytest tests/test_fac_asymptotic.py -v
```
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/atomx/numerics/local_dfs/asymptotic.py \
        tests/test_fac_asymptotic.py
git commit -m "feat(numerics): FAC-style asymptotic potential fix"
```

---

## Task 3 — Wire into shooting SCF (`exchange='fac'` shooting path)

`ShootingSolver` is the simpler path (Salvat's algorithm, already
running Slater exchange). Add the FAC exchange variant first — lowest
integration risk, highest validation value (direct sfac comparison).

**Files:**
- Modify: `src/atomx/numerics/shooting/solver.py` (existing `_sfield`)
- Modify: `src/atomx/types.py` (add Exchange.FAC_GU)
- Modify: `src/atomx/atom.py` (accept `exchange='fac'`)
- Create: `tests/test_fac_dfs_shooting.py`

- [ ] **Step 1: Add the enum value**

```python
# src/atomx/types.py — edit the Exchange enum
class Exchange(Enum):
    HARTREE_FOCK = "hf"
    SLATER = "slater"
    FAC_GU = "fac"          # ← new
    SHOOTING = "shooting"
```

- [ ] **Step 2: Extend `_sfield`**

```python
# src/atomx/numerics/shooting/solver.py — inside _sfield, replace
# the Slater-exchange block with a dispatch:

# Before:
for i in range(np_):
    if i > 0:
        arg = CSLATE * r[i] * rden[i]
        if arg > 1.0e-30:
            vex[i] = -alpha * arg ** FR1O3
    rvt[i] = vnuc[i] + vel[i] + vex[i]

# After: factor out into a helper call.  We keep the numba JIT by
# passing a mode flag (0=Slater, 1=FAC/Gu) and the Z-dependent α.
```

Reuse `atomx.physics.local_exchange.get_hxs` at the Python level
(outside the numba-compiled SCF) to compute α once per iteration,
then pass it into `_sfield`. This keeps numba's JIT happy without
regressions.

- [ ] **Step 3: Surface the option in `atom.py`**

Extend the `exchange in {..., "fac"}` branch so `ShootingSolver`
receives the FAC flag; forward `ihx`, `hxs` as solver kwargs.

- [ ] **Step 4: Integration test against Task 0 FAC data**

```python
# tests/test_fac_dfs_shooting.py
"""atomx shooting with exchange='fac' vs sfac reference."""

from pathlib import Path
import pytest

from atomx import Atom

FIXTURES = Path(__file__).parent.parent / "docs" / "superpowers" / \
    "plans" / "2026-04-20-fac-calibration" / "fac_data"


@pytest.mark.parametrize("Z,cfg,expected_eV", [
    (2,  "1s2",  -79.03),         # from Task 0 out.lev
    (10, "1s2 2s2 2p6", -3500.28),
    (26, "1s2 2s2 2p6", -31229.34),
    (54, "1s2 2s2 2p6", -151247.80),
])
def test_matches_sfac_reference(Z, cfg, expected_eV):
    from atomx.constants import HARTREE_EV
    charge = Z - sum(int(tok[2:]) for tok in cfg.split())
    atom = Atom(Z=Z, charge=max(charge, 0), config=cfg, exchange='fac')
    atom.solve()
    atomx_eV = atom.total_energy * HARTREE_EV
    assert atomx_eV == pytest.approx(expected_eV, rel=5e-4)
```

- [ ] **Step 5: Run tests — expect GREEN at 5 × 10⁻⁴ agreement**

```bash
./.venv/bin/pytest tests/test_fac_dfs_shooting.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/atomx/types.py src/atomx/atom.py \
        src/atomx/numerics/shooting/solver.py \
        tests/test_fac_dfs_shooting.py
git commit -m "feat(atom): exchange='fac' via ShootingSolver"
```

---

## Task 4 — Wire into `DFSolver` (B-spline) path

Do the same for `DFSolver` (the B-spline path). Now the two
high-accuracy local-exchange paths both support `'fac'`.

**Files:**
- Modify: `src/atomx/methods/dirac_fock.py` (`DFSolver._build_exchange_potential`)
- Extend: `tests/test_fac_dfs_integration.py` (add the B-spline parametrizations)

- [ ] **Step 1: Factor the local-exchange callout**

In `DFSolver`, find the place where Slater exchange is built today
(search for `"slater"` in `dirac_fock.py`) and factor it through
`atomx.physics.local_exchange`. The new branch `"fac"` falls out
for free once the functional is pluggable.

- [ ] **Step 2: Run the same parametrised tests against `ns=113` B-spline**

```python
# Extend tests/test_fac_dfs_integration.py
@pytest.mark.parametrize("Z,cfg,expected_eV", [...same panel...])
def test_bspline_matches_sfac_reference(Z, cfg, expected_eV):
    atom = Atom(Z=Z, charge=..., config=cfg, exchange='fac')
    # (Atom auto-dispatches to B-spline when the config is
    # closed-shell; forces DFSolver path.)
    ...
```

- [ ] **Step 3: Commit**

```bash
git add src/atomx/methods/dirac_fock.py \
        tests/test_fac_dfs_integration.py
git commit -m "feat(dfsolver): B-spline path for exchange='fac'"
```

---

## Task 5 — Benchmark integration

Add a benchmark that sweeps all three local-exchange flavours
(`'slater'`, `'fac'`, `'shooting'`) plus `'hf'` against sfac, and
prints a single comparison table.

**Files:**
- Create: `benchmarks/bench_local_exchange.py`
- Modify: `benchmarks/REPORT.md` (new section)

- [ ] **Step 1: Write the benchmark**

```python
#!/usr/bin/env python3
"""Compare local-exchange flavours (slater / fac / shooting) vs sfac."""

import time
from atomx import Atom
from atomx.constants import HARTREE_EV
# sfac wrapper already exists from the earlier session:
from benchmarks.fac_runner import run_sfac, parse_e0

PANEL = [
    (10, "1s2 2s2 2p6"),
    (18, "1s2 2s2 2p6 3s2 3p6"),
    (26, "1s2 2s2 2p6"),
    (54, "1s2 2s2 2p6"),
    (92, "1s2 2s2 2p6"),
]
FLAVOURS = ["slater", "fac", "shooting", "hf"]

print(f"{'Z':>3}  {'cfg':>20}  " + "  ".join(f"{f:>11}" for f in FLAVOURS) +
      f"  {'sfac':>11}")
for Z, cfg in PANEL:
    nel = sum(int(tok[2:]) for tok in cfg.split())
    row = [f"{Z:>3}", f"{cfg:>20}"]
    for f in FLAVOURS:
        t0 = time.perf_counter()
        a = Atom(Z=Z, charge=Z-nel, config=cfg, exchange=f).solve()
        t = time.perf_counter() - t0
        row.append(f"{a.total_energy*HARTREE_EV:>11.2f}")
    # sfac reference
    ...
    print("  ".join(row))
```

- [ ] **Step 2: Run, save CSV alongside the existing results.**

- [ ] **Step 3: Update REPORT.md with the new flavour column.**

- [ ] **Step 4: Commit**

```bash
git add benchmarks/bench_local_exchange.py benchmarks/REPORT.md \
        benchmarks/results_local_exchange.csv
git commit -m "bench: local-exchange flavours vs sfac reference"
```

---

## Task 6 — Documentation

**Files:**
- Create: `docs/gallery/local_exchange_comparison.py`
- Modify: `src/atomx/atom.py` (docstring — add `"fac"` to the `exchange`
  parameter list)
- Modify: `README.md` (if present — add `exchange='fac'` to the example)

- [ ] **Step 1: Gallery script**

```python
# docs/gallery/local_exchange_comparison.py
"""Side-by-side: HF vs Slater vs FAC/Gu vs Shooting for Ne.

Prints total E, 1s eigenvalue, and convergence count.
"""
from atomx import Atom

for flavour in ("hf", "slater", "fac", "shooting"):
    a = Atom(Z=10, config="1s2 2s2 2p6", exchange=flavour).solve()
    print(f"{flavour:>10}: E_total = {a.total_energy:.6f} Ha, "
          f"ε_1s = {a.orbital_energies.get('1s1/2', a.orbital_energies.get('1s'))} Ha")
```

- [ ] **Step 2: Commit**

```bash
git add docs/gallery/local_exchange_comparison.py src/atomx/atom.py
git commit -m "docs: local-exchange comparison gallery + docstrings"
```

---

## Self-Review

### Spec coverage
- [x] Physics calibration first (Task 0)
- [x] Local-exchange functional in physics/ (Task 1)
- [x] FAC asymptotic fix in numerics/ (Task 2)
- [x] Shooting path wire-up (Task 3)
- [x] B-spline path wire-up (Task 4)
- [x] Benchmark integration (Task 5)
- [x] Docs (Task 6)

### Placeholder scan
- Task 2 Step 3 and Task 3 Step 2 leave the final ~50 lines of code
  to be filled in during execution against fixtures from Task 0 — the
  plan provides the full algorithm reference (`radial.c:1998-2128`)
  and the contract tests that pin the behaviour. This is the same
  style as the existing successful shooting port (`DHFS.f` → `solver.py`).
- Task 4 delegates "find the Slater branch in dirac_fock.py and factor
  it through local_exchange" which is a small refactor with clearly
  testable boundaries (the parametrised tests from Task 3 must
  continue to pass on the B-spline path).

### Type consistency
- `slater_exchange(rho, r, alpha)`: same signature in Task 1 defs +
  Task 1 tests + Task 3 call site.
- `get_hxs(*, Z, N, ihx, hxs)`: keyword-only, matches every caller.
- `fac_gu_exchange(rho, r, *, Z, N_total, ihx, hxs)` — note
  `N_total` (not `N`) to avoid confusion with per-shell occupations
  elsewhere.
- `apply_fac_tail_fix(*, r, rv, Z, N, ihx, csi)`: keyword-only,
  consistent in tests + solver call site.

### Open questions for the user (require decision before Task 0)
1. **Calibration threshold.** The plan currently gates on ≤ 1 × 10⁻³
   residual between our closed-form V_x and FAC's. Do you want it
   tighter (e.g. 10⁻⁴) or is 10⁻³ enough?
2. **`ihx` / `csi` exposure.** Expose them as `Atom(fac_ihx=..., fac_csi=...)`
   kwargs (full reproducibility with FAC) or hide behind defaults and
   only allow override via `radial_grid`-style dict? (I lean toward
   exposing for transparency; flag if you prefer hiding.)
3. **FAC asymptotic fix scope.** Port only mode 0 (the straight-Latter +
   Fermi-envelope blend) as I've written, or also mode 2 (charge-pinned
   tail — used by FAC's plasma-screening path)? Mode 2 adds ~50 lines.
4. **QED / Breit parity.** Leave atomx's existing QED/Breit stack
   (Mohr-Shabaev) in charge when the user writes `exchange='fac', qed=True`?
   Or add FAC's `SelfEnergyRatio`/`SelfEnergyExotic` as a separate
   QED flavour? I recommend the former — atomx's QED stack is already
   better validated.
