# Native complete state actions

The new [ClusterAction](../../../../../src/atomx/numerics/cluster_actions.py)
reuses [signed fermion actions](../../../../../src/atomx/numerics/fermion.py)
to apply pure core-cluster strings, exact `exp(±T)` and `exp(-T) H exp(T)` to
mask-to-coefficient dictionaries. It allocates neither a complete determinant
mask list nor an amplitude×determinant×determinant generator bank. Every
reachable signed intermediate is retained; only exactly represented zeros are
removed. This is a finite state-action engine, not a Wick/normal-order diagram
generator or a production atomic RCCSDT implementation.

Root integrates the explicit `action_representation='state_action'` option into
finite CC and EOM-EA; default `dense` remains available. Those consumers still
enumerate complete sectors, retain dense H/Jacobian or projected matrices and
use their original residual/space/resource controls. Removing a generator bank
does not remove combinatorial sector growth or establish a speedup. The previous
[triples qualification and solver failure](../native_triples.md) is retained
unchanged.

## Stable scalar API

```python
ClusterAction(reference_mask, spinor_count, rank_limit, sector_electrons,
              determinant_cap=256, amplitude_cap=128,
              term_application_cap=2_000_000)
action.apply(amplitudes, state)
action.unit_action(index, state)
action.exponential(amplitudes, state, sign)  # sign = +1 or -1
action.similarity(amplitudes, hamiltonian_terms, state)
```

Results are owned signed state dictionaries. Complete ordered `excitations`,
`amplitudes_by_rank`, `sector_dimension` and `max_order` are immutable plan
metadata. Core reference and action sector may have different particle numbers;
`unit_action` is the same reference-normalized coordinate used in exact CC
derivatives. Real/complex scalar amplitudes and states are supported by this
engine; consuming methods retain their own stricter physical interfaces.
H terms include the constant and all number-conserving strings, with in-range
indices and at most 2M elementary operations. No two-body rank or four-term BCH
shortcut is assumed by the engine.

Let D=comb(M,N_sector), A be the complete selected excitation count,
K=min(N_core,M−N_core), and L_H the input operator-term count. Before fermion
actions, the full-sector pair-count bounds are D·A (constructor/apply), D
(unit action), K·D·A (exponential), and D·(2K·A+L_H) (similarity). Terms with zero
amplitude/coefficient remain in this census. Bounds apply **per scalar call**,
not to a nonlinear solve, batched Jacobian or all EA columns. Each pair can
require up to 2M elementary fermion operations. Cluster/state storage is
O(A·rank+D); validating/copying an explicit H sequence additionally uses
O(L_H·M). Python dictionary overhead is extra and no certified process-RSS
limit is claimed.

## Independent verification

[Seven tests](../../../../../tests/test_cluster_actions.py) independently
assemble fermion signs and dense operator matrices. They compare complete
complex signed core/N+1 vectors, every unit coordinate, inverse exponentials
and similarity actions against dense matrix exponentials; an explicitly
two-body operator also matches independent nested-commutator BCH arithmetic.
An independent complete three-electron FCI eigenvector and `log(1+C)` supply a
nonzero connected T3 and the complete transformed residual. Further controls
retain a disconnected T3×T1 quadruple and a `1e−200` amplitude, reject caps and
invalid operators before action, and preserve empty/filled-sector identity.

Actual run, 2026-10-13, `.venv`, CPU 26, BLAS/OMP/Numba threads 1:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=src:. timeout 45s taskset -c 26 .venv/bin/python -m pytest -q tests/test_cluster_actions.py --timeout=20 --durations=5
```

**7 passed in 0.18 s**; Ruff passed. This timing is validation scope, not a
controlled performance result. An isolated import/action with a meta-path guard
rejecting every `atomx.methods` import also passed. Source review of root's
adapters, complete excitation census, analytic Jacobian and EA normalization
found no practical defect at the source hashes recorded in [receipt](receipt.json).
Root owns the separate numerical integration panel and current guides.

A broader existing architecture panel initially returned **13 passed / 1
failed**: the shooting subprocess reported Numba-cache
`ReferenceError: underlying object has vanished`. That exact existing failure
then passed with a task-local `NUMBA_CACHE_DIR` (**1 passed in 13.27 s**); no
shooting source, criterion or shared cache was changed. The receipt preserves
both observations. Generated task-local compiler caches are expendable artifacts.

## Reference scope and next increment

The saved EXP-T models at commit `aa8d809ad5818d4aa618dcdadbd7f12257ecbc28`
separate [0h0p core T3 feedback](https://github.com/aoleynichenko/EXP-T/blob/aa8d809ad5818d4aa618dcdadbd7f12257ecbc28/src/rcc/models/sector00_ccsdt.c)
(singles/doubles/triples definitions at lines 114/143/251) from
[0h1p S3 and folded terms](https://github.com/aoleynichenko/EXP-T/blob/aa8d809ad5818d4aa618dcdadbd7f12257ecbc28/src/rcc/models/sector01_ccsdt.c)
(lines 285/1806). These were read for model coverage; no implementation text was
copied. Exact core similarity acting on N+1 proves finite EOM actions, not the
separate valence-universal FS-RCC equations or the Yb⁺ paper's input model.

The smallest useful continuation is to derive one complete linked contraction
family from native fermion algebra and validate all signed finite entries
against this action oracle. A full normal-ordered tensor BCH residual needs
new contraction/normal-order machinery and explicit core/valence projections;
`fermion.py` alone does not provide it. No larger cap, external backend,
experimental fitting, borrowed correction or production precision upgrade is
introduced. Documentation disposition: this new receipt/README owns the engine
evidence; root owns current guides. No new general rule is promoted.
