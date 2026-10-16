# Native finite connected triples

Atomx now solves iterative **CCSDT** in its existing finite determinant CC engine,
through `compute(..., method='CCSDT')`. The default remains CCSD. This is an
equation-level finite anchor, not a reproduction of the 69-electron Yb⁺ paper,
valence-universal FS-RCCSDT, or an angular-compressed production atomic solver.
No external electronic-structure backend, fitted energy or borrowed correction
enters this implementation.

The [native source](../../../../src/atomx/methods/coupled_cluster.py) retains
the complete fixed-N determinant representation and solves

\[
 T=T_1+T_2+T_3,\qquad
 \langle\Phi_\mu|e^{-T}He^T|\Phi_0\rangle=0\quad(\mu\in S,D,T).
\]

The excitation exponentials terminate exactly after `min(N_occupied,N_virtual)`
powers; this order is independent of the connected-rank cutoff. Consequently,
disconnected quadruple and higher products remain in the wavefunction. The
standard exponential/projected construction is described by
[Bartlett and Musiał, Rev. Mod. Phys. 79, 291 (2007)](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.79.291).
That reference supplies method context; qualification here comes from explicit
independent finite-FCI/cumulant tests, rather than an asserted published-number
reproduction.

Private `_cluster_generators(masks, reference_mask, excitations)` and
`_exponential_action(t_matrix, vector, max_order, sign)` preserve reference
fermion phases and support a core cluster acting on a complete N+1 sector.
Consumers preflight the sector and allocations. Closed-core connected T3 and an
electron-attachment 3p2h operator are distinct; sharing these helpers does not
identify EOM-EA with the paper's FS-RCC equations.

## Independent numerical anchors

[Six new tests](../../../../tests/test_finite_ccsdt.py) independently assemble
one-/two-body fermion strings, include interleaved occupied/virtual signs, obtain
FCI by Hermitian diagonalization, and construct connected amplitudes using
`T=log(1+C)`. Orbital phase changes and reversed determinant storage are tested.
All determinants and all allowed excitation coordinates are retained.

| Complete model | SDT projected residual / Ha | Full eigenstate residual / Ha | CCSDT − FCI / Ha |
|---|---:|---:|---:|
| 3 electrons / 6 spinors, seed 1729, 20 determinants, 9+9+1 amplitudes | 7.36e−16 | 7.08e−16 | +2.22e−15 |
| 4 electrons / 8 spinors, seed 2718, 70 determinants, 16+36+16 amplitudes | 4.19e−15 | 3.27e−3 | −3.97e−7 |

The first model's connected T3 is `0.0007256834225672586`, compared with the
independent FCI logarithm `0.0007256834225668417`. Its CCSD full residual is
`0.004411813553867753 Ha`; triples are nonzero and materially distinguish the
two equations. The second model exposes the omitted connected-T4 defect:
a converged projected residual does not establish FCI or experimental accuracy,
and the CC energy need not be variational.

A separate complete four-electron, noninteracting 3+1 fragment test checks
additive energies and the full product-state projector. Its connected T4
vanishes while its disconnected quadruple coefficient remains nonzero; rank-3
CI lacks that product. A two-electron control has no possible T3 coordinate and
agrees with CCSD/FCI. Resource guards reject a complete space exceeding a cap
before generator allocation; unsupported method labels are rejected explicitly.

## Retained solver failure and correction

The first rank-3 test run used the inherited finite-difference MINPACK Jacobian:
**1 failed, 7 passed**. The noninteracting 3+1 test stalled at continuation 0.5
with projected residual `0.046717616394524694 Ha`. An independent full-FCI
logarithm at that same stage satisfied the SDT equations to
`1.7522875511710723e−15 Ha`, with connected T4 `3.22866e−17`. Thus the fixture,
full space and CCSDT equations were not changed to obtain a pass.

The solver now supplies the exact commuting-excitation derivative

\[
 \partial_{t_a}(\bar H|\Phi_0\rangle)
 =e^{-T}H\tau_a e^T|\Phi_0\rangle-\tau_a\bar H|\Phi_0\rangle.
\]

An independent full matrix-exponential central-difference test checks every
Jacobian coordinate (`step=1e−5`, absolute comparison gate `1e−9`). The unchanged
failed fixture now passes. Loss of finite-difference sensitivity on very small
symmetry-zero coordinates is a plausible mechanism; these checks establish the
solver correction, not a unique attribution of every prior MINPACK decision.
Continuation, `xtol=1e−11`, residual gate `1e−10 Ha`, resonance policy and physical
model remain unchanged. No amplitude screening or denominator floor was added.

## Resource and replay boundary

Defaults remain 256 determinants, 128 amplitudes, 128 MiB and 2000 function
evaluations per continuation stage. The memory gate now includes dense
generators plus a conservative similarity/Jacobian array estimate
`8*(2*D²+10*D*A+4*A²+4*D)` bytes. The 20/19 anchor uses 60,800 generator bytes plus
48,992 estimated workspace bytes; the 70/68 control uses 2,665,600 plus 609,408.
This is an array admission estimate, not a certified process-RSS bound;
Python/SciPy overhead is extra. Jacobian evaluation counts are reported.

Actual validation, 2026-10-13, `.venv` (NumPy 2.2.6 / SciPy 1.15.3), CPU 26,
BLAS/OMP/Numba threads 1:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=src:. timeout 90s taskset -c 26 .venv/bin/python -m pytest -q tests/test_finite_ccsdt.py tests/test_finite_ccsd.py tests/test_scaffolds.py --timeout=45 --durations=5
```

**12 passed in 0.57 s**, including the pre-existing native Be²⁺ Dirac–Coulomb
CCSD/complete-CSF anchor. This is correctness validation, not a controlled
performance comparison. Ruff passes the two owned Python files.

Source identity: prior dirty finite-CCSD source
`9116734e3f523521e00cf94fb700fa88402acffccef8a21e543b90cdb7f32ee8`;
current source `7601b301ab238b87d8daae8d0d2b5c493e05064923b309692eac7ba8314f030b`;
new tests `3a760b967ca69fca844b4c16e075cb89dc4e0695e9fd6af5052d08ab7db12555`.
Seeded Hamiltonian hashes are
`9a2d02370de3d8005f1baefc4b20b2062b3c4302cc5ee48adf06665949c9c871` (3e/6)
and `e28e8a769b6c4924a4d7398fb6dd5f7a3d2c4e70894e4551ea452dd3d78ef372` (4e/8).
The tests define the replayable integrals, masks, reference, signs and gates.

Documentation disposition: this owned note and source docstrings describe the
new branch, rank boundary and retained failure. Root integration owns the current
capability/testing/FSCC guides and stale scaffold wording. No new general rule is
promoted: independent rank/phase/residual anchors apply only within this finite
contract. Large-Yb triples, left states, dressed transition operators, nuclear
model/Breit coverage and production scaling remain separately unqualified.
