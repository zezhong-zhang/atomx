# Test-suite physics audit — rubric

Repo: `/Users/zhangzz/code/ASAP/atomx` (Python relativistic atomic structure).
Suite: 165 files under `tests/`, ~33 000 lines, 2 326 collected tests, ~550 s.

## Why this audit exists

The suite was written incrementally, often *after* a bug was found, and a large
part of it records **what atomx happened to print** rather than **what physics
requires**.  Two visible symptoms, both measured:

1. The same physical quantity is pinned to different numbers in different
   files with no way for a reader to tell which physics each number is.
   Ne ground state appears as `-128.6299783505843`, `-128.6920`,
   `-128.623437`, `-128.623390000` across four files — point nucleus vs
   finite nucleus vs Slater exchange vs a bug — and nothing says so.
2. Tolerances that are physically meaningless: `He+ total_energy == approx(-2.0, abs=0.5)`
   is a 25 % band on a hydrogenic energy known to twelve digits.

A test suite is only a physics gate if each assertion can be **derived
independently of atomx**.  Everything else is change-detection, which is fine,
but it must not be mistaken for validation.

## Classify every test (function/method) into exactly one class

### A — FIRST-PRINCIPLES
The expected value is derivable without atomx and without any other atomic
code.  Acceptable justifications:
- closed-form analytic result (hydrogenic `E_n = -Z^2/(2n^2)`, Coulomb/`R^k`
  integrals, Gaunt/3j/6j algebra, hydrogenic `<r>`, fine-structure formula);
- an exact theorem/symmetry: hermiticity, rotational invariance
  (`[H, J^2] = 0`), parity selection, `H` block structure, gauge invariance,
  Wigner–Eckart ratios, detailed balance, the `A`↔`gf` identity;
- sum rules: TRK `f`-sum, `Σ A_ij` closure, `Σ_i f_ij = N`;
- a **variational bound**: `E_exact ≤ E_MBPT2 ≤ E_HF` (this is how
  `tests/test_mbpt2.py` correctly sandwiches He);
- a known limit: `c → ∞` non-relativistic limit, `Z → 1`, weak/strong field,
  `r → 0` / `r → ∞` asymptotics, plane-wave Born limit, high-`T` Maxwellian
  limit;
- dimensional/unit identity (SI ↔ Hartree ↔ cm⁻¹ ↔ Rydberg round-trip).

**A-class requires the derivation or the theorem to be named in the test.**
A comment saying "should be" is not a derivation.

### B — EXTERNAL ORACLE
Compared against data atomx did not produce: NIST ASD, DBSR_HF, FAC/`sfac`,
GRASP, Salvat's DHFS tables, `~/code/ASAP/WF_TXT`, a cited paper table.
Must name the source **and** the number.  B-class is strong only if the number
is quoted in the test, not read at runtime from an atomx-generated file.

### C — INTERNAL CROSS-CHECK
Two atomx code paths agree (shooting vs B-spline, length vs velocity gauge,
direct vs adjoint, frozen vs re-solved).  Legitimate **only if the two paths
share no kernel**; state which kernels differ.  If they share the kernel, this
is D-class wearing a disguise — mark it `C-degenerate`.

### D — FROZEN SELF-REGRESSION
A literal that is only known because atomx printed it earlier.  Change
detection.  Mark with `@pytest.mark.snapshot` (the marker already exists and is
documented as "asserts nothing external").  D-class is not wrong, but a
D-class-only file must not be counted as coverage of a physics claim.

### E — TAUTOLOGY / NO-OP
Asserts nothing that can fail for a physics reason:
- `assert x is not None`, `assert len(r) == n`, dtype/shape checks with no
  value check;
- mock-call assertions (`mock.assert_called_once_with`) where the mock *is*
  the subject;
- a test that re-implements a one-line function and checks it agrees with
  itself;
- error-message *text* matching where the error type alone would do;
- tests of code that is never reached from any public entry point.
E-class should be deleted, or kept only if it guards a real silent-failure
mode (then say which).

### F — PLUMBING
Serialization round-trips, CLI parsing, provider contracts, dataclass
frozenness, deprecation messages.  Legitimate; not physics.  Keep, but do not
let it dilute the physics count.

## Method

For every test function: read the body, then decide.  Record evidence.

Required evidence per row: `file:line`, the asserted quantity, the literal, the
tolerance, the class, and one sentence of justification naming the principle or
source.  If you cannot name it after reading the surrounding context *and*
`git log -S<literal> -- <file>`, class it D and say so — do not guess A.

`git log -S` is the key tool for provenance: a literal introduced in the same
commit as a bug fix is D; a literal whose commit message cites NIST is B.

## Severity of findings

- **S1** — an A/B-class claim in the code or docs that no test actually gates,
  or a test that *would pass for a wrong result* (tolerance so wide, or
  comparison against a value atomx itself produced).
- **S2** — a quantity pinned inconsistently across files, or pinned with a
  tolerance that cannot distinguish the physics at issue.
- **S3** — D-class masquerading as validation (not marked `snapshot`), or
  duplicated coverage.
- **S4** — E-class noise, dead tests, misleading names.

## Deliverable

Write `docs/test_audit/<slice>.md` with:

1. A one-paragraph verdict for the slice.
2. A table of every test file in the slice: file, tests, class mix
   (e.g. `A2 B1 C3 D9 E2`), worst severity, one-line note.
3. A numbered findings list, each with file:line evidence, the measured
   number, and a **concrete remediation** (what to replace it with, naming the
   first-principles check that should stand in its place).
4. A "model tests" list: the tests in this slice that are already
   first-principles and should be imitated, with the reason.

Do not modify any file other than your own `docs/test_audit/<slice>.md`.
