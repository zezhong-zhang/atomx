# Import-style successor and interpretation correction

This successor preserves the original [scientific evidence](../README.md),
[failed import-style check](../lint_import_style_failure.json), and exact
[lint-original snapshots](../lint_original/manifest.json). Only import order,
precise E402 exemptions for scientific imports intentionally after resource
gates, and the administrative repository ancestor depth for this nested
directory changed. Production code, physical arithmetic, source/input gates,
resource limits and tolerances did not change.

The [science identity chain](science_identity_chain.json) compares complete AST
structure after excluding import declarations and the ROOT administrative
binding. Both probe and independent verifier are identical under that rule.
The premeasurement [new contract](contract.json) binds the actual successor
source rather than reusing an old source hash.

## Actual successor execution

The successor [probe](probe.py) and [independent verifier](verify.py) both
executed under the original one-thread, 3 GiB, 120-second per-mode and
300-second total contract. DC, static Breit and dynamic transverse all passed
the unchanged 1e-11 Ha signed source/K/energy and 1e-12 Ha retained root gates.
[Results](result.json) and [saved-array replay](verification.json) are newly
executed evidence. Every scientific case value and all 39 saved arrays are
bitwise identical to the original increment; timing/source metadata differs.
All 354 successor source snapshots and original/archived input hashes passed.

[Style checks](style_checks.json) passed Ruff for these two successor scripts.
The original scripts' Ruff failure remains archived and is not relabeled as
passing.

## Correction to the partial-operator interpretation

The original probe computes `fock_K = C_Q† F_N C_Q`. This is the normal-ordered
Fock **QQ projection**, not the complete 12-block `Q[F_N,T]|Phi>` operator.
In the T1-to-T2 block it omits the subtraction `T1 F_N|Phi>` when the orbitals
are noncanonical. The recorded roughly 0.4263 Ha quantity therefore means
the **F_N QQ projection + PP/HH control residual**. It must not be presented
as a residual of the complete Fock commutator plus PP/HH, or as an experiment
or heavy-atom energy error. Historical JSON keys are retained for bitwise
source/output succession; the interpretation here supersedes that wording.

The complete native K/root comparison is unaffected: that calculation uses
the full `Q(H T - T H)|Phi>` and the public native solver's actual noncanonical
subtraction. The narrower T2-to-T2 difference is also unaffected: in the DD
block the Fock commutator equals the projected Fock action. The actual maximum
missing off-diagonal DD after F_N QQ + PP/HH is 0.07935305 / 0.07937383 /
0.07937383 Ha for DC/static/dynamic. These are appropriate signed targets for
the next native particle-hole contraction. Independent full PH attribution
belongs to its separate review evidence, not this style-only successor.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  .venv/bin/python docs/atomic_native_roadmap_2026_10_11/sd_linked001/style_successor/verify.py
```

This remains a bounded finite native physics discriminator. No production
heavy-atom solver, triples, new NIST roots or performance speedup is claimed.
