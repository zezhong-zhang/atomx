# Bound same-point CI Hamiltonian retention

This additive implementation retains the complete matrix already built for a
native CI trial. It removes the need for a second public matrix build when a
checkpoint consumer reads that same trial. The default is `False`; no optimizer,
radial snapshot, Hamiltonian formula, state-average weight, physical space or
scientific threshold changed.

The final [unit contract](unit_contract003.json) and
[execution](unit_execution003.json) record **85 focused tests passed**, in
2.46 s of pytest time, 2.718 s observed process time, peak child RSS 508,064 KiB.
These are bounded small native controls, not a controlled speedup measurement or
a completed 531-CSF qualification. CPU affinity and all numerical thread limits
were one, the address-space cap was 3.95 GiB and the hard wall cap 300 s.
The [log](unit_controls003.log) preserves actual test results. Earlier
[78-test controls](unit_execution.json) and
[85-test boundary revision](unit_execution002.json) remain historical records.

## Public consumption and scope

```python
from atomx.methods.ci import ci_solve

ci = ci_solve(csfs, orbitals, orbital_keys, workspace, Z, ns,
              nuclear_potential=nuclear, max_determinants=cap,
              retain_hamiltonian=True)
H = ci.hamiltonian_for(csfs, orbitals, orbital_keys, workspace, Z, ns,
                       nuclear_potential=nuclear, max_determinants=cap)
# H is the immutable complete original float64 matrix, in Hartree.
# Independently check H @ ci.mixing_coefficients, all eigenvalues and Gram.
```

`CSFOrbitalFunctional(..., retain_hamiltonian=True)` propagates the same explicit
option to its ordinary complete CI trials. A supplied retained `CIResult` is
checked against the actual point before its signed density/gradient is consumed.
The existing optimizer already carries accepted trial CI results to the gradient
and checkpoint callback; its implementation does not need to change.
`CIRadialSnapshot.solve(..., retain_hamiltonian=True)` likewise forwards this
option through its existing public interface without a radial-snapshot edit.

The typed `CIHamiltonianSnapshot` is produced only by
`build_ci_hamiltonian_snapshot` from public physical inputs. It reuses public
`CIReferenceBinding.capture/validate` and calls unchanged public
`build_ci_matrix` once. It binds the full ordered CSF electron counts, parity and
angular signatures; orbital-basis order; **all** provided signed radial spinors,
including unused keys; complete workspace; sampled nuclear operator; Z/ns;
normalized determinant cap; representation, units and literal DC operator terms.
`identity_payload()` returns a fresh serializable header; `input_sha256` and
`matrix_sha256` are separate identities. Objective, source epoch, tracking and
optimizer metric remain separate checkpoint identities.

The matrix owns immutable bytes. Caller/builder aliases cannot mutate it, and
NumPy `setflags(write=True)` rejects. Both before/after producer input validation
and the consumer getter reject changed signed amplitudes, CSF/order, workspace,
nuclear contents or keys. Arbitrary matrix construction/import and unpickling
are rejected. Save complete physical arrays and headers for independent replay;
the local Python object is not a persisted scientific certificate.

First supported retention is **complete dense real B-spline Dirac-Coulomb**.
Explicit subset requests reject even when `n_states == len(csfs)`. Grid,
Davidson, Breit/frequency maps, scalar shifts (including an explicit empty map)
and effective one-body operators reject retention before a physical build.
Only Python `bool` flags are accepted. Existing ordinary default routes remain
available for these operators and representations; coverage is not silently
reduced to DC. `max_determinants=None` and its explicit default value have the
same normalized identity. A missing owner rejects the getter rather than
performing an unreported rebuild.

Ownership does not certify mutable CI eigenvalues/vectors. Complete eigenpair
residual, Gram, target/tracker identity, signed gradient and original metric
checks remain mandatory. Existing legacy `CIResult` inputs without an owner may
still supply explicitly checked gradients; they cannot provide a reuse
Hamiltonian. No saved-column equality inside a degenerate eigenspace is imposed.

## Actual controls and source identity

The [new focused tests](../../../tests/test_ci_hamiltonian_snapshot.py) measure
actual forwarding counters: one full build for an energy trial, **zero additional
builds** for accepted gradient completion and checked checkpoint retrieval, and
one additional build for independent uncached replay. The complete H and all
energies match the ordinary producer entrywise; every signed radial-gradient
component matches. Full residual/Gram gates remain in the tests.

Controls exercise real post-build P/workspace/CSF/key/nuclear mutation; byte
ownership, dataclass/header immutability; stale supplied retained-CI rejection
before either gradient or rebuild; unused spinor keys; nuclear closures;
default/legacy compatibility; normalized cap and operator/spectrum/flag refusal.
Affected dense-cluster, radial-snapshot, basis-tracking, determinant-cap and
signed radial-gradient tests include independently rediagonalized derivatives
at ns35 and ns120. This validation does not establish experimental accuracy,
stationarity of a larger model or end-to-end optimizer performance.

The predecessor is the immutable [design](../checkpoint_hamiltonian_reuse_design/README.md)
and complete353 source map
`10d4813c3ab626215f4bbc6b9f478fc0b4478a801d477d57fd35928338573f2f`.
Only two existing source files changed and one physics-owned source was added.
The final complete354 map is
`3304f5895bcfce70708f1aa9a77bca8a7cfc3eca9a9d393380035c42ed67df2d`;
[source identity](source_identity002.json) and [full archived source](source002/atomx/)
bind the actual dirty worktree. The original candidate354 map23c38 and its
complete [source archive](source/atomx/) remain saved. The final source-only
revision adds API docstrings; [AST evidence](documentation_source_identity.json)
shows all non-docstring ASTs identical. The final unit controls nevertheless ran
on the exact final source. No frozen publicrun/proof/result was rewritten.

## Documentation disposition and pending production qualification

| Entry point | Disposition and reason |
| --- | --- |
| CIResult/ci_solve docstrings | Updated with typed retention, operator scope, full-spectrum dense selection and mandatory eigenpair verification. |
| CSFOrbitalFunctional docstring | Updated with explicit trial retention and same-point supplied-CI validation; no default change. |
| Architecture ownership / capability / migration / verification guides | Reviewed; existing statements do not claim H retention or require a different default. Retained; this additive API is documented at its changed consuming interfaces and this current topic. |
| Frozen design/run003 and older scientific proof | Retained as source-bound historical evidence; it is not rewritten as an implemented optimization or a stationarity pass. |
| Root Yb current topic/navigation | Parent integration owns the incoming link and the eventual full531 qualification status. |

Native consumer preparation in `hamiltonian_snapshot_qualification/` will bind
the exact run003 accepted320 point, complete531 model, original P/x/inverses,
tracker, weights, source/nucleus and independent1011-check reference. It must
explicitly review the source-only objective-identity transition and compare
retain=False fresh H/CI/full gradient against the retained trial and saved g/pg,
keeping original H/CI, 1e-9 g and 1e-10 pg equality gates. Root authorization and
the consumer's prospective resource contract are required before execution.
The unchanged reference residual 1.9434848235489216e-6 remains above 1e-7;
retention cannot change that scientific status.

Expected consumer counts are three ordinary public builds (ordinary CI,
independent H, retained trial), two CI solves and two full gradients;
getter/accepted-gradient reuse must add zero H/CI. Timing includes binding,
copy/hash, callback and archive cost. Matrix retention adds at least 2.15 MiB
per live531 CI and accepted/trial matrices can coexist. Paired stage timing,
controlled-load admission and a later identical whole-solve trajectory are
separate from this unit implementation; no CPU/GPU speedup is claimed here.

Distillation: when a complete CI result feeds a same-point scientific
checkpoint, retain its public-built H with complete physical identity, use the
checked getter, and keep independent residual/Gram/gradient replay. Small-model
proof supports the API and redundancy removal only. Larger-model timing and
scientific status remain separately qualified; no new general rule is needed.
