# Complete-state NIST audit and gap queue

`run004/` is the current frozen evidence. `run001/`, `run002/` and `run003/` retain prior
successful snapshots with earlier source/dependency/input manifests. The new
reusable `tools/atomic_nist_audit.py` accepts caller-owned resolved state records,
local NIST records, a predeclared complete state panel, explicit intervals and
multipoles. It reuses the existing ASD `Level` adapter, configuration convention,
CODATA conversion and air/vacuum conversion. It performs no I/O or solver dispatch.

Every state owns a deep immutable Hamiltonian/core/space/nucleus/normalization/
units/energy-zero identity snapshot. Different models, partial theory panels and
duplicate zero states reject. Excited-state matching requires unique species,
configuration, J, parity and evaluated unflagged energy, without choosing the
nearest energy. Nominal configuration labels do not certify mixing weights.

Observed lines require consistent endpoint IDs and labels and a known native
medium. Missing and ambiguous lines stay in the required denominator. Ritz and
level-derived wavelengths remain separate, without extra observational credit.
E1/E2/M1/M2 rules test necessary electronic J/parity conditions only; no matrix
element, LS purity, rate or emitted intensity follows. Missing uncertainties are
not replaced by print digits or stationarity residuals; nonzero endpoint
uncertainties need stated covariance. Without a theory uncertainty, standardized
residuals remain unknown even if a reference uncertainty exists. Air wavelengths
are converted for comparison; their uncertainty propagation is left unknown.
The current covariance guard requires exactly zero covariance for an exact zero
endpoint and uses a scaled ULP bound for tiny uncertainties. Scaled propagation
preserves finite nonzero values down to 1e−200 and up to 1e200 in controls. The
previous absolute covariance slack is superseded; actual Yb unknown uncertainties
and reported discrepancies are unchanged.

The actual run consumes the complete five archived Yb II root payloads and the
current full-array norm certificate from `parallel_six_norm001/run002`, and
archives the full local ASD5.12 database. Its immutable Yb subset contains 349
levels and 327 lines. The same four evaluated excitation references and two
splittings reproduce the original comparison and atlas, with the original 1%
gate. Excitation errors remain 0.9250%, 1.1684%, 4.1413% and 3.3799%; the two
splitting errors remain +3.1465% and −9.3620%. One of four excitations and zero
of two splittings pass. Two of four ground-state target transitions have unique
observed vacuum wavelengths; the two E2 targets lack observed lines in this
snapshot. All four target excited-level uncertainties are absent.

`audit.json` keeps each score, denominator, source, classification, unknown and
original numerical certificate. `iteration_requests.json` orders real failed
observables by relative discrepancy and binds the complete five states and
common identity. Hypotheses include complete-space convergence, nonlinear/triple
correlation, common omitted operator corrections and state mixing; the gap alone
does not identify a cause. Requests require a new prospective full-space resource
contract and cannot launch a cropped or spliced model.

```sh
.venv/bin/python docs/atomic_pci_sd_native_2026_10_09/parallel_six_nist001/probe.py --output docs/atomic_pci_sd_native_2026_10_09/parallel_six_nist001/replay004
.venv/bin/python -m pytest -q tests/test_atomic_nist_audit.py tests/test_atomic_complete_norm_consumer.py tests/test_nist_atlas_export.py
```

The automatic probe has one CPU thread, 4 GiB address cap and 120 s wall cap;
it archives every actual source dependency and complete scientific input. No GPU
or new spectral solver action occurs. `handoff.json` registers both audit gates
and separately failed accuracy gates. Thirty-seven controls passed, including
mutated identity, bool species/J, mixed spaces, incomplete/multiple references,
unknown medium, missing observations/uncertainties, correlated interval
uncertainties, forbidden multipoles and lost signed norm interference.
`highest_level_reached=false` remains explicit.
