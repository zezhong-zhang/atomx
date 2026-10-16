# First independent native / tensor triples comparator

The first case is exactly the existing `tests/test_eom_attachment.py` seven-spinor,
three-electron **synthetic two-body toy**: seed 194025, frozen constant −2.3 Ha,
all signed h/F/ERI coordinates, occupied indices 0–2 and active indices 3–6.
Its labels omit magnetic partners. It is not a spherical atom, closed atomic
core, radial Dirac–Coulomb calculation, NIST comparison or Sahoo2025 EA result.
`SpinorIntegrals` supplies the same owned native integral boundary to both methods.

Patched EXP-T C1 reads explicit occupation flags and contains no even-N/Kramers
restriction. Odd N is therefore retained as a legitimate finite equation test;
it is not falsely advertised as a closed-shell atomic core. The separately
authorized Ne6+ 1s²2s² / eight-complete-s-spinor seam has not been launched.
The proposal chronology and this scope choice are retained in [contract.json](contract.json).

The complete core and attached spaces contain 35 determinants each. Native
CCSDT has 12 singles, 18 doubles and four connected triples; for N=3 this
matches the full-core FCI excitation rank. In N+1 the complete R≤3p2h pool has
34 rows; the omitted 4p3h row remains in every full-H diagnostic. Nonzero T3
action in N+1 is required. Four lowest projected EOM roots define the fixed
four-column model, with explicit projector conditioning and root-gap gates.

Native dense CCSDT/EOM and EXP-T tensor FSCC use the same full operator and
energy constant. EOM diagonalizes the complete 34-row projected similarity
operator; FSCC solves a valence-universal, intermediate-normalized four-column
wave operator with folded terms. Their matrices/waves/roots are compared only
after independently checking actual tensor core, retained Bloch equations and
normalization. General EOM/FSCC identity and full attached FCI equality are not
assumed. Omitted-rank and full physical residuals remain visible diagnostics.

The eligible external binary is the previously validated corrected public
`parallel001/public_backend_source-build/expt.x`, SHA `cf230c3c…`, with
`ATOMX_FOV_T2_001`, sector01 SHA `f73ee35a…`, and all 307 declared source hashes
matching. The older manually built candidate lacks the public capability
declaration and is not used. Isolated libgfortran and actual runtime libraries
are bound. Original core/valence T1/T2/T3 diagrams, input files, full H_eff,
signed wave operators and complete numerical arrays remain in the run directory.

A private byte-identical snapshot of current Atomx source prevents parallel
root edits from changing this experiment. The snapshot includes optional
state-action code, but this first run explicitly requests `dense`.
No shared source, backend, existing model, Fov element or threshold is changed.

Pre-admission: CPU 28, one numerical thread, 8 GiB address space, 110 s inner
and 120 s outer limits; one external launch has a 60 s timeout and 512 MiB
backend cap. The tensor estimate is 4,396 coordinates / 2,265,232 working bytes;
native generator and decoded tensor caps are separately declared. Load and
available memory are checked before numerical imports. First failure is retained;
no changed rank/core/model/threshold or automatic retry is allowed.

`run.py` defaults to source-only verification. Scientific execution uses the
repository `.venv`, serial thread environment, `taskset -c 28`, `timeout 120s`
and `--execute`. Fourteen existing capability/source-patch tests passed before
launch. Unpaired phase timings are descriptive and imply no speedup.
Documentation impact is confined to this record; root owns current guides/EARS.
