# Public accepted-point resume: affected regression

The prospective [contract](contract.json) fixes eight existing test files and
their original hashes. The [runner](run.py) binds every actual dirty Atomx Python
source, adds the new numerical resume controls, uses one CPU and one BLAS/Numba
thread with a fresh generated compiler cache, and enforces a 300-second process
timeout. This test contract does not impose a memory cap.

[Verification](verification.json) passed: **115 tests in 95.18 seconds**,
comprising 75 unchanged affected checks and 40 new resume controls. The complete
source and all original/new test bytes remained unchanged during execution.
[Execution identity](execution_binding.json) and the original [pytest output](pytest.log)
retain the exact command, source map, fixture hashes and host admission.

The tests validate the numerical interface and unchanged physical consumer
paths. They do not certify the expanded Yb⁺ solution, full core-valence coverage,
experimental accuracy or a performance speedup. The actual
[531-CSF consumer](../public_mixed_resume_run/README.md) has a separate scientific
contract, source epoch, runtime caps and independent full-gradient endpoint replay.
