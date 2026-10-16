# Guard002 source review finding

The source review missed a fixture convention mismatch: the public functional stores `L(raw_h) = tril(raw_h) + tril(raw_h, -1).T`, while guard002 compared it directly with archived raw builder output. Its initial draft readiness conclusion is withdrawn. The actual invocation failed before the H producer: three one-body assemblies completed, zero H/CI/gradient calls.

[Final source and actual-evidence finding](final_finding.json) binds the frozen consumer and failed root invocation. This review added zero physical calls and did not modify existing source or evidence. Guard002 remains unqualified. A successor must compare logical matrices with logical matrices, retain a separate exact raw-h/S control, and pass its own original scientific gates.
