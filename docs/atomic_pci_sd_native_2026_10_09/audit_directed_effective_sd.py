"""Measure directed fixed-energy effective-operator symmetries without repair."""

import argparse
import hashlib
import itertools
import json
import shutil
from pathlib import Path

import numpy as np

OWNER = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case", default="sigma2_solved004")
parser.add_argument("--run-name", default="directed_effective_audit001")
args = parser.parse_args()
if not args.case.startswith("sigma2_solved") or not args.case[13:].isdigit():
    raise ValueError("explicit source-bound solved case required")
if (not args.run_name.startswith("directed_effective_audit")
        or not args.run_name[24:].isdigit()):
    raise ValueError("explicit fresh directed audit identifier required")
case = OWNER / args.case
out = OWNER / args.run_name
out.mkdir(exist_ok=False)
shutil.copy2(Path(__file__), out / Path(__file__).name)
contract = dict(
    hypothesis="Directed fixed-column-energy SD screening must not be treated as Hermitian CI input without measuring complete physical matrix identities.",
    fixed=f"Same source-bound C2+ nativeDC .37virtual-rotation fixed-energy case00, all6s-p radial columns/16magnetic states and120antisymmetric pairs; all Sigma1/Sigma2 outputs from {args.case}. No symmetrization or energy extrapolation.",
    source_contract=json.loads((case / "contract.json").read_text()),
    criterion_Ha=1e-11,
    distinguishing="Particle-swap covariance is a scalar two-body identity. Measure adjoint defects of Sigma1, irreducible pairSigma2 and combined frozen-energy valenceH; a nonzero defect is recorded, not averaged away.",
    scope="finite directed operator consumer audit, not physical energy/metric/experimental validation",
)
(out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
inputs = np.load(case / "inputs.npz")
vectors = np.load(case / "case00/complete_vectors.npz")
labels = tuple(tuple(int(i) for i in row) for row in inputs["labels"])
nc = inputs["core_s"].shape[1]
virtual = labels[nc:]
keys = tuple(dict.fromkeys(label[:2] for label in virtual))
sigma1 = np.zeros((len(virtual),) * 2)
files = [case / "inputs.npz", case / "case00/complete_vectors.npz"]
for key in keys:
    p = case / f"case00/column_{key[0]}_{key[1]}.npz"
    files.append(p)
    column = np.load(p)
    rowkeys = tuple(k for k in keys if k[1] == key[1])
    for v, label in enumerate(virtual):
        if label[:2] == key:
            for m, target in enumerate(virtual):
                if target[1:] == label[1:]:
                    sigma1[m, v] = column["sigma1"][rowkeys.index(target[:2])]
gamma = vectors["native"]
swap_defect = float(np.max(abs(gamma - gamma.transpose(1, 0, 3, 2))))
pairs = tuple(itertools.combinations(range(len(virtual)), 2))
sigma2 = np.asarray(
    [
        [
            0.5
            * (
                gamma[m, n, v, w]
                - gamma[m, n, w, v]
                - gamma[n, m, v, w]
                + gamma[n, m, w, v]
            )
            for v, w in pairs
        ]
        for m, n in pairs
    ]
)
g = inputs["direct"][nc:, nc:, nc:, nc:]
bare_two = np.asarray(
    [[g[m, n, v, w] - g[m, n, w, v] for v, w in pairs] for m, n in pairs]
)
h1 = inputs["fock"][nc:, nc:] + sigma1
embedded = np.asarray(
    [
        [
            h1[m, v] * (n == w)
            - h1[m, w] * (n == v)
            - h1[n, v] * (m == w)
            + h1[n, w] * (m == v)
            for v, w in pairs
        ]
        for m, n in pairs
    ]
)
h = embedded + bare_two + sigma2
record = dict(
    full_magnetic_states=len(virtual),
    full_pair_elements=sigma2.size,
    simultaneous_particle_swap_defect_Ha=swap_defect,
    sigma1_adjoint_defect_Ha=float(np.max(abs(sigma1 - sigma1.T))),
    sigma2_pair_adjoint_defect_Ha=float(np.max(abs(sigma2 - sigma2.T))),
    combined_pair_adjoint_defect_Ha=float(np.max(abs(h - h.T))),
    bare_pair_adjoint_defect_Ha=float(np.max(abs(bare_two - bare_two.T))),
    source_sha256={
        str(p.relative_to(OWNER)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    },
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
)
record["particle_covariance_status"] = (
    "pass" if swap_defect <= contract["criterion_Ha"] else "fail"
)
record["hermitian_consumer_status"] = (
    "pass"
    if max(
        record[k]
        for k in (
            "sigma1_adjoint_defect_Ha",
            "sigma2_pair_adjoint_defect_Ha",
            "combined_pair_adjoint_defect_Ha",
        )
    )
    <= contract["criterion_Ha"]
    else "unverified; measured directed adjoint defect prohibits implicit Hermitian consumption"
)
np.savez_compressed(
    out / "complete_matrices.npz",
    sigma1=sigma1,
    sigma2=sigma2,
    bare_two=bare_two,
    embedded=embedded,
    combined=h,
    pairs=np.asarray(pairs),
    virtual_labels=np.asarray(virtual),
)
(out / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")
print(
    json.dumps(
        {k: v for k, v in record.items() if k not in ("source_sha256", "script_sha256")}
    )
)
