"""Sparse independent multicore antisymmetry anchors without Hamiltonian fitting."""

import argparse
import hashlib
import json
import runpy
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

OWNER = Path(__file__).resolve().parent
ROOT = OWNER.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(OWNER)]

from monovalent_norm_control import norm_components  # noqa: E402

from atomx.methods.core_sd import plan_core_sd  # noqa: E402
from atomx.methods.dirac_fock import DFSolver, Subshell  # noqa: E402
from atomx.methods.monovalent_sd import FiniteMonovalentSD, _core_terms  # noqa: E402
from atomx.methods.reduced_monovalent_sd import plan_reduced_monovalent_sd  # noqa: E402
from atomx.numerics.bspline.coulomb_features import (  # noqa: E402
    PolynomialCellCoulombFeatures,
)
from atomx.numerics.fermion import apply_terms  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def square(state):
    return sum(abs(value) ** 2 for value in state.values())


def main(args):
    out = OWNER / args.run_name
    out.mkdir(exist_ok=False)

    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + "\n")

    save(
        "contract.json",
        dict(
            hypothesis="Full core hole/particle antisymmetry and valence Pauli/interference norm hold for different radial holes, same-kappa distinct radial holes and different kappa holes",
            space="abstract orthonormal complete multiplets,10 core electrons/10virtual magnetic spinors; no Hamiltonian or real-atom energy claim",
            keys=[[3, -1], [4, -1], [3, 1], [3, -2]],
            signs=[1, -1],
            seed=481219,
            gate=1e-11,
            stop="Any component/total/M/source gate failure rejects",
            memory_bytes=64 * 1024**2,
            no_magnetic_quartic_tensor=True,
        ),
    )
    files = [
        Path(__file__),
        OWNER / "monovalent_norm_control.py",
        ROOT / "src/atomx/methods/monovalent_sd.py",
        ROOT / "src/atomx/angular/sd_rho.py",
        ROOT / "src/atomx/numerics/fermion.py",
        ROOT / "tests/test_reduced_monovalent_sd.py",
    ]
    hashes = {}
    for path in files:
        relative = path.relative_to(ROOT)
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = digest(path)
    save("source_sha256.json", hashes)
    radial = DFSolver(Z=4, subshells=[Subshell(1, -1, 2)], ns=30, ks=7).solve()
    features = PolynomialCellCoulombFeatures(radial._bsp_data)
    occupied = ((1, -1), (2, -1), (2, 1), (2, -2))
    virtual = ((3, -1), (4, -1), (3, 1), (3, -2))
    cp = plan_core_sd(occupied, virtual, features.identity)
    vp = plan_reduced_monovalent_sd(cp, virtual)
    labels = tuple(
        (nr, k, m)
        for nr, k in occupied + virtual
        for m in range(-2 * abs(k) + 1, 2 * abs(k), 2)
    )
    core = SimpleNamespace(
        plan=cp, integrals=SimpleNamespace(labels=labels, noccupied=10)
    )
    reduced = SimpleNamespace(core=core, plan=vp)
    helper = runpy.run_path(str(ROOT / "tests/test_reduced_monovalent_sd.py"))
    rng = np.random.default_rng(481219)
    crho = rng.normal(size=cp["raw_amplitudes"]) * 0.013
    ct = _core_terms(core, crho)
    cs = tuple(t for t in ct if len(t[1]) == 2)
    cd = tuple(t for t in ct if len(t[1]) == 4)
    total_cd = square(apply_terms(cd, {(1 << 10) - 1: 1.0}))
    cases = []
    checks = 0
    maximum = 0.0
    for key in virtual:
        rho = rng.normal(size=vp["amplitudes"][key]) * 0.017
        covariance = {1: [], -1: []}
        for m in range(-2 * abs(key[1]) + 1, 2 * abs(key[1]), 2):
            v = labels.index((*key, m))
            finite = SimpleNamespace(
                core=core,
                integrals=core.integrals,
                valence_index=v,
                reference_mask=((1 << 10) - 1) | (1 << v),
            )
            finite.excitations = FiniteMonovalentSD._excitations(finite)
            state = {finite.reference_mask: 1.0}
            for sign in (1, -1):
                physical = helper["expanded"](reduced, key, sign * rho, finite)
                vt = FiniteMonovalentSD._valence_terms(finite, physical)
                ts, td = apply_terms(cs, state), apply_terms(cd, state)
                vs = apply_terms(tuple(t for t in vt if len(t[1]) == 2), state)
                vd = apply_terms(tuple(t for t in vt if len(t[1]) == 4), state)
                full = apply_terms(ct + vt, state)
                full[finite.reference_mask] = full.get(finite.reference_mask, 0.0) + 1
                expected = dict(
                    valence_singles=square(vs),
                    valence_doubles=square(vd),
                    core_singles_unblocked=square(ts),
                    core_singles_valence_doubles_interference=2
                    * sum(value * vd.get(mask, 0.0) for mask, value in ts.items()),
                    core_doubles_total=total_cd,
                    core_doubles_pauli_blocked=total_cd - square(td),
                    core_doubles_unblocked=square(td),
                    norm_squared=square(full),
                )
                actual = norm_components(cp, vp, crho, sign * rho, key, twoM=m)
                errors = {k: abs(actual[k] - value) for k, value in expected.items()}
                if max(errors.values()) > 1e-11:
                    save(
                        "failure.json",
                        dict(
                            key=key,
                            twoM=m,
                            sign=sign,
                            expected=expected,
                            actual=actual,
                            errors=errors,
                        ),
                    )
                    raise AssertionError("multicore physical norm gate failed")
                cases.append(
                    dict(
                        key=key,
                        twoM=m,
                        sign=sign,
                        errors=errors,
                        actual=actual,
                        expected=expected,
                    )
                )
                covariance[sign].append(actual["norm_squared"])
                checks += len(errors)
                maximum = max(maximum, max(errors.values()))
        for values in covariance.values():
            if max(values) - min(values) > 1e-11:
                raise AssertionError("complete M norm covariance failed")
            checks += 1
    if any(digest(ROOT / name) != sha for name, sha in hashes.items()):
        raise AssertionError("multicore source binding changed")
    save(
        "result.json",
        dict(
            status="pass",
            checks=checks,
            cases=len(cases),
            maximum_error=maximum,
            rows=cases,
            source_match=True,
            core_amplitudes=cp["raw_amplitudes"],
            valence_amplitudes={str(k): v for k, v in vp["amplitudes"].items()},
            scope="abstract finite orthonormal fermion wave norm; different and same radial/core-kappa holes; no large-space/performance/property claim",
        ),
    )
    print(
        json.dumps(
            dict(status="pass", checks=checks, cases=len(cases), maximum_error=maximum)
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="monovalent_norm_multicore003")
    main(parser.parse_args())
