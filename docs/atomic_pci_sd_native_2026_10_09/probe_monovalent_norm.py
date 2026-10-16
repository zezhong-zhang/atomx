"""Independent complete finite fermion control of private reduced physical norm."""

import argparse
import hashlib
import json
import os
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

from atomx.methods.monovalent_sd import FiniteMonovalentSD, _core_terms  # noqa: E402
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
            hypothesis="Complete radial Gram norm equals explicit independent fermionic (1+Tcore+Sv)|Phiv> for all M and signed interference, including valence Pauli blocking of core doubles",
            variants=["original", "streamed", "rotated", "occupied_rotated"],
            physical_space="complete original12spinor native DC Be+; four separately matched full orbital settings",
            gate=1e-11,
            signs=[1, -1],
            random_seed=819140,
            memory_bytes=64 * 1024**2,
            stop="Any individual component, total norm, M covariance, source or resource gate failure rejects; no changed tolerance or packing",
            scope="Private finite physical norm control; no property, full-SD eigenstate, performance or highest-level certification",
            load=os.getloadavg(),
        ),
    )
    files = (
        sorted((ROOT / "src/atomx").rglob("*.py"))
        + sorted((ROOT / "src/atomx").rglob("*.f90"))
        + sorted((ROOT / "tools").glob("build_pci_*.py"))
        + [
            ROOT / "tests/test_reduced_monovalent_sd.py",
            ROOT / "tests/test_monovalent_sd.py",
            Path(__file__),
            OWNER / "monovalent_norm_control.py",
        ]
    )
    hashes = {}
    for path in files:
        relative = path.relative_to(ROOT)
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = digest(path)
    save("source_sha256.json", hashes)
    helper = runpy.run_path(str(ROOT / "tests/test_reduced_monovalent_sd.py"))
    rows, arrays, maxima = [], {}, {}
    counter = 0

    class Factory:
        def mktemp(self, name):
            directory = out / name
            directory.mkdir()
            return directory

    for variant in ("original", "streamed", "rotated", "occupied_rotated"):
        core, core_result, reduced = helper["reduced_core"].__wrapped__(
            Factory(), SimpleNamespace(param=variant)
        )
        rng = np.random.default_rng(819140)
        crho = rng.normal(size=len(core_result.amplitudes)) * 0.013
        ct = _core_terms(core, crho)
        cs = tuple(t for t in ct if len(t[1]) == 2)
        cd = tuple(t for t in ct if len(t[1]) == 4)
        core_double = square(
            apply_terms(cd, {(1 << core.integrals.noccupied) - 1: 1.0})
        )
        for key in ((2, -1), (3, -1), (2, 1), (2, -2)):
            rho = rng.normal(size=reduced.plan["amplitudes"][key]) * 0.017
            arrays[f"{variant}_{key[0]}_{key[1]}_core"] = crho
            arrays[f"{variant}_{key[0]}_{key[1]}_valence"] = rho
            magnetic = []
            for m in range(-2 * abs(key[1]) + 1, 2 * abs(key[1]), 2):
                v = core.integrals.labels.index((*key, m))
                finite = FiniteMonovalentSD(core, core_result, v, work_cap=200_000_000)
                for sign in (1, -1):
                    packed = sign * rho
                    physical = helper["expanded"](reduced, key, packed, finite)
                    vt = finite._valence_terms(physical)
                    state = {finite.reference_mask: 1.0}
                    ts = apply_terms(cs, state)
                    td = apply_terms(cd, state)
                    vs = apply_terms(tuple(t for t in vt if len(t[1]) == 2), state)
                    vd = apply_terms(tuple(t for t in vt if len(t[1]) == 4), state)
                    full = apply_terms(ct + vt, state)
                    full[finite.reference_mask] = (
                        full.get(finite.reference_mask, 0.0) + 1.0
                    )
                    expected = dict(
                        valence_singles=square(vs),
                        valence_doubles=square(vd),
                        core_singles_unblocked=square(ts),
                        core_singles_valence_doubles_interference=2
                        * sum(value * vd.get(mask, 0.0) for mask, value in ts.items()),
                        core_doubles_total=core_double,
                        core_doubles_pauli_blocked=core_double - square(td),
                        core_doubles_unblocked=square(td),
                        norm_squared=square(full),
                    )
                    actual = norm_components(
                        core.plan, reduced.plan, crho, packed, key, twoM=m
                    )
                    errors = {
                        name: abs(actual[name] - value)
                        for name, value in expected.items()
                    }
                    for name, error in errors.items():
                        maxima[name] = max(maxima.get(name, 0.0), error)
                        if error > 1e-11:
                            save(
                                "failure.json",
                                dict(
                                    variant=variant,
                                    key=key,
                                    twoM=m,
                                    sign=sign,
                                    actual=actual,
                                    expected=expected,
                                    errors=errors,
                                ),
                            )
                            raise AssertionError(
                                "private full physical norm/component gate failed"
                            )
                        counter += 1
                    rows.append(
                        dict(
                            variant=variant,
                            key=key,
                            twoM=m,
                            sign=sign,
                            actual=actual,
                            expected=expected,
                            errors=errors,
                        )
                    )
                    magnetic.append((sign, actual["norm_squared"]))
            for sign in (1, -1):
                values = [value for s, value in magnetic if s == sign]
                spread = max(values) - min(values)
                if spread > 1e-11:
                    raise AssertionError(
                        "physical scalar norm differs across complete M multiplet"
                    )
                maxima["complete_M_covariance"] = max(
                    maxima.get("complete_M_covariance", 0.0), spread
                )
                counter += 1
    mismatch = [name for name, value in hashes.items() if digest(ROOT / name) != value]
    if mismatch:
        raise AssertionError("physical norm control source changed")
    np.savez(out / "inputs.npz", **arrays)
    save(
        "result.json",
        dict(
            status="pass",
            checks=counter,
            cases=len(rows),
            maxima=maxima,
            rows=rows,
            source_match=True,
            input_sha256=digest(out / "inputs.npz"),
            scope="independent finite real physical linearized SD wave norm including core spectators/interference/Pauli; no production adoption, property, physical convergence or timing claim",
        ),
    )
    print(
        json.dumps(dict(status="pass", checks=counter, cases=len(rows), maxima=maxima))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="monovalent_norm001")
    main(parser.parse_args())
