"""Production frozen Gu field versus archived/rebuilt controls, no energy fitting."""

import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()
)
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "benchmarks")]
import bench_systematic_xray as b

from atomx.constants import BOHR_RADIUS_M
from atomx.methods.dirac_fock import DFSolver
from atomx.scattering.models import ContinuumModel, PotentialModel
from atomx.scattering.photoionization import subshell_photoionization

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "atomic_consistency_2026_10_02"
PRIOR = HERE.parent / "error_attribution_2026_10_02"
REF = HERE.parent / "systematic_xray_2026_10_02"
refs = json.loads((REF / "references.json").read_text())


def run(el, ns=180):
    z, shells = b.CASES[el]
    result = DFSolver.from_configuration(
        z,
        b.config(shells),
        exchange="gu",
        ns=ns,
        ks=9,
        r_max=80,
        nuclear="point",
        starting_potential="tfda",
    ).solve(max_iter=300, tol=1e-9, damping=0.5, mode="orbital")
    assert result.converged
    archive = json.loads((OLD / f"{el}-gu.json").read_text())
    base = ContinuumModel.fac_frozen_core(n_grid=16000, r_max=80)
    models = {
        "production": base,
        "production_shooting32000": replace(base, shooting_n_grid=32000),
        "production_nrad128000": replace(
            base, potential=replace(base.potential, n_radial=128000)
        ),
        "rebuilt_legacy": replace(
            base, potential=PotentialModel.static_hartree_exchange(exchange="fac")
        ),
    }
    variants = {}
    selected = [s.label for s in result.subshells if s.n <= 2]
    for name, model in models.items():
        pi = subshell_photoionization(
            result,
            refs[el]["photons_eV"],
            continuum_model=model,
            orbitals=selected,
            thresholds_eV=archive["thresholds_eV"]["relaxed"],
        )
        total = {sh: np.zeros(3) for sh in ("1s", "2s", "2p")}
        for sh in result.subshells:
            if sh.label in selected:
                total[f"{sh.n}{'spdf'[sh.l]}"] += (
                    pi.sigma_by_orbital_a0_sq[sh.label] * BOHR_RADIUS_M**2 / 1e-22
                )
        variants[name] = {sh: values.tolist() for sh, values in total.items()}
    for shell in ("1s", "2s", "2p"):
        np.testing.assert_array_equal(
            variants["production"][shell], variants["production_nrad128000"][shell]
        )
        if ns == 180:
            prior = json.loads((PRIOR / f"pi-{el}.json").read_text())["variants_Mb"][
                "native_16000"
            ][shell]
            np.testing.assert_allclose(
                variants["rebuilt_legacy"][shell], prior, rtol=1e-11
            )
    record = dict(
        element=el,
        ns=ns,
        control_thresholds="archived native Gu relaxed Delta-SCF",
        variants_Mb=variants,
    )
    (HERE / f"{el}-ns{ns}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(el, ns, "production panel complete", flush=True)
    return record


if __name__ == "__main__":
    records = [run(el) for el in b.CASES]
    records += [run(el, ns=360) for el in ("C", "Ne")]
    rows = []
    for record in records:
        el, ns = record["element"], record["ns"]
        for variant, values in record["variants_Mb"].items():
            for shell, points in values.items():
                for photon, sigma, ref in zip(
                    refs[el]["photons_eV"],
                    points,
                    refs[el]["fac_pi"][shell],
                    strict=True,
                ):
                    rows.append(
                        dict(
                            element=el,
                            ns=ns,
                            variant=variant,
                            shell=shell,
                            photon_eV=photon,
                            sigma_Mb=sigma,
                            archived_fac_Mb=ref,
                            ratio=sigma / ref,
                        )
                    )
    with (HERE / "pi_validation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
