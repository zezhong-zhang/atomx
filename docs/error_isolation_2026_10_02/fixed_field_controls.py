"""Separate basis, SCF field and continuum reconstruction; diagnostic only."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()
)
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "benchmarks")]
import bench_systematic_xray as b

import atomx.scattering.radiative_recombination as rr
from atomx.constants import BOHR_RADIUS_M
from atomx.methods.dirac_fock import DFSolver
from atomx.numerics.bspline.local_potential import (
    local_central_potential_from_radial_density,
    radial_density_from_orbitals,
    solve_orbitals_in_local_potential,
)
from atomx.potentials.static_field import static_hartree_potential
from atomx.scattering.models import ContinuumModel
from atomx.scattering.photoionization import subshell_photoionization

HERE = Path(__file__).resolve().parent
REF = HERE.parent / "systematic_xray_2026_10_02"
OLD = HERE.parent / "atomic_consistency_2026_10_02"
refs = json.loads((REF / "references.json").read_text())


def solver_for(el, ns):
    z, shells = b.CASES[el]
    return DFSolver.from_configuration(
        z,
        b.config(shells),
        exchange="gu",
        ns=ns,
        ks=9,
        r_max=80,
        nuclear="point",
        starting_potential="tfda",
    )


def native_field(result, interpolation="linear"):
    orbital = result.orbitals_by_subshell()
    density, number = radial_density_from_orbitals(
        {k: pq[:2] for k, pq in orbital.items()},
        result._bsp_data,
        {k: pq[2] for k, pq in orbital.items()},
    )
    data = result._bsp_data
    parts = local_central_potential_from_radial_density(
        Z=result.Z,
        rho_rad=density,
        gr=data["gr"],
        grw=data["grw"],
        n_total=number,
        exchange="fac",
    )
    radii = data["gr"].ravel()
    screening = (data["gr"] * (parts["central"] + result.Z / data["gr"])).ravel()
    # Interpolate screening, retaining the analytic point nuclear singularity.
    spline = PchipInterpolator(radii, screening, extrapolate=False)

    def field(r):
        r = np.asarray(r)
        if interpolation == "linear":
            charge = np.interp(r, radii, screening, left=0.0, right=number - 1)
        else:
            charge = spline(np.clip(r, radii[0], radii[-1]))
            charge = np.where(r < radii[0], 0.0, charge)
            charge = np.where(r > radii[-1], number - 1, charge)
        return (-result.Z + charge) / r

    return field


def pi(el, result, field):
    archive = json.loads((OLD / f"{el}-gu.json").read_text())
    selected = [sh.label for sh in result.subshells if sh.n <= 2]
    original = rr._rr_potential
    try:
        rr._rr_potential = lambda *args: (field, 1.0)
        values = subshell_photoionization(
            result,
            refs[el]["photons_eV"],
            orbitals=selected,
            thresholds_eV=archive["thresholds_eV"]["relaxed"],
            continuum_model=ContinuumModel.fac_frozen_core(n_grid=16000, r_max=80),
        )
    finally:
        rr._rr_potential = original
    total = {shell: np.zeros(3) for shell in ("1s", "2s", "2p")}
    for sh in result.subshells:
        if sh.label in selected:
            total[f"{sh.n}{'spdf'[sh.l]}"] += (
                values.sigma_by_orbital_a0_sq[sh.label] * BOHR_RADIUS_M**2 / 1e-22
            )
    return {shell: val.tolist() for shell, val in total.items()}


def fixed_bound(el, reference, field, ns):
    data = solver_for(el, ns).bsp_data
    radii = data["gr"].ravel()
    orbital, data, energies = solve_orbitals_in_local_potential(
        Z=reference.Z,
        potential_r=radii,
        potential_v=field(radii),
        orbital_keys=reference.orbitals_by_subshell(),
        ns=ns,
        ks=9,
        bsp_data=data,
    )
    result = copy.copy(reference)
    result._bsp_data = data
    result.orbitals_raw = {}
    result.orbital_energies = {}
    for sh in reference.subshells:
        pq = orbital[(sh.n, sh.kappa)]
        result.orbitals_raw.setdefault(sh.kappa, []).append((*pq, sh.occupation))
        result.orbital_energies[sh.label] = energies[(sh.n, sh.kappa)]
    return result


def run(el):
    output = {"element": el, "control_only": True, "variants_Mb": {}, "energies_Ha": {}}
    variants = output["variants_Mb"]
    reference = solver_for(el, 180).solve(
        max_iter=300, tol=1e-9, damping=0.5, mode="orbital"
    )
    assert reference.converged
    base_field = native_field(reference)
    variants["scf180_native_field_linear"] = pi(el, reference, base_field)
    variants["scf180_native_field_pchip"] = pi(
        el, reference, native_field(reference, "pchip")
    )
    for n in (4000, 8000, 16000, 32000, 64000, 128000):
        rebuilt = static_hartree_potential(
            reference, n_radial=n, include_exchange=True, exchange="fac", r_max=80
        )
        variants[f"scf180_rebuilt_field{n}"] = pi(el, reference, rebuilt)
    for n in (4000, 32000, 128000):
        rebuilt = static_hartree_potential(
            reference,
            n_radial=n,
            include_exchange=True,
            exchange="fac",
            r_max=80,
            fac_self_interaction_correction=False,
        )
        variants[f"scf180_no_sic_field{n}"] = pi(el, reference, rebuilt)
    for ns in (180, 260, 360):
        result = fixed_bound(el, reference, base_field, ns)
        variants[f"fixed_field180_bound{ns}"] = pi(el, result, base_field)
        output["energies_Ha"][f"fixed_field180_bound{ns}"] = result.orbital_energies
    output["energies_Ha"]["scf180"] = reference.orbital_energies
    pchip = native_field(reference, "pchip")
    for ns in (260, 360):
        result = fixed_bound(el, reference, pchip, ns)
        variants[f"fixed_pchip180_bound{ns}"] = pi(el, result, pchip)
    result = solver_for(el, 360).solve(
        max_iter=300, tol=1e-9, damping=0.5, mode="orbital"
    )
    assert result.converged
    output["energies_Ha"]["scf360"] = result.orbital_energies
    field360 = native_field(result)
    variants["scf360_native_field_linear"] = pi(el, result, field360)
    variants["scf360_rebuilt_field4000"] = pi(
        el, result, static_hartree_potential(result, include_exchange=True, r_max=80)
    )
    # Holding the original bound state separates SCF potential drift.
    variants["scf180_with_field360"] = pi(el, reference, field360)
    radius = np.geomspace(1e-7, 80, 16000)
    np.savez_compressed(
        HERE / f"{el}-fields.npz",
        radius=radius,
        scf180=base_field(radius),
        scf360=field360(radius),
    )
    (HERE / f"{el}-fixed-field.json").write_text(json.dumps(output, indent=2) + "\n")
    print(el, "complete", flush=True)


if __name__ == "__main__":
    for element in sys.argv[1:] or ("C", "Ne"):
        run(element)
