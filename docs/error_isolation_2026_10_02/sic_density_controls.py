"""Hold density coefficients fixed; isolate linear sampling before SIC fitting."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()
)
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "benchmarks")]
import fixed_field_controls as f

import atomx.numerics.bspline.local_potential as local
from atomx.numerics.bspline.evaluate import eval_P, eval_Q
from atomx.numerics.yk import compute_yk
from atomx.physics.local_exchange import fac_gu_exchange
from atomx.potentials.static_field import static_hartree_potential


def capture_sic(call):
    """Observe actual port locals, without duplicating the fitting algorithm."""
    original = sys.gettrace()
    record = {}

    def trace(frame, event, _arg):
        if (
            frame.f_code is local._fac_csi_screening_correction.__code__
            and event == "return"
        ):
            for key in ("j", "jm", "r0", "ur", "a", "b", "count"):
                if key in frame.f_locals:
                    value = frame.f_locals[key]
                    record[key] = (
                        value.item() if isinstance(value, np.generic) else value
                    )
        return trace

    try:
        sys.settrace(trace)
        result = call()
    finally:
        sys.settrace(original)
    assert {"r0", "ur", "a", "b"} <= record.keys()
    return result, record


def exact_density_field(result, n, smooth_hartree=False):
    data = result._bsp_data
    gr = data["gr"]
    orbitals = result.orbitals_by_subshell()
    density, number = local.radial_density_from_orbitals(
        {k: pq[:2] for k, pq in orbitals.items()},
        data,
        {k: pq[2] for k, pq in orbitals.items()},
    )
    direct = compute_yk(density, gr, data["grw"], data["nv"], gr.shape[1], 0).ravel()
    radius = gr.ravel()
    rh = np.geomspace(min(radius[0], 1e-8 / result.Z), 80, n)
    # Same Hartree interpolation and final interpolation as production.
    hartree_charge = np.interp(rh, radius, radius * direct, right=number)
    if smooth_hartree:
        interpolator = PchipInterpolator(radius, radius * direct)
        hartree_charge = interpolator(np.clip(rh, radius[0], radius[-1]))
        hartree_charge = np.where(rh > radius[-1], number, hartree_charge)
    hartree = np.where(rh < radius[0], direct[0], hartree_charge / rh)
    # Only change: evaluate original B-spline density instead of np.interp.
    rho = np.zeros_like(rh)
    for p, q, occupation in orbitals.values():
        rho += occupation * (eval_P(p, data, rh) ** 2 + eval_Q(q, data, rh) ** 2)
    exchange = fac_gu_exchange(
        rho / (4 * np.pi * rh**2), rh, Z=result.Z, N_total=number
    )
    charge = local._fac_csi_screening_correction(
        rh, rh * (hartree + exchange), n_total=number
    )
    charge = np.minimum(charge, number - 1)

    def field(r):
        r = np.asarray(r)
        return (-result.Z + np.interp(r, rh, charge, right=number - 1)) / r

    return field


def run(el):
    result = f.solver_for(el, 180).solve(
        max_iter=300, tol=1e-9, damping=0.5, mode="orbital"
    )
    assert result.converged
    output = {
        "element": el,
        "control_only": True,
        "variants_Mb": {},
        "sic_parameters": {},
    }
    native, record = capture_sic(lambda: f.native_field(result))
    output["sic_parameters"]["bound_native_gauss"] = record
    for n in (4000, 8000, 16000, 32000, 64000, 128000):
        for kind in ("linear_density", "exact_density", "smooth_hartree"):
            if kind == "linear_density":

                def make(n=n):
                    return static_hartree_potential(
                        result,
                        n_radial=n,
                        include_exchange=True,
                        exchange="fac",
                        r_max=80,
                    )

            else:

                def make(n=n, kind=kind):
                    return exact_density_field(
                        result, n, smooth_hartree=(kind == "smooth_hartree")
                    )

            field, record = capture_sic(make)
            name = f"{kind}_{n}"
            output["sic_parameters"][name] = record
            output["variants_Mb"][name] = f.pi(el, result, field)
    (f.HERE / f"{el}-sic-density.json").write_text(json.dumps(output, indent=2) + "\n")
    print(el, "SIC density controls complete", flush=True)


if __name__ == "__main__":
    for element in sys.argv[1:] or ("C", "Ne"):
        run(element)
