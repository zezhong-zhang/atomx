"""Compare the captured public Auger API, including its default settings.

Uses the same initial configurations as XATOM/FAC. The LS API takes one
j=l+1/2 radial representative per n-l shell; these are configuration rates,
not FAC resolved-level rates. Reference kinetic energies are explicit inputs.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from unittest.mock import patch

import compare as panel
import numpy as np

from atomx import Atom, Correlation
from atomx.scattering import autoionization as api

HERE = Path(__file__).resolve().parent


def label(shell):
    n, l = panel.shell_key(shell)
    return f"{n}{'spdf'[l]}{2 * l + 1}/2"


def load_previous_api():
    spec = importlib.util.spec_from_file_location(
        "previous_coarse_api", HERE / "baseline_coarse_api.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def same_radial_audit(result, a, b, reference):
    """Inject XATOM R^K into the actual production LS chain.

    The fake continuum tags each l4. Each cache insertion evaluates direct
    followed by exchange radials, so a two-call counter selects R12/R21.
    Both patched numerical functions are restored after each API call.
    """
    na, la = panel.shell_key(a)
    nb, lb = panel.shell_key(b)
    counts = {}

    def continuum(energy, kappa, potential, **kwargs):
        r = np.array([1e-8, float(kwargs["r_max"])])
        return r, np.full(2, float(kappa)), np.zeros(2), 0.0

    def radial(P1, Q1, P2, Q2, P3, Q3, P4, Q4, r, rank):
        l4 = int(round(-float(P4[0]) - 1))
        assert np.allclose(P4, -(l4 + 1))
        count = counts.get((l4, rank), 0)
        counts[l4, rank] = count + 1
        assert count < 2
        first, second = ((na, la), (nb, lb)) if count == 0 else ((nb, lb), (na, la))
        key = f"{rank},{first[0]}{first[1]},{second[0]}{second[1]},10,{l4}"
        return reference["radial"].get(key, 0.0)

    with (
        patch(
            "atomx.numerics.shooting.dirac_continuum.solve_dirac_continuum_shooting",
            continuum,
        ),
        patch("atomx.numerics.quadrature.grid_slater_rk", radial),
    ):
        out = api.auger_rate_csf_average(
            result,
            hole_label="1s1/2",
            source_label=label(a),
            source2_label=label(b),
            resonance_energy_eV=800.0,
        )
    target = reference["auger_channels"][f"{a}+{b}"]["rate_per_s"]
    ratio = out["Gamma_per_s"] / target
    if abs(ratio - 1) > 3e-5:
        raise AssertionError(f"Production LS same-radial audit {a}+{b}: {ratio}")
    return {
        "production_LS_rate_per_s": out["Gamma_per_s"],
        "xatom_rate_per_s": target,
        "ratio": ratio,
    }


def main():
    xatom = json.loads((HERE / "xatom.json").read_text())
    fac = json.loads((HERE / "fac.json").read_text())
    previous = load_previous_api()
    output = {}
    for element, case in panel.CASES.items():
        initial = dict(case["shells"])
        initial["1s"] = 1
        atom = Atom(
            Z=case["Z"],
            charge=1,
            config=[panel.config(initial)],
            correlation=Correlation(radial_optimization="fac_frozen"),
            corrections="none",
        )
        start = time.perf_counter()
        atom.solve()
        result = atom._result
        if getattr(result, "converged", True) is False:
            raise RuntimeError(f"{element}: public Atom solve did not converge")
        record = {
            "configuration": panel.config(initial),
            "radial_optimization": "fac_frozen",
            "exchange": atom.exchange,
            "ns": atom.ns,
            "ks": atom.ks,
            "requested_r_max": atom.r_max,
            "r_max": float(result.bsp_data["t"][-1]),
            "nuclear": atom.nuclear,
            "orbital_eV": {
                k: float(v) * panel.HARTREE_EV
                for k, v in result.orbital_energies.items()
            },
            "solve_seconds": time.perf_counter() - start,
            "occupations": {str(k): float(v) for k, v in result.occupations.items()},
            "basis_grid_points": int(result.bsp_data["gr"].size),
            "channels": {},
            "same_radial_production_LS_audit": {},
        }
        for channel, xrow in xatom[element]["auger"]["auger_channels"].items():
            a, b = channel.split("+")
            frow = fac[element]["auger_channels"][channel]
            fac_energy = (
                sum(r["DELTA_E_eV"] * r["rate_per_s"] for r in frow["levels"])
                / frow["rate_per_s"]
            )
            orbital_energy = (
                result.orbital_energies[label(a)]
                + result.orbital_energies[label(b)]
                - result.orbital_energies["1s1/2"]
            ) * panel.HARTREE_EV
            row = {
                "xatom_per_s": xrow["rate_per_s"],
                "fac_per_s": frow["rate_per_s"],
                "xatom_electron_eV": xrow["electron_eV"],
                "fac_mean_electron_eV": fac_energy,
                "atomx_orbital_sum_electron_eV": orbital_energy,
                "source_labels": [label(a), label(b)],
            }
            variants = [
                (
                    "previous_API_default",
                    previous.auger_rate_csf_average,
                    xrow["electron_eV"],
                    {},
                ),
                ("new_LS_default", api.auger_rate_csf_average, xrow["electron_eV"], {}),
                (
                    "new_LS_16000",
                    api.auger_rate_csf_average,
                    xrow["electron_eV"],
                    {"shooting_n_grid": 16000},
                ),
                (
                    "new_LS_FAC_energy_16000",
                    api.auger_rate_csf_average,
                    fac_energy,
                    {"shooting_n_grid": 16000},
                ),
                (
                    "new_LS_orbital_energy_16000",
                    api.auger_rate_csf_average,
                    orbital_energy,
                    {"shooting_n_grid": 16000},
                ),
            ]
            if element == "Ne":
                variants.append(
                    (
                        "new_JJ_16000",
                        api.auger_rate_csf_average,
                        xrow["electron_eV"],
                        {"shooting_n_grid": 16000, "angular_mode": "jj"},
                    )
                )
            for name, func, energy, options in variants:
                api._LS_TRACE = []
                start = time.perf_counter()
                rate = func(
                    result,
                    hole_label="1s1/2",
                    source_label=label(a),
                    source2_label=label(b),
                    resonance_energy_eV=energy,
                    **options,
                )["Gamma_per_s"]
                elapsed = time.perf_counter() - start
                trace = api._LS_TRACE
                api._LS_TRACE = None
                assert np.isfinite(rate) and rate >= 0
                row[name] = {
                    "rate_per_s": float(rate),
                    "seconds": elapsed,
                    "ratio_to_xatom": rate / xrow["rate_per_s"],
                    "ratio_to_fac": rate / frow["rate_per_s"],
                    "radial_trace": trace,
                }
            record["channels"][channel] = row
            record["same_radial_production_LS_audit"][channel] = same_radial_audit(
                result, a, b, xatom[element]["auger"]
            )
            print(
                element,
                channel,
                "default/XATOM",
                row["new_LS_default"]["ratio_to_xatom"],
                "fine/XATOM",
                row["new_LS_16000"]["ratio_to_xatom"],
                flush=True,
            )
        variant_names = [
            "previous_API_default",
            "new_LS_default",
            "new_LS_16000",
            "new_LS_FAC_energy_16000",
            "new_LS_orbital_energy_16000",
        ] + (["new_JJ_16000"] if element == "Ne" else [])
        record["totals"] = {
            name: sum(row[name]["rate_per_s"] for row in record["channels"].values())
            for name in variant_names
        }
        record["totals"].update(
            xatom_per_s=xatom[element]["auger"]["auger_total_per_s"],
            fac_per_s=fac[element]["auger_total_per_s"],
        )
        output[element] = record
        panel.dump(HERE / "public-auger.json", output)
        print(element, "total rates", record["totals"], flush=True)


if __name__ == "__main__":
    main()
