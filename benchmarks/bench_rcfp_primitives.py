#!/usr/bin/env python3
"""Report exact reduced-subshell primitive coefficients.

This benchmark is intentionally FAC-independent for now.  It records stable
values from AtomX's exact m-scheme backend for the RCFP primitives that will be
compared against FAC/Gaigalas tables as the pure-Python RCFP layer expands.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from atomx.angular.csf_recoupling import (  # noqa: E402
    RecoupledCSFState,
    chain_w_reduced,
    chain_wxw_scalar,
    discover_chain_two_body_scalar_coefficients,
    discover_double_excitation_coefficients_simple,
    discover_single_excitation_coefficients_simple,
    discover_two_body_scalar_coefficients_simple,
    make_chain_coupled_state,
)
from atomx.angular.rcfp import (  # noqa: E402
    reduced_axw_coeff,
    reduced_creation_coeff,
    reduced_pair_annihilation_coeff,
    reduced_pair_creation_coeff,
    reduced_subshell_states,
    reduced_w_coeff,
    reduced_wxa_coeff,
    reduced_wxw0_coeff,
)


def main() -> None:
    rows = []

    vac = reduced_subshell_states(1, 0)[0]
    one = reduced_subshell_states(1, 1)[0]
    closed = reduced_subshell_states(1, 2)[0]
    rows.append({
        "case": "j1/2 q0->q1 creation",
        "value": reduced_creation_coeff(one, vac),
    })
    rows.append({
        "case": "j1/2 q1->q2 creation",
        "value": reduced_creation_coeff(closed, one),
    })
    rows.append({
        "case": "j1/2 q1 W K0",
        "value": reduced_w_coeff(one, one, 0),
    })
    rows.append({
        "case": "j1/2 q1 WxW0 K0",
        "value": reduced_wxw0_coeff(one, one, 0),
    })
    rows.append({
        "case": "j1/2 q0->q1 AxW K0 Rj",
        "value": reduced_axw_coeff(one, vac, 0, 1),
    })
    rows.append({
        "case": "j1/2 q0->q1 WxA K0 Rj",
        "value": reduced_wxa_coeff(one, vac, 0, 1),
    })
    rows.append({
        "case": "j1/2 q0->q2 pair creation K0",
        "value": reduced_pair_creation_coeff(closed, vac, 0),
    })
    rows.append({
        "case": "j1/2 q2->q0 pair annihilation K0",
        "value": reduced_pair_annihilation_coeff(vac, closed, 0),
    })

    for state in reduced_subshell_states(3, 2):
        rows.append({
            "case": f"j3/2 q2 J{state.J2} W K0",
            "value": reduced_w_coeff(state, state, 0),
        })
        rows.append({
            "case": f"j3/2 q2 J{state.J2} WxW0 K0",
            "value": reduced_wxw0_coeff(state, state, 0),
        })

    csf_state = RecoupledCSFState((one, reduced_subshell_states(3, 1)[0]), total_J2=2)
    coeffs = discover_two_body_scalar_coefficients_simple(csf_state, csf_state)
    rows.append({
        "case": "simple CSF two-body scalar WxW0 pair 0-1",
        "value": coeffs[0].coefficient if coeffs else 0.0,
    })
    p0 = reduced_subshell_states(3, 0)[0]
    p1 = reduced_subshell_states(3, 1)[0]
    spectator = closed
    single_bra = RecoupledCSFState((vac, p1, spectator), total_J2=2)
    single_ket = RecoupledCSFState((one, p0, spectator), total_J2=2)
    single = discover_single_excitation_coefficients_simple(single_bra, single_ket)
    rows.append({
        "case": "simple CSF single excitation coeff 0->1 with spectator",
        "value": single[0].coefficient if single else 0.0,
    })
    p2 = reduced_subshell_states(3, 2)[0]
    double_bra = RecoupledCSFState((vac, p2, spectator), total_J2=0)
    double_ket = RecoupledCSFState((closed, p0, spectator), total_J2=0)
    double = discover_double_excitation_coefficients_simple(double_bra, double_ket)
    rows.append({
        "case": "simple CSF paired double excitation coeff 0->1 with spectator",
        "value": double[0].coefficient if double else 0.0,
    })
    mixed_bra = RecoupledCSFState((vac, p0, p1, p1, spectator), total_J2=0)
    mixed_ket = RecoupledCSFState((one, p1, p0, p0, spectator), total_J2=0)
    mixed = discover_double_excitation_coefficients_simple(mixed_bra, mixed_ket)
    rows.append({
        "case": "simple CSF mixed double excitation coeff",
        "value": mixed[0].coefficient if mixed else 0.0,
    })
    chain = make_chain_coupled_state((one, p1, one), cumulative_J2=(1, 2, 1))
    rows.append({
        "case": "chain CSF W K0 active middle with spectator recoupling",
        "value": chain_w_reduced(chain, chain, active=1, K2=0),
    })
    rows.append({
        "case": "chain CSF WxW0 active pair 0-1 with spectator recoupling",
        "value": chain_wxw_scalar(chain, chain, active_left=0, active_right=1, K2=0),
    })
    discovered_chain = discover_chain_two_body_scalar_coefficients(chain, chain)
    rows.append({
        "case": "chain CSF discovered WxW0 coefficient count",
        "value": float(len(discovered_chain)),
    })

    out = ROOT / "benchmarks" / "results_rcfp_primitives.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "value"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"case": row["case"], "value": f"{row['value']:.16e}"})

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
