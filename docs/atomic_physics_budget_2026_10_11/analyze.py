"""Recompute physical diagnostic scales from immutable complete-state evidence.

No atomic solve, fitted correction, tail bound or additive error budget is made.
"""

import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def read_inputs(contract):
    values = {}
    for name, item in contract["inputs"].items():
        raw = (REPO / item["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError(f"Changed scientific input: {name}")
        values[name] = json.loads(raw)
    return values


def main():
    contract = json.loads((HERE / "contract.json").read_text())
    data = read_inputs(contract)
    facts = json.loads((HERE / "reference_facts.json").read_text())
    current = data["current_five_states"]
    ha = current["constants"]["HARTREE_CM1"]
    si = facts["SI"]
    ev_cm = si["e_C"] / (100 * si["h_J_s"] * si["c_m_s"])
    ionization = facts["NIST_ionization"]["value_eV"] * ev_cm
    ground = current["rows"][0]
    if ground["key"] != [6, -1] or len(current["rows"]) != 5:
        raise ValueError("Complete same-ground five-state input is required")
    rows, closure = [], []
    ground_error = ground["attachment_energy_Ha"] * ha + ionization
    for row in current["rows"]:
        absolute = row["attachment_energy_Ha"] * ha
        absolute_error = absolute - (row["experimental_cm1"] - ionization)
        defect = absolute_error - ground_error
        closure.append(abs(defect - row["error_cm1"]))
        df = (row["orbital_energy_Ha"] - ground["orbital_energy_Ha"]) * ha
        correlation = (
            row["correlation_energy_Ha"] - ground["correlation_energy_Ha"]
        ) * ha
        closure.append(abs(df + correlation - row["computed_cm1"]))
        rows.append(
            dict(
                key=row["key"],
                twoJ=row["twoJ"],
                configuration=row["configuration"],
                experiment_cm1=row["experimental_cm1"],
                theory_cm1=row["computed_cm1"],
                defect_cm1=row["error_cm1"],
                defect_Ha=row["error_cm1"] / ha,
                DF_transition_cm1=df,
                current_SD_differential_correlation_cm1=correlation,
                required_DF_to_experiment_difference_cm1=row["experimental_cm1"] - df,
                attachment_cm1=absolute,
                absolute_attachment_defect_cm1=absolute_error,
                original_1percent_pass=row["gate_pass"],
            )
        )
    if max(closure) > 1e-8:
        raise ValueError("Current complete-state arithmetic does not close")
    keyed = {tuple(row["key"]): row for row in rows}
    moments = []
    for name, keys in (("5d", [(5, 2), (5, -3)]), ("6p", [(6, 1), (6, -2)])):
        pair = [keyed[k] for k in keys]
        weights = [r["twoJ"] + 1 for r in pair]
        centroid = sum(
            w * r["defect_cm1"] for w, r in zip(weights, pair, strict=True)
        ) / sum(weights)
        splitting = pair[1]["defect_cm1"] - pair[0]["defect_cm1"]
        saved = next(x for x in current["splittings"] if x["label"] == name)
        if abs(splitting - saved["error_cm1"]) > 1e-8:
            raise ValueError("Full doublet splitting identity failed")
        moments.append(
            dict(
                term=name,
                weights=weights,
                centroid_defect_cm1=centroid,
                splitting_defect_cm1=splitting,
            )
        )
    controls = []
    historical_closure = []
    for item in data["historical_cc_controls"]["matched_controls"]:
        shifts = item["total_state_shift_Ha"]
        for state, stored in item["transition_shift_cm"].items():
            historical_closure.append(
                abs((shifts[state] - shifts["6s1/2"]) * ha - stored)
            )
        nonzero = [
            abs(v) for k, v in item["transition_shift_cm"].items() if k != "6s1/2"
        ]
        controls.append(
            dict(
                control=item["control"],
                complete_five_state=True,
                signed_transition_shift_cm1=item["transition_shift_cm"],
                absolute_range_cm1=[min(nonzero), max(nonzero)],
                scope="Historical bounded nonlinear CC/DC model; not a current LCCSD correction or tail bound.",
            )
        )
    if max(historical_closure) > 1e-5:
        raise ValueError("Historical complete five-state differences do not close")
    dynamic = [r for r in data["frozen_dynamic_control"]["rows"] if r["key"] != [6, -1]]
    literature = data["literature_tables"]
    literature_replay_error = []
    components = literature["safronova2009_printed_attachment_components_cm1"]
    for name, relative in literature["safronova2009_relative_components_cm1"].items():
        for term, value in relative.items():
            literature_replay_error.append(
                abs(components[name][term] - components["6s_1/2"][term] - value)
            )
    if max(literature_replay_error):
        raise ValueError("Primary-table upper-minus-ground arithmetic failed")
    nms = [
        dict(
            key=r["key"],
            estimated_shift_cm1=-r["experiment_cm1"]
            * si["m_e_kg"]
            / (facts["mass_proxy"]["A"] * si["u_kg"]),
        )
        for r in rows[1:]
    ]
    illustrative = []
    gap = facts["two_level_proxy"]["assumed_bare_gap_cm1"]
    for v in facts["two_level_proxy"]["assumed_couplings_cm1"]:
        illustrative.append(
            dict(
                assumed_gap_cm1=gap,
                assumed_V_cm1=v,
                repulsion_magnitude_cm1=(math.hypot(gap, 2 * v) - gap) / 2,
                small_component_weight=(1 - gap / math.hypot(gap, 2 * v)) / 2,
            )
        )
    radial_growth = []
    for task in data["band_preflight"]["tasks"]:
        lmax, emax = task["lmax"], task["energy_max_Ha"]
        count = task["serial_cpu_resource"]["virtual_radials"]
        radial_growth.append(
            dict(
                lmax=lmax,
                Emax_Ha=emax,
                radial_virtual_count=count,
                count_ratio=count / 414,
                virtual_pair_count_proxy_ratio=(count / 414) ** 2,
                historical_host_reservation_bytes=task["serial_cpu_resource"][
                    "host_admission_reservation_bytes"
                ],
                historical_admission=task["admission"],
                scope="Source-bound run004 complete-band counts and historical admission; pair proxy is not exact angular work, memory, time or speedup. Later bounded norm repair does not certify full production admission.",
            )
        )
    result = dict(
        status="complete saved-evidence physics-scale analysis; no new atomic roots",
        highest_level_reached=False,
        new_atomic_root_solves=0,
        fitted_corrections=False,
        total_physical_error_bound_cm1=None,
        inputs=contract["inputs"],
        reference_facts_sha256=hashlib.sha256(
            (HERE / "reference_facts.json").read_bytes()
        ).hexdigest(),
        ionization_anchor_cm1=ionization,
        threshold_listed_uncertainty_cm1=facts["NIST_ionization"][
            "listed_uncertainty_eV"
        ]
        * ev_cm,
        current_five_states=rows,
        doublet_diagnostics=moments,
        current_state_assignment_scope=current["scope"],
        numerical_residual_warning="Retained equation residual is not an eigenvalue or physical-space error bound.",
        historical_model_controls=controls,
        frozen_CI_Breit_controls=[
            dict(
                key=r["key"],
                static_minus_DC_cm1=r["static_minus_DC_cm1"],
                dynamic_minus_static_cm1=r["dynamic_minus_static_cm1"],
            )
            for r in dynamic
        ],
        primary_literature_transition_components_cm1=literature[
            "safronova2009_relative_components_cm1"
        ],
        literature_SD_minus_MBPT_cm1=literature[
            "porsev2012_SD_minus_MBPT_excitation_cm1"
        ],
        literature_warning="Current LCCSD already resums higher-order Coulomb. Published E3/SD-minus-MBPT are diagnostic scales, not absent additive corrections.",
        historical_scalar_tail=data["historical_scalar_tail"],
        normal_mass_shift_proxy=nms,
        two_level_hypothetical=illustrative,
        cost_proxies=radial_growth,
        selected_next="same-model complete five-state g-band versus energy-tail differential screen, then select one admitted full LCCSD increment from measured response and resource cost",
        selection_reason="Current D centroid defect873.8cm^-1 and opposite D/P splitting defects establish differential model error. Correlation is a leading unverified hypothesis given historical scales and current unclosed l<=3/E<5Ha space; omitted Breit and other responses remain separate candidates.",
        next_execution_status="planned, unlaunched; not a registered production executor",
        verification=dict(
            current_closure_max_cm1=max(closure),
            historical_closure_max_cm1=max(historical_closure),
            primary_table_arithmetic_max_cm1=max(literature_replay_error),
            complete_states=5,
            complete_excitations=4,
            complete_doublets=2,
            original_experimental_failures_preserved=True,
        ),
        card_summary=(
            f"物理优先：当前Yb完整五态/DC/ℓ≤3/<5Ha，D重心偏差{moments[0]['centroid_defect_cm1']:+.1f}cm⁻¹、P重心{moments[1]['centroid_defect_cm1']:+.1f}；"
            f"共同ASD阈值下基态偏浅{ground_error:.1f}cm⁻¹、D上态偏浅1451–1579。"
            "先判别g分波与5→10Ha对完整上下态的相关响应，再补同模型非线性/三激发及Coulomb–Breit交叉；"
            "冻结CI动态−静态仅0.4–3.9cm⁻¹，非当前SD修正或上界。文献和历史量级不相加。"
            "新方向尚未启动生产求根；每次卡片更新量级账与实际增量。"
        ),
    )
    read_inputs(contract)
    (HERE / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result["verification"], ensure_ascii=False))


if __name__ == "__main__":
    main()
