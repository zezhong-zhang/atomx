"""Four matched physical-space cells; reporting arithmetic, no new solving."""

import json

from prepare_virtuals import HERE, OLD, ROOT, save, sha

from atomx.constants import HARTREE_CM1


def main():
    paths = {
        "frozen4f_sd": HERE.parent / "native_tensor_ccsdt001/actual001/result.json",
        "active4f_sd": OLD / "actual003/result.json",
        "frozen4f_sdp": HERE / "corner001/result.json",
        "active4f_sdp": HERE / "actual001/result.json",
    }
    rows = {key: json.loads(path.read_text()) for key, path in paths.items()}
    assert rows["frozen4f_sdp"]["status"] == "PASS_FROZEN4F_COMPLETE_6P_CORNER"
    assert rows["active4f_sdp"]["status"] == "PASS_COMPLETE_6P_NATIVE_CCSDT_R3"
    cells = {}
    for key, row in rows.items():
        first = key == "frozen4f_sd"
        core = (
            row["cores"]["CCSDT"]["energy_Ha"]
            if first
            else row["core"]["energy_relative_Ha"]
        )
        ea = {}
        for name in ("S", "D3", "D5"):
            if first:
                target = next(
                    t
                    for t in row["targets"]["CCSDT_R3"]
                    if t["target"] == name and t["m2"] == 1
                )
                ea[name] = target["energy_Ha"] - core
            else:
                ea[name] = row["targets"][name]["EA_Ha"]
        lines = row["lines_cm1"]["CCSDT_R3"] if first else row["lines_cm1"]
        cells[key] = dict(
            core_electrons=4 if key.startswith("frozen") else 18,
            virtual_spinors=12 if key.endswith("_sd") else 18,
            core_energy_Ha=core,
            EA_Ha=ea,
            lines_cm1=lines,
            splitting_cm1=lines["D5"] - lines["D3"],
        )
        for name in ("D3", "D5"):
            assert abs((ea[name] - ea["S"]) * HARTREE_CM1 - lines[name]) < 1e-8
    increments = {}
    for name in ("D3", "D5"):

        def v(key, name=name):
            return cells[key]["lines_cm1"][name]

        p_frozen = v("frozen4f_sdp") - v("frozen4f_sd")
        p_active = v("active4f_sdp") - v("active4f_sd")
        f_sd = v("active4f_sd") - v("frozen4f_sd")
        f_sdp = v("active4f_sdp") - v("frozen4f_sdp")
        interaction = p_active - p_frozen
        assert abs(interaction - (f_sdp - f_sd)) < 1e-9
        increments[name] = dict(
            p_at_frozen4f=p_frozen,
            p_at_active4f=p_active,
            f_at_sd=f_sd,
            f_at_sdp=f_sdp,
            model_nonadditivity_cm1=interaction,
        )
    saved = dict(
        status="PASS_SAME_STATE_FOUR_MODEL_ARITHMETIC",
        cells=cells,
        conditional_line_increments_cm1=increments,
        inputs={str(p.relative_to(ROOT)): sha(p) for p in paths.values()},
        scope="FiniteSDT/R3 model increments. Nonadditivity is a difference of conditional spectral increments, not a diagram decomposition, portable correction or experimental error bar. Source epochs differ only by the already qualified canonical-output/numerical changes; previous target equations are replayed at current source.",
    )
    save(HERE / "four_model_comparison.json", saved)
    print(
        json.dumps(
            dict(lines=cells["active4f_sdp"]["lines_cm1"], increments=increments),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
