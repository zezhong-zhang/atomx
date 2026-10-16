"""Independent Pauli kinematics only; no Atomx/NumPy, H, CI or gradients.

This is an upper bound for implementation support: all magnetic determinants
in each jj pattern with nonzero N(J) are retained. Angular coefficients may
remove additional exactly-zero raw rows. No amplitudes are pruned here.
"""
import hashlib
import json
import runpy
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    source = HERE.parent / "physical_frontier004/count_spaces.py"
    data = runpy.run_path(str(source), run_name="independent_kinematic_definitions")
    core = [orb for row in data["CORE"].values() for orb in row]
    base, layer = data["BASE_PARTICLES"], data["LAYER"]
    sectors = {(1, -1), (1, 1), (3, -1), (3, 1), (5, 1)}

    def angular(orbital):
        _, twoj, ell, m = orbital
        kappa = -(twoj + 1) // 2 if twoj == 2 * ell + 1 else (twoj + 1) // 2
        return kappa, m

    closed = Counter(angular(orb) for orb in core)

    def model(particles, double=False):
        patterns = defaultdict(Counter)
        records = []
        families = [([], 0, 1), (core, 1, 2)]
        if double:
            families.append((data["CORE"]["5p"], 2, 3))
        for holes, nh, np in families:
            for h in combinations(holes, nh):
                hocc = tuple(sorted(Counter(s[0] for s in h).items()))
                for p in combinations(particles, np):
                    pocc = tuple(sorted(Counter(s[0] for s in p).items()))
                    parity = -1 if (sum(s[2] for s in h) + sum(s[2] for s in p)) % 2 else 1
                    pattern = hocc, pocc, parity
                    m = sum(s[3] for s in p) - sum(s[3] for s in h)
                    patterns[pattern][m] += 1
                    if (m, parity) in sectors:
                        groups = closed.copy()
                        groups.subtract(angular(s) for s in h)
                        groups.update(angular(s) for s in p)
                        key = tuple(sorted((a, count) for a, count in groups.items() if count))
                        records.append((m, parity, pattern, key))
        output = {sector: Counter() for sector in sectors}
        for j, parity, pattern, key in records:
            if patterns[pattern][j] - patterns[pattern][j + 2] > 0:
                output[j, parity][key] += 1
        return output

    models = dict(baseline=model(base), rank_5p_double=model(base, True),
                  balanced_virtual=model(base + layer))
    archive = json.loads((HERE / "models.json").read_text())
    controls, results = [], {}
    for name, groups in models.items():
        rows = []
        expected = {(row["twoJ"], row["parity"]): row["raw_M_equals_J_determinants"]
                    for row in archive["models"][name]["complete_target_blocks"]}
        for sector in sorted(sectors):
            raw = sum(groups[sector].values())
            controls.append(dict(name=f"{name}/Jparity{sector}/raw", actual=raw,
                                 expected=expected[sector], passed=raw == expected[sector]))
            bra = models["baseline"][sector]
            pairs = sum(count * groups[sector][key] for key, count in bra.items())
            rows.append(dict(twoJ=sector[0], parity=sector[1], raw_determinants=raw,
                             compatible_kappa_m_pairs_upper=pairs))
        total = sum(row["compatible_kappa_m_pairs_upper"] for row in rows)
        results[name] = dict(blocks=rows, total_compatible_pairs_upper=total,
                             original_pair_cap=1_000_000, upper_bound_within_cap=total <= 1_000_000)
    output = dict(
        status="independent_kinematic_upper_bound_only", Atomx_imports=0, NumPy_imports=0,
        new_H_builds=0, new_CI_solves=0, gradients=0, optimizer_calls=0,
        inputs_sha256={"physical_frontier004/count_spaces.py": hashlib.sha256(source.read_bytes()).hexdigest(),
                       "models.json": hashlib.sha256((HERE / "models.json").read_bytes()).hexdigest()},
        controls=controls, all_controls_passed=all(row["passed"] for row in controls), models=results,
        method="Complete Pauli magnetic rows at M=J for patterns with N(J)>0; scalar radial overlap "
        "pairs only equal occupied(kappa,m) multisets. Common deeper closed shells cancel in the group key. "
        "Raw archive counts match each selected sector. This is not a measured public angular support/minor count.",
        execution_requirement="Recompute actual public support/group count before physical H/CI/virtual; "
        "use unchanged1million cap and reject rather than prune or relax.")
    if not output["all_controls_passed"]:
        raise ValueError("Kinematic archive equality failed")
    path = HERE / "overlap_budget_preflight.json"
    if path.exists():
        raise FileExistsError("Preserve existing preflight")
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({name: row["total_compatible_pairs_upper"] for name, row in results.items()}))


if __name__ == "__main__":
    main()
