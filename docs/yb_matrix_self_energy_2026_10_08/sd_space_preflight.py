"""Combinatorial SD/EA-space estimate only; no amplitudes or CC prediction."""
from collections import Counter
from pathlib import Path
import json
import sys
import time
from math import comb

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/"yb_diagnosis_repair_2026_10_08"))
from common import KEYS, load_reference, save, sha


def coupled_pairs(counts):
    histogram = Counter()
    for a, ka in enumerate(counts):
        ja = 2*abs(ka)-1
        la = ka if ka > 0 else -ka-1
        for kb in list(counts)[a:]:
            jb = 2*abs(kb)-1
            lb = kb if kb > 0 else -kb-1
            different = counts[ka]*counts[kb] if ka != kb else comb(counts[ka], 2)
            identical = counts[ka] if ka == kb else 0
            for twoJ in range(abs(ja-jb), ja+jb+1, 2):
                histogram[twoJ, (la+lb) % 2] += different+(identical if twoJ % 4 == 0 else 0)
    return histogram


def main():
    started = time.monotonic()
    # Before counting: fixed complete l6/200 spaces and 19 occupied radial
    # orbitals. No integral allocation; 60 s/64 MiB cap. These are symmetry
    # dimensions, not measured SD memory or a proof of convergence.
    contract = dict(hypothesis="Angular reduction may make tensor storage practical while dense Q-Q storage is impossible",
                    fixed="both complete l6/200 preflight spaces; all 19 radial core orbitals",
                    scope="core singles/doubles coupled to J0 and EA 1p/2p1h symmetry counts, not an SD implementation",
                    gates="identical-j Pauli count equals binomial spin-orbital dimension; two distinct radial shells equal direct-product dimension",
                    cap_seconds=60, allocation="integer histograms only; no amplitude/integral matrix; <=64 MiB")
    save(ROOT/"sd_space_contract.json", contract)
    # Analytic degeneracy anchors independent of the many-body solver.
    for k in (-1, -2, -3, -4):
        j = 2*abs(k)-1
        one = coupled_pairs({k: 1})
        assert sum((J+1)*n for (J, _), n in one.items()) == comb(j+1, 2)
        two = coupled_pairs({k: 2})
        assert sum((J+1)*n for (J, _), n in two.items()) == comb(2*(j+1), 2)
    results = []
    preflight = json.loads((ROOT/"preflight/results.json").read_text())
    for case in preflight["cases"]:
        core, _, _, _ = load_reference(case["case"]["reference"])
        occupied = Counter(s.kappa for s in core.subshells)
        virtual = {b["kappa"]: b["count"] for b in case["blocks"]}
        cp, vp = coupled_pairs(occupied), coupled_pairs(virtual)
        singles = sum(n*virtual.get(k, 0) for k, n in occupied.items())
        doubles = sum(n*vp[J, parity] for (J, parity), n in cp.items())
        vspin = sum(2*abs(k)*n for k, n in virtual.items())
        m_upper = comb(68, 2)*comb(vspin, 2)
        ea = {}
        for label, (_, k) in KEYS.items():
            jv = 2*abs(k)-1
            lv = k if k > 0 else -k-1
            dimension = 0
            for kh, ncore in occupied.items():
                jh = 2*abs(kh)-1
                lh = kh if kh > 0 else -kh-1
                dimension += ncore*sum(n for (J, parity), n in vp.items()
                                      if parity == (lv+lh) % 2 and abs(J-jh) <= jv <= J+jh)
            ea[label] = dict(one_particle_dimension=virtual[k], two_particle_one_hole_dimension=dimension,
                             one_real_vector_GiB=(dimension+virtual[k])*8/1024**3,
                             dense_QQ_real_GiB=dimension**2*8/1024**3)
        results.append(dict(case=case["case"]["name"], virtual_spin_orbitals=vspin,
                            core_singles_dimension=singles, core_J0_doubles_dimension=doubles,
                            one_core_SD_real_vector_GiB=(singles+doubles)*8/1024**3,
                            unadapted_mscheme_doubles_upper_count=m_upper, EA_sector=ea))
    assert time.monotonic()-started < 60
    save(ROOT/"sd_space_preflight.json", dict(cases=results, anchors_passed=True,
                                             preflight_sha256=sha(ROOT/"preflight/results.json"),
                                             source_script_sha256=sha(__file__), seconds=time.monotonic()-started,
                                             scope="Counting only. Several work vectors, recoupling, integral storage and contraction work are not estimated here; core/EA sectors require linked-diagram and partition analysis. No SD spectrum or resource feasibility pass."))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
