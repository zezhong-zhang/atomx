"""Bounded independent same-reference DC SD/MBPT2 diagnostic.

The raw native first-order context is NOT a converged CoreSDResult. Public
solution identities stay untouched; native COMMON is explicitly restored.
"""

import hashlib
import itertools
import json
import os
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT / "source"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from test_monovalent_sd import elementary, full_operator  # noqa: E402
from test_reduced_monovalent_sd import (  # noqa: E402
    MAGNETIC_COLUMNS,
    expanded,
    reduced_core,
    test_complete_reduced_signed_map_against_fock_and_primary_anchor,
)

from atomx.methods.monovalent_sd import FiniteMonovalentSD, _core_terms  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def terms_from_integrals(integrals):
    """Independent complete one/two-body expansion; no native SD RHS."""
    n = len(integrals.labels)
    terms = [(float(integrals.frozen_energy_Ha), ())]
    for p, q in itertools.product(range(n), repeat=2):
        value = float(integrals.one_body_Ha[p, q].real)
        if value:
            terms.append((value, ((p, True), (q, False))))
    for p, q in itertools.combinations(range(n), 2):
        for r, s in itertools.combinations(range(n), 2):
            value = float(integrals.eri(p, q, r, s).real)
            if value:
                terms.append((value, ((p, True), (q, True), (s, False), (r, False))))
    return tuple(terms)


class Factory:
    def __init__(self, variant):
        self.variant = variant

    def mktemp(self, name):
        path = OUT / (name + "-" + self.variant + "-" + os.environ.get("ATOMX_DIAGNOSTIC_RUN", "run001"))
        path.mkdir(exist_ok=False)
        return path


def main():
    contract = json.loads((OUT / "contract.json").read_text())
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    assert all(digest(ROOT / p) == h for p, h in contract["source_sha256"].items())
    save("probe_identity.json", {"sha256": digest(__file__), "load": os.getloadavg()})
    start = time.perf_counter()
    rows, arrays = [], {}
    all_passed = True
    for variant in contract["fixed"]["variants"]:
        before = time.perf_counter()
        core, result, reduced = reduced_core.__wrapped__(
            Factory(variant), SimpleNamespace(param=variant)
        )
        integrals = core.integrals
        n, o = len(integrals.labels), integrals.noccupied
        hterms = terms_from_integrals(integrals)
        cmasks, hc = full_operator(n, o, hterms)
        amasks, ha = full_operator(n, o + 1, hterms)
        cindex = cmasks.index((1 << o) - 1)
        eps = integrals.fock_Ha.diagonal().real
        cdiag = np.array([sum(eps[i] for i in range(n) if mask & (1 << i)) for mask in cmasks])
        adiag = np.array([sum(eps[i] for i in range(n) if mask & (1 << i)) for mask in amasks])
        gaps = cdiag[cindex] - cdiag
        core_wave = np.divide(hc[:, cindex], gaps, out=np.zeros(len(cmasks)), where=np.abs(gaps) > 1e-8)
        # Same diagonal-Fock reference; exact first-order singles AND doubles.
        born = core.source / core.denominators
        core_terms = _core_terms(core, born)  # representation bridge only
        _, tc = full_operator(n, o, core_terms)
        _, ta = full_operator(n, o + 1, core_terms)
        core_direction_error = float(np.max(abs(tc[:, cindex] - core_wave)))
        expected_core_e2 = float(hc[cindex] @ core_wave)
        actual_core_e2 = core.energy(born)
        variant_row = dict(variant=variant, spinors=n, core_payload_bytes=core.plan["complete_payload_bound_bytes"], reduced_payload_bytes=reduced.plan["complete_payload_bound_bytes"], core_residual_Ha=result.residual_max_Ha, core_first_order_signed_error=core_direction_error, core_mbpt2_expected_Ha=expected_core_e2, core_mbpt2_native_Ha=actual_core_e2, core_mbpt2_error_Ha=abs(actual_core_e2-expected_core_e2), magnetic_columns=[], native_source_identity=dict(core.provenance), backend_manifest=core._manifest)
        variant_row["pass"] = core_direction_error < 1e-11 and abs(actual_core_e2-expected_core_e2) < 1e-11
        for key, m in MAGNETIC_COLUMNS:
            test_complete_reduced_signed_map_against_fock_and_primary_anchor((core, result, reduced), key, m)
            v = integrals.labels.index((*key, m))
            finite = FiniteMonovalentSD(core, result, v, work_cap=200_000_000)
            aindex = amasks.index(finite.reference_mask)
            attachment = np.zeros((len(amasks), len(cmasks)))
            for j, mask in enumerate(cmasks):
                target, sign = elementary(mask, ((v, True),))
                if sign:
                    attachment[amasks.index(target), j] = sign
            phase = elementary(finite.core_mask, ((v, True),))[1]
            source = ha[:, aindex] - phase * attachment @ hc[:, cindex]
            source[aindex] = 0
            agaps = adiag[aindex] - adiag
            s_wave = np.divide(source, agaps, out=np.zeros(len(amasks)), where=np.abs(agaps) > 1e-8)
            class_a_expected = float(ha[aindex] @ s_wave)
            class_b_expected = float(ha[aindex] @ ta[:, aindex] - expected_core_e2)
            attached_wave = np.divide(ha[:, aindex], agaps, out=np.zeros(len(amasks)), where=np.abs(agaps) > 1e-8)
            rspt_attachment = float(ha[aindex] @ attached_wave - expected_core_e2)
            with core._lock:
                prepared = reduced._prepare(key)
                context, slots, diagonal, reference, _ = prepared
                partial = context[0]
                group = core.plan["groups"][partial]
                # Raw diagnostic context; no fake converged certificate is made.
                reduced.valence._init(group.index(key)+1, key[1], len(born), np.asfortranarray(np.zeros_like(born)))
                _, native_source = reduced._evaluate(np.zeros(len(diagonal)), prepared)
                val_born = native_source / diagonal
                class_a_native, _ = reduced._evaluate(val_born, prepared, linear=True)
                reduced.valence._init(group.index(key)+1, key[1], len(born), np.asfortranarray(born))
                class_b_native, _ = reduced._evaluate(np.zeros(len(diagonal)), prepared)
                # Restore the actual immutable solved core COMMON and context.
                core.action(result.amplitudes)
                reduced._prepare(key)
            expanded_born = expanded(reduced, key, val_born, finite)
            native_wave = np.zeros(len(amasks))
            for amplitude, (_, _, operations) in zip(expanded_born, finite.excitations, strict=True):
                target, sign = elementary(finite.reference_mask, operations)
                native_wave[amasks.index(target)] += sign * amplitude
            direction_error = float(np.max(abs(native_wave - s_wave)))
            row = dict(key=list(key), twoM=m, complete_signed_map_pass=True, finite_amplitudes=finite.plan["amplitudes"], reduced_amplitudes=len(val_born), valence_first_order_signed_error=direction_error, class_A_2p1h_expected_Ha=class_a_expected, class_A_native_Ha=class_a_native, class_A_error_Ha=abs(class_a_native-class_a_expected), class_B_core_response_minus_spectator_expected_Ha=class_b_expected, class_B_native_Ha=class_b_native, class_B_error_Ha=abs(class_b_native-class_b_expected), complete_RSPT2_attachment_expected_Ha=rspt_attachment, complete_native_second_order_Ha=class_a_native+class_b_native, closure_error_Ha=abs(class_a_expected+class_b_expected-rspt_attachment))
            row["pass"] = max(row["class_A_error_Ha"],row["class_B_error_Ha"],direction_error,row["closure_error_Ha"]) < 1e-11
            variant_row["magnetic_columns"].append(row)
            arrays[f"{variant}_{key[0]}_{key[1]}_{m}_native_valence_born"] = val_born
            arrays[f"{variant}_{key[0]}_{key[1]}_{m}_oracle_valence_wave"] = s_wave
            all_passed &= row["pass"]
        variant_row["wall_seconds"] = time.perf_counter()-before
        rows.append(variant_row)
        all_passed &= variant_row["pass"]
        save("result.json", dict(pass_=bool(all_passed),variants=rows,scope=contract["scope"],production_certificate=False,experimental_accuracy_claim=False))
        if variant_row["wall_seconds"] > 120:
            raise RuntimeError("individual variant exceeded 120 seconds")
    np.savez_compressed(OUT/"arrays.npz",**arrays)
    unchanged = all(digest(ROOT / p) == h for p,h in contract["source_sha256"].items())
    save("verification.json", dict(pass_=bool(all_passed and unchanged),source_unchanged=unchanged,live_signed_map_checks=sum(len(r["magnetic_columns"]) for r in rows),live_second_order_columns=sum(len(r["magnetic_columns"]) for r in rows),core_first_born_checks=len(rows),skip_count=0,wall_seconds=time.perf_counter()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,load_after=os.getloadavg(),scope=contract["scope"]))
    print(json.dumps({"passed":bool(all_passed and unchanged),"variants":len(rows),"columns":sum(len(r["magnetic_columns"]) for r in rows),"wall_seconds":time.perf_counter()-start}))
    if not all_passed or not unchanged:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
