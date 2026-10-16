"""Independent complete four-core DC topology gate, no production changes."""

import importlib.util
import json
import os
import resource
import sys
import time

import numpy as np
from probe import OUT, ROOT, digest, terms_from_integrals

sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT / "tools")]

from test_monovalent_sd import elementary, full_operator  # noqa: E402
from test_reduced_monovalent_sd import expanded  # noqa: E402

from atomx.methods.core_sd import CoreSD, plan_core_sd  # noqa: E402
from atomx.methods.dirac_fock import DFSolver, Subshell  # noqa: E402
from atomx.methods.mbpt import build_virtual_basis  # noqa: E402
from atomx.methods.monovalent_sd import (  # noqa: E402
    FiniteMonovalentSD,
    _core_terms,
    plan_finite_monovalent_sd,
)
from atomx.methods.reduced_monovalent_sd import (  # noqa: E402
    ReducedMonovalentSD,
    plan_reduced_monovalent_sd,
)
from atomx.methods.valence_sd import plan_valence_sd  # noqa: E402
from atomx.numerics.bspline.coulomb_features import (  # noqa: E402
    PolynomialCellCoulombFeatures,
)


def physical_seed(core, values):
    """Preserve exact simultaneous pair exchange in direct-rho radial blocks."""
    result = values.copy()
    virtual = core.plan["virtual_keys"]
    groups = core.plan["groups"]
    cursor = sum(
        sum(key in virtual for key in group)
        for a in core.plan["occupied_keys"]
        for group in groups
        if a in group
    )
    for a, b, im, jn, _ in core.plan["channels"]:
        nm = sum(key in virtual for key in groups[im])
        nn = sum(key in virtual for key in groups[jn])
        block = result[cursor : cursor + nm * nn].reshape((nm, nn), order="F")
        if a == b and im == jn:
            block[:] = 0.5 * (block + block.T)
        cursor += nm * nn
    assert cursor == len(values)
    return result


def main():
    contract = json.loads((OUT / "four_core_contract.json").read_text())
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    assert all(
        digest((OUT if p == "probe.py" else ROOT) / p) == h
        for p, h in contract["source_sha256"].items()
    )
    start = time.perf_counter()
    run = OUT / "four-core-run003"
    run.mkdir(exist_ok=False)
    (run / "probe_identity.json").write_text(
        json.dumps(
            {"source_sha256": digest(__file__), "load": os.getloadavg()}, indent=2
        )
        + "\n"
    )
    occupied = ((1, -1), (2, -1))
    virtual = ((3, -1), (4, -1), (2, -2))
    labels = tuple(
        (nr, k, m)
        for nr, k in occupied + virtual
        for m in range(-2 * abs(k) + 1, 2 * abs(k), 2)
    )
    admissions = [
        plan_finite_monovalent_sd(labels, 4, i, work_cap=2_000_000_000)
        for i in range(4, len(labels))
    ]
    (run / "admission.json").write_text(json.dumps(admissions, indent=2) + "\n")
    radial = DFSolver(
        Z=6, subshells=[Subshell(1, -1, 2), Subshell(2, -1, 2)], ns=30, ks=7
    ).solve()
    orbitals = {key: radial.orbitals_by_subshell()[key][:2] for key in occupied}
    for k in (-1, -2):
        _, states = build_virtual_basis(
            radial, k, e_max=50, potential_mode="VN", orthogonal_to_occupied=True
        )
        for i, state in enumerate(states[: 2 if k == -1 else 1]):
            orbitals[(i + 3 if k == -1 else i + 2), k] = state[:2]
    plan = plan_core_sd(
        occupied, virtual, PolynomialCellCoulombFeatures(radial._bsp_data).identity
    )
    val_plan = plan_valence_sd(plan, virtual)
    reduced_plan = plan_reduced_monovalent_sd(plan, virtual)
    assert reduced_plan["complete_payload_bound_bytes"] < 256 * 1024**2
    spec = importlib.util.spec_from_file_location(
        "four_core_builder", ROOT / "tools/build_pci_monovalent_backend.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    builder.build_monovalent(
        os.environ["ATOMX_PCI_SOURCE"],
        run / "backend",
        plan["dimensions"],
        os.environ["ATOMX_PCI_COMPILER"],
        nxv=val_plan["nxv"],
        nvh=val_plan["nvh"],
        runtime_directory=os.environ["ATOMX_PCI_RUNTIME"],
    )
    core = CoreSD(
        orbitals,
        radial._bsp_data,
        6,
        30,
        occupied_keys=occupied,
        virtual_keys=virtual,
        backend_directory=run / "backend",
        nuclear_potential=radial._nuclear_potential,
        radial_representation="polynomial",
        memory_bytes=256 * 1024**2,
    )
    solved = core.solve(residual_tolerance_Ha=1e-12)
    reduced = ReducedMonovalentSD(core, solved, valence_keys=virtual)
    ints = core.integrals
    n, o = len(ints.labels), ints.noccupied
    terms = terms_from_integrals(ints)
    cm, hc = full_operator(n, o, terms)
    am, ha = full_operator(n, o + 1, terms)
    ref = (1 << o) - 1
    ci = cm.index(ref)
    raw = np.random.default_rng(751386).normal(size=core.source.shape) * 0.013
    _, tc = full_operator(n, o, _core_terms(core, raw))
    expected = hc[:, ci] + hc @ tc[:, ci] - tc @ hc[:, ci]
    qsd = np.array([0 < (ref & ~mask).bit_count() <= 2 for mask in cm])
    _, actual_operator = full_operator(
        n, o, _core_terms(core, core.action(raw) - core.source)
    )
    unrestricted_error = float(np.max(abs(expected[qsd] + actual_operator[qsd, ci])))
    physical = physical_seed(core, raw)
    _, tc = full_operator(n, o, _core_terms(core, physical))
    expected = hc[:, ci] + hc @ tc[:, ci] - tc @ hc[:, ci]
    _, actual_operator = full_operator(
        n, o, _core_terms(core, core.action(physical) - core.source)
    )
    action_error = float(np.max(abs(expected[qsd] + actual_operator[qsd, ci])))
    energy_error = abs(core.energy(physical) - float(hc[ci] @ tc[:, ci]))
    np.savez_compressed(
        run / "signed_seed_control.npz",
        unrestricted_rho=raw,
        physical_rho=physical,
        physical_expected_Ha=expected,
        physical_encoded_native_Ha=actual_operator[:, ci],
    )
    eps = ints.fock_Ha.diagonal().real
    cdiag = np.array([sum(eps[i] for i in range(n) if mask & (1 << i)) for mask in cm])
    adiag = np.array([sum(eps[i] for i in range(n) if mask & (1 << i)) for mask in am])
    cgaps = cdiag[ci] - cdiag
    coupled_core_degenerate = float(
        np.max(abs(hc[(abs(cgaps) < 1e-8) & (np.arange(len(cm)) != ci), ci]), initial=0)
    )
    assert coupled_core_degenerate < 1e-11
    tcore = np.divide(hc[:, ci], cgaps, out=np.zeros(len(cm)), where=abs(cgaps) > 1e-8)
    born = core.source / core.denominators
    _, ct = full_operator(n, o, _core_terms(core, born))
    _, at = full_operator(n, o + 1, _core_terms(core, born))
    expected_core = float(hc[ci] @ tcore)
    native_core = core.energy(born)
    core_direction_error = float(np.max(abs(ct[:, ci] - tcore)))
    result = dict(
        source_sha256=digest(__file__),
        spinors=n,
        core_electrons=o,
        core_determinants=len(cm),
        attached_determinants=len(am),
        core_signed_QSD_action_error_Ha=action_error,
        unrestricted_raw_negative_control_error_Ha=unrestricted_error,
        unrestricted_raw_failure_expected=True,
        core_signed_energy_error_Ha=energy_error,
        core_first_order_signed_error=core_direction_error,
        core_mbpt2_expected_Ha=expected_core,
        core_mbpt2_native_Ha=native_core,
        core_mbpt2_error_Ha=abs(native_core - expected_core),
        core_residual_Ha=solved.residual_max_Ha,
        rows=[],
        scope=contract["scope"],
        production_certificate=False,
    )
    passed = (
        max(
            action_error,
            energy_error,
            core_direction_error,
            abs(native_core - expected_core),
        )
        < 1e-11
    )
    for key in virtual:
        for projection in range(-2 * abs(key[1]) + 1, 2 * abs(key[1]), 2):
            v = ints.labels.index((*key, projection))
            finite = FiniteMonovalentSD(core, solved, v, work_cap=2_000_000_000)
            ai = am.index(finite.reference_mask)
            attachment = np.zeros((len(am), len(cm)))
            for j, mask in enumerate(cm):
                target, sign = elementary(mask, ((v, True),))
                if sign:
                    attachment[am.index(target), j] = sign
            phase = elementary(ref, ((v, True),))[1]
            source = ha[:, ai] - phase * attachment @ hc[:, ci]
            source[ai] = 0
            gaps = adiag[ai] - adiag
            coupled_degenerate = float(
                np.max(
                    abs(source[(abs(gaps) < 1e-8) & (np.arange(len(am)) != ai)]),
                    initial=0,
                )
            )
            assert coupled_degenerate < 1e-11
            s_wave = np.divide(
                source, gaps, out=np.zeros(len(am)), where=abs(gaps) > 1e-8
            )
            ae = float(ha[ai] @ s_wave)
            be = float(ha[ai] @ at[:, ai] - expected_core)
            full_wave = np.divide(
                ha[:, ai], gaps, out=np.zeros(len(am)), where=abs(gaps) > 1e-8
            )
            rspt = float(ha[ai] @ full_wave - expected_core)
            with core._lock:
                prepared = reduced._prepare(key)
                context, _, diagonal, _, _ = prepared
                group = core.plan["groups"][context[0]]
                reduced.valence._init(
                    group.index(key) + 1,
                    key[1],
                    len(born),
                    np.asfortranarray(np.zeros_like(born)),
                )
                _, sv = reduced._evaluate(np.zeros(len(diagonal)), prepared)
                vborn = sv / diagonal
                av, _ = reduced._evaluate(vborn, prepared, linear=True)
                reduced.valence._init(
                    group.index(key) + 1, key[1], len(born), np.asfortranarray(born)
                )
                bv, _ = reduced._evaluate(np.zeros(len(diagonal)), prepared)
                core.action(solved.amplitudes)
                reduced._prepare(key)
            wave = np.zeros(len(am))
            for coefficient, (_, _, ops) in zip(
                expanded(reduced, key, vborn, finite), finite.excitations, strict=True
            ):
                target, sign = elementary(finite.reference_mask, ops)
                wave[am.index(target)] += sign * coefficient
            row = dict(
                key=list(key),
                twoM=projection,
                class_A_expected_Ha=ae,
                class_A_native_Ha=av,
                class_A_error_Ha=abs(ae - av),
                class_B_expected_Ha=be,
                class_B_native_Ha=bv,
                class_B_error_Ha=abs(be - bv),
                valence_first_order_signed_error=float(np.max(abs(wave - s_wave))),
                RSPT2_attachment_Ha=rspt,
                closure_error_Ha=abs(ae + be - rspt),
                coupled_degenerate_source_Ha=coupled_degenerate,
            )
            row["pass"] = (
                max(
                    row["class_A_error_Ha"],
                    row["class_B_error_Ha"],
                    row["closure_error_Ha"],
                    row["valence_first_order_signed_error"],
                )
                < 1e-11
            )
            result["rows"].append(row)
            passed &= row["pass"]
    result.update(
        pass_=bool(passed),
        wall_seconds=time.perf_counter() - start,
        peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_unchanged=all(
            digest((OUT if p == "probe.py" else ROOT) / p) == h
            for p, h in contract["source_sha256"].items()
        ),
        skip_count=0,
        backend_manifest=core._manifest,
    )
    (run / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "pass_",
                    "core_signed_QSD_action_error_Ha",
                    "core_mbpt2_error_Ha",
                    "wall_seconds",
                    "peak_RSS_KiB",
                )
            }
        )
    )
    if not passed or not result["source_unchanged"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
