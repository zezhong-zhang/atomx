"""Complete solved noncanonical Sigma2, full signed values and joint response.

Bounded magnetic tensors are independent finite reference objects. Production
valence columns use the public matrix-free interface and all declared channels.
"""

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OWNER = Path(__file__).resolve().parent
REF = Path("/home/zzhang/code/atomx-references/2026-10-09-ci-fscc")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from build_pci_sigma2_backend import build_sigma2 as build_public_sigma2  # noqa: E402
from build_pci_valence_backend import build_valence  # noqa: E402
from build_sigma2_fock_control import build_sigma2_fock  # noqa: E402
from finite_direct_coulomb import direct_coulomb  # noqa: E402
from finite_valence_fixture import prepare  # noqa: E402
from general_rho_control import core_numerators, expand_valence  # noqa: E402
from probe_sigma2_control003 import valence_channels  # noqa: E402
from sigma2_fock_oracle import action as fock_action  # noqa: E402
from sigma2_magnetic_oracle import action as coulomb_action  # noqa: E402
from sigma2_magnetic_oracle import expand_sigma2  # noqa: E402

from atomx.angular.recoupling import l_of_kappa  # noqa: E402
from atomx.angular.sd_rho import kappa_from_partial  # noqa: E402
from atomx.methods.sigma2_sd import Sigma2SD  # noqa: E402
from atomx.methods.valence_sd import ValenceSD  # noqa: E402


def dimensions(occupied, virtual, lmax):
    rows = {key: valence_channels(occupied, key[1], lmax, "rho") for key in virtual}
    vw, cn = [], []
    for iv, v in enumerate(virtual):
        for iw, w in enumerate(virtual):
            for im in range(2 * lmax + 1):
                km = kappa_from_partial(im + 1)
                for jn in range(2 * lmax + 1):
                    kn = kappa_from_partial(jn + 1)
                    for k in range(
                        max(abs(abs(km) - abs(v[1])), abs(abs(kn) - abs(w[1]))),
                        min(abs(km) + abs(v[1]) - 1, abs(kn) + abs(w[1]) - 1) + 1,
                    ):
                        if (l_of_kappa(km) + l_of_kappa(v[1])) % 2 == (
                            l_of_kappa(kn) + l_of_kappa(w[1])
                        ) % 2:
                            vw.append((iv, iw, im, jn, k))
                for c, (_, kc) in enumerate(occupied):
                    for k in range(
                        max(abs(abs(kc) - abs(v[1])), abs(abs(km) - abs(w[1]))),
                        min(abs(kc) + abs(v[1]) - 1, abs(km) + abs(w[1]) - 1) + 1,
                    ):
                        if (
                            not (l_of_kappa(kc) + l_of_kappa(v[1]) + k) % 2
                            and not (l_of_kappa(km) + l_of_kappa(w[1]) + k) % 2
                        ):
                            cn.append((iv, iw, c, im, k))
    sizes = dict(
        NZ=len(virtual),
        NXV=1,
        NVH=1,
        NVH2=sum(map(len, rows.values())),
        NVWH=len(vw) + 1,
        NXCN=len(cn) + 1,
        NXV2=sum(
            max(
                len(valence_channels(occupied, key[1], lmax, kind))
                for kind in ("x1", "x2")
            )
            for key in virtual
        )
        + 1,
    )
    return rows, sizes


def main(run_name, derivative_order, public_screening, energy_mode,
         core_virtual_rotation, compare_valence):
    if not run_name.startswith("sigma2_solved") or not run_name[13:].isdigit():
        raise ValueError("explicit fresh sigma2_solvedNNN evidence identifier required")
    here = OWNER / run_name
    here.mkdir(exist_ok=False)
    sources = tuple(
        OWNER / name
        for name in (
            Path(__file__).name,
            "build_sigma2_control.py",
            "build_sigma2_fock_control.py",
            "sigma2_control.f90",
            "sigma2_fock_control.f90",
            "sigma2_magnetic_oracle.py",
            "sigma2_fock_oracle.py",
            "finite_direct_coulomb.py",
            "finite_valence_fixture.py",
            "general_rho_control.py",
            "probe_sigma2_control003.py",
            "build_valence_fock_control.py",
            "build_valence_control.py",
        )
    ) + tuple(
        ROOT / name
        for name in (
            "tools/build_pci_valence_backend.py",
            "tools/build_pci_sd_backend.py",
            "src/atomx/methods/core_sd.py",
            "src/atomx/methods/valence_sd.py",
            "src/atomx/methods/_pci_valence.f90",
            "src/atomx/methods/sigma2_sd.py",
            "src/atomx/methods/_pci_sigma2.f90",
            "tools/build_pci_sigma2_backend.py",
        )
    )
    hashes = {}
    for src in sources:
        relative = src.relative_to(ROOT)
        target = here / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        hashes[str(relative)] = hashlib.sha256(src.read_bytes()).hexdigest()
    (here / "source_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    contract = dict(
        hypothesis="Original complete Coulomb screening plus irreducible Fov contribution, on solved noncanonical public valence columns, matches independent full magnetic values and total joint energy response.",
        fixed="C2+ all1s2/2s2 core, all6declared s/p radial virtuals/16magnetic states, Z6/ns40/ks7, actual nativeDC Fock, .37rad paired virtual rotations, no synthetic operator, no fitted energy; solved core fixed for energy response.",
        energy_mode=energy_mode,
        core_virtual_rotation_rad=core_virtual_rotation,
        occupied_projector="2s/3s rotated before constructing all native DC integrals"
        if core_virtual_rotation else "original DHF projector",
        matched_valence_case=compare_valence,
        energy="Each partial wave uses its lowest declared Fock-diagonal virtual energy+.075+delta"
        if energy_mode == "partial_wave_reference"
        else "Each column eps_v+.075+delta",
        derivative="all valence energies shift by same delta; Sigma2 explicit denominators slope2, complete implicit column response",
        gates=dict(
            signed_values_Ha=1e-11,
            signed_joint_response_Ha=1e-11,
            raw_core_and_valence_residual_Ha=1e-10,
            difference_abs_Ha=1e-9,
            difference_rel=1e-6,
        ),
        difference_steps=[1e-3, 3e-4, 1e-4],
        public_screening=public_screening,
        difference_order=derivative_order,
        difference_refinement="order8 Richardson from h,h/2,h/4,h/8, retaining original steps/gate; order2 failures kept"
        if derivative_order == 8
        else "centered order2",
        preflight="512MiB complete core/valence/screening/Krylov and bounded finite reference; one worker, no timing claim.",
        stopping="On any failure preserve full vectors, actual sources and unchanged gates; no truncation/fit/tolerance relaxation.",
        status="finite directed fixed-energy solved screening equation proof; no Hermitian CI metric or physical spectrum claim",
        load=os.getloadavg(),
    )
    (here / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    occupied = ((1, -1), (2, -1))
    virtual = ((3, -1), (4, -1), (2, 1), (3, 1), (2, -2), (3, -2))
    rows, sizes = dimensions(occupied, virtual, 1)

    def public_builder(source, directory, dims, compiler, *, nxv, nvh, runtime):
        return build_valence(
            source,
            directory,
            dims,
            compiler,
            nxv=nxv,
            nvh=nvh,
            runtime_directory=runtime,
        )

    def builder(source, directory, dims, compiler, *, nxv, nvh, runtime):
        sizes.update(NXV=nxv, NVH=nvh)
        # Complete screening padding, response wrapper scratch, radial controls,
        # magnetic reference arrays and 40 full valence/Krylov work vectors.
        extra = 8 * (
            dims["NXX"] ** 2 * 16 * sum(sizes.values())
            + 20**4 * 20
            + 1600 * 40
            + dims["NHF"] * 40 * 6
        )
        if extra + 82 * 1024**2 > 512 * 1024**2:
            raise MemoryError("complete finite solved screening payload exceeds cap")
        (here / "resource.json").write_text(
            json.dumps(
                dict(
                    extra_bytes=extra,
                    core_bound_bytes=82 * 1024**2,
                    cap_bytes=512 * 1024**2,
                    sizes=sizes,
                    load=os.getloadavg(),
                ),
                indent=2,
            )
            + "\n"
        )
        if public_screening:
            return build_public_sigma2(
                source,
                directory,
                dims,
                compiler,
                sizes=sizes,
                runtime_directory=runtime,
            )
        return build_sigma2_fock(
            source,
            directory,
            dims,
            compiler,
            sizes=sizes,
            runtime=runtime,
            valence_builder=public_builder,
        )

    core, orbitals, plan, actual_rows, _, _, problem, fock = prepare(
        here, REF, backend_builder=builder, fock_probe=False, virtual_rotation=0.37,
        core_virtual_rotation=core_virtual_rotation,
    )
    if actual_rows != rows or tuple(plan["virtual_keys"]) != virtual:
        raise AssertionError("complete channel/source identities differ")
    result = problem.solve(residual_tolerance_Ha=1e-12)
    valence = ValenceSD(problem, result, valence_keys=virtual)
    screening = Sigma2SD(valence) if public_screening else None
    core_s, core_d = core_numerators(problem, plan, result.amplitudes)
    direct, direct_error = direct_coulomb(
        problem, orbitals, core._bsp_data, radial_representation="bound_features"
    )
    eps = np.diag(fock).copy()
    labels = problem.integrals.labels
    nc, n = problem.integrals.noccupied, len(labels)
    input_arrays = dict(
        direct=direct,
        eps=eps,
        core_s=core_s,
        core_d=core_d,
        fock=fock,
        labels=np.asarray(labels),
        core_amplitudes=result.amplitudes,
    )
    np.savez_compressed(here / "inputs.npz", **input_arrays)
    old = np.load(OWNER / compare_valence / "inputs.npz")
    match = {
        name: bool(np.array_equal(value, old[name]))
        for name, value in input_arrays.items()
    }
    (here / "public_input_equality.json").write_text(json.dumps(match, indent=2) + "\n")
    if not all(match.values()):
        raise AssertionError(
            "combined backend changed matched public physical inputs/core"
        )
    physical_fov = float(np.max(abs(fock[:nc, nc:])))
    (here / "reference_change.json").write_text(json.dumps(dict(
        core_virtual_rotation_rad=core_virtual_rotation,
        actual_fov_max_Ha=physical_fov,
        physical_identity=dict(problem.integrals.identity),
        meaning="explicit changed occupied reference; not orbital optimization or invariance",
    ), indent=2) + "\n")
    if core_virtual_rotation and physical_fov < 1e-3:
        raise AssertionError("strong native occupied-virtual probe is ineffective")
    # Independent references preserve the actual float64 physical input values.
    ld_direct, ld_eps, ld_core_s, ld_core_d, ld_fock = (
        np.asarray(a, dtype=np.longdouble) for a in (direct, eps, core_s, core_d, fock)
    )
    ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS")
    iptr = np.ctypeslib.ndpointer(dtype=np.int32, flags="F_CONTIGUOUS")
    init = problem._backend.atomx_vw_init
    init.argtypes = [ctypes.c_int, iptr, ptr]
    init.restype = None
    load = problem._backend.atomx_vw_load
    load.argtypes = [ctypes.c_int, ctypes.c_int, iptr, ptr, ptr]
    load.restype = None
    d = plan["dimensions"]
    radial_shape = (d["NXX"], d["NXX"])
    keys = np.asfortranarray(np.asarray(virtual).T, dtype=np.int32)
    records = []
    cases = {}
    scales = (1.0, 0.5, 0.25, 0.125) if derivative_order == 8 else (1.0,)
    deltas = (0.0,) + tuple(
        s * h * scale
        for h in contract["difference_steps"]
        for scale in scales
        for s in (-1, 1)
    )
    for case_index, delta in enumerate(deltas):
        out = here / f"case{case_index:02d}"
        out.mkdir()
        sigmas = np.zeros((n, n, nc, n))
        primes = np.zeros_like(sigmas)
        energies = eps + 0.075 + delta
        if energy_mode == "partial_wave_reference":
            for v in range(nc, n):
                energies[v] = (
                    min(eps[j] for j in range(nc, n) if labels[j][1] == labels[v][1])
                    + 0.075
                    + delta
                )
        radial = {}
        radial_prime = {}
        column_records = []
        solved_columns = []
        for key in virtual:
            slot = labels.index((*key, -2 * abs(key[1]) + 1))
            column = valence.solve(
                key, float(energies[slot]), residual_tolerance_Ha=1e-11
            )
            solved_columns.append(column)
            full = np.zeros(radial_shape + (len(rows[key]),), order="F")
            prime = np.zeros_like(full, order="F")
            cursor = 0
            for c, (_, im, jn, _) in enumerate(rows[key]):
                for nn, nk in enumerate(plan["groups"][jn]):
                    for m, mk in enumerate(plan["groups"][im]):
                        if mk in virtual and nk in virtual:
                            full[m, nn, c] = column.numerators[cursor]
                            prime[m, nn, c] = column.numerator_derivative[cursor]
                            cursor += 1
            if cursor != len(column.numerators):
                raise AssertionError("complete numerator coverage differs")
            radial[key], radial_prime[key] = full, prime
            for mj in range(-2 * abs(key[1]) + 1, 2 * abs(key[1]), 2):
                v = labels.index((*key, mj))
                sigmas[:, :, :, v] = expand_valence(problem, plan, rows[key], full, v)
                primes[:, :, :, v] = expand_valence(problem, plan, rows[key], prime, v)
            np.savez_compressed(
                out / f"column_{key[0]}_{key[1]}.npz",
                numerators=column.numerators,
                prime=column.numerator_derivative,
                sigma1=column.sigma1_Ha,
                sigma1_prime=column.sigma1_derivative,
            )
            column_records.append(
                dict(
                    key=key,
                    energy_Ha=column.energy_parameter_Ha,
                    residual_Ha=column.residual_max_Ha,
                    response_residual_Ha=column.derivative_residual_max_Ha,
                    matvecs=column.matrix_vector_products,
                    provenance=dict(column.provenance),
                )
            )
        # Valence initialization restores complete core numerators in native COMMON.
        init(
            len(virtual),
            keys,
            np.asfortranarray(
                [energies[labels.index((*k, -2 * abs(k[1]) + 1))] for k in virtual]
            ),
        )
        for i, key in enumerate(virtual):
            channels = np.asfortranarray(np.asarray(rows[key]).T + 1, dtype=np.int32)
            channels[3] -= 1
            load(i + 1, len(rows[key]), channels, radial[key], radial_prime[key])
        shape = radial_shape + (sizes["NVWH"] + 1,)
        count = np.zeros(1, dtype=np.int32)
        channels = np.zeros((5, sizes["NVWH"]), dtype=np.int32, order="F")
        coulomb = np.zeros(shape, order="F")
        coulomb_prime = np.zeros_like(coulomb, order="F")
        native_fock = np.zeros_like(coulomb, order="F")
        native_fock_prime = np.zeros_like(coulomb, order="F")
        problem._call(
            "atomx_vw_evaluate_joint", [count, channels, coulomb, coulomb_prime]
        )
        coulomb_count = int(count[0])
        coulomb_channels = channels.copy()
        problem._call(
            "atomx_vw_fock", [count, channels, native_fock, native_fock_prime]
        )
        if int(count[0]) != coulomb_count or not np.array_equal(
            channels, coulomb_channels
        ):
            raise AssertionError(
                "Fov and original Coulomb screening channel identities differ"
            )
        full_rows = channels[:, : count[0]].T - np.asarray([1, 1, 1, 1, 0])
        native_c = expand_sigma2(labels, plan, virtual, full_rows, coulomb)
        native_cp = expand_sigma2(labels, plan, virtual, full_rows, coulomb_prime)
        native_f = expand_sigma2(labels, plan, virtual, full_rows, native_fock)
        native_fp = expand_sigma2(labels, plan, virtual, full_rows, native_fock_prime)
        ld_sigmas, ld_primes, ld_energies = (
            np.asarray(a, dtype=np.longdouble) for a in (sigmas, primes, energies)
        )
        independent_c = coulomb_action(
            ld_direct, ld_eps, ld_core_s, ld_core_d, ld_sigmas, ld_energies
        )
        independent_cp = coulomb_action(
            ld_direct,
            ld_eps,
            ld_core_s,
            ld_core_d,
            ld_sigmas,
            ld_energies,
            magnetic_primes=ld_primes,
        )
        independent_f = fock_action(ld_fock, ld_eps, ld_sigmas, ld_energies, nc)
        independent_fp = fock_action(
            ld_fock, ld_eps, ld_sigmas, ld_energies, nc, primes=ld_primes
        )
        native = native_c + native_f
        native_prime = native_cp + native_fp
        independent = independent_c + independent_f
        independent_prime = independent_cp + independent_fp
        errors = {
            name: float(np.max(abs(a - b)))
            for name, a, b in (
                ("coulomb_Ha", native_c, independent_c),
                ("coulomb_prime_Ha", native_cp, independent_cp),
                ("fock_Ha", native_f, independent_f),
                ("fock_prime_Ha", native_fp, independent_fp),
                ("total_Ha", native, independent),
                ("total_prime_Ha", native_prime, independent_prime),
            )
        }
        if screening is not None:
            public_result = screening.evaluate(solved_columns)
            expected = (coulomb + native_fock)[:, :, 1 : coulomb_count + 1]
            expected_prime = (coulomb_prime + native_fock_prime)[
                :, :, 1 : coulomb_count + 1
            ]
            if not np.array_equal(
                public_result.radial_Ha, expected
            ) or not np.array_equal(public_result.joint_derivative, expected_prime):
                raise AssertionError(
                    "public screening changed complete native signed outputs"
                )
            (out / "public_provenance.json").write_text(
                json.dumps(dict(public_result.provenance), indent=2) + "\n"
            )
            if case_index == 0:
                full_public = np.zeros_like(native)
                full_public_prime = np.zeros_like(native_prime)
                for m, nn, v, w in np.ndindex(native.shape):
                    args = (
                        labels[m + nc],
                        labels[nn + nc],
                        labels[v + nc],
                        labels[w + nc],
                    )
                    full_public[m, nn, v, w] = public_result.direct_element(*args)
                    full_public_prime[m, nn, v, w] = public_result.direct_element(
                        *args, derivative=True
                    )
                errors["public_direct_Ha"] = float(
                    np.max(abs(full_public - independent))
                )
                errors["public_direct_prime_Ha"] = float(
                    np.max(abs(full_public_prime - independent_prime))
                )
                np.savez_compressed(
                    out / "public_direct.npz",
                    values=full_public,
                    derivative=full_public_prime,
                )
        np.savez_compressed(
            out / "complete_vectors.npz",
            sigmas=sigmas,
            primes=primes,
            energies=energies,
            native=native,
            native_prime=native_prime,
            independent=independent,
            independent_prime=independent_prime,
            native_coulomb=native_c,
            native_coulomb_prime=native_cp,
            native_fock=native_f,
            native_fock_prime=native_fp,
        )
        record = dict(
            case=case_index,
            delta_Ha=delta,
            full_signed_coordinates=native.size,
            magnetic_columns=n - nc,
            errors=errors,
            columns=column_records,
        )
        records.append(record)
        (here / "comparison.json").write_text(
            json.dumps(
                dict(
                    core_residual_Ha=result.residual_max_Ha,
                    direct_error_Ha=direct_error,
                    records=records,
                ),
                indent=2,
            )
            + "\n"
        )
        print(
            json.dumps(dict(case=case_index, delta_Ha=delta, errors=errors)), flush=True
        )
        if max(errors.values()) > 1e-11:
            raise AssertionError(
                "complete solved signed value/joint-response gate failed"
            )
        cases[delta] = (native, native_prime)
    differences = []
    for h in contract["difference_steps"]:
        stencil = [
            (cases[h * scale][0] - cases[-h * scale][0]) / (2 * h * scale)
            for scale in scales
        ]
        for order in range(1, len(scales)):
            stencil = [
                (4**order * stencil[i + 1] - stencil[i]) / (4**order - 1)
                for i in range(len(stencil) - 1)
            ]
        derivative = stencil[0]
        error = abs(derivative - cases[0.0][1])
        gate = 1e-9 + 1e-6 * abs(cases[0.0][1])
        record = dict(
            step_Ha=h,
            max_error_Ha=float(np.max(error)),
            max_gate_ratio=float(np.max(error / gate)),
            status="pass" if np.all(error <= gate) else "fail",
        )
        differences.append(record)
        np.savez_compressed(
            here / f"difference_{h}.npz",
            derivative=derivative,
            analytic=cases[0.0][1],
            errors=error,
            gate=gate,
        )
    (here / "re_solved_derivative.json").write_text(
        json.dumps(differences, indent=2) + "\n"
    )
    if any(r["status"] != "pass" for r in differences):
        raise AssertionError("full jointly re-solved derivative gate failed")
    (here / "result.json").write_text(
        json.dumps(
            dict(
                status="pass",
                cases=len(deltas),
                full_signed_coordinates=65536,
                magnetic_columns=16,
                source_sha256=hashes,
                core_provenance=problem.provenance,
                scope=contract["status"],
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--derivative-order", type=int, choices=(2, 8), default=2)
    parser.add_argument("--public-screening", action="store_true")
    parser.add_argument("--core-virtual-rotation", type=float, default=0.0)
    parser.add_argument("--compare-valence", default="public_valence002")
    parser.add_argument(
        "--energy-mode",
        choices=("per_column", "partial_wave_reference"),
        default="per_column",
    )
    args = parser.parse_args()
    try:
        main(
            args.run_name,
            args.derivative_order,
            args.public_screening,
            args.energy_mode,
            args.core_virtual_rotation,
            args.compare_valence,
        )
    except Exception as exc:
        failure = OWNER / args.run_name / "failure.json"
        if failure.parent.exists():
            failure.write_text(
                json.dumps(dict(type=type(exc).__name__, message=str(exc)), indent=2)
                + "\n"
            )
        raise
