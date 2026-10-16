"""Saved quadrature support and retained-CI diagonal diagnostics, zero solves."""
import hashlib
import json
import os
import resource
import signal
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    contract = load(HERE / "contract.json")
    for name, digest in contract["inputs_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Saved radial input changed")
    if (HERE / "result.json").exists():
        raise FileExistsError("Preserve radial result")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) < 3:
        raise ValueError("Three original CPUs required")
    os.sched_setaffinity(0, {affinity[-2]})
    cap = contract["resources"]["address_space_GiB"] * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    def timeout(*_):
        raise TimeoutError("Saved radial algebra cap")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(contract["resources"]["wall_s"])
    import numpy as np

    workspace = np.load(RUN.parent / "native_run/nuclear_eol005_projected_increment/orbitals_workspace.npz", allow_pickle=False)
    canonical = np.load(RUN / "canonical29_orbitals.npz", allow_pickle=False)
    geometry = np.load(RUN / "native_hS_original_geometry.npz", allow_pickle=False)
    ns = int(workspace["basis_ns"])
    r, w = workspace["basis_gr"].astype(np.longdouble).ravel(), workspace["basis_grw"].astype(np.longdouble).ravel()
    order = np.argsort(r)
    r, w = r[order], w[order]

    def samples(component):
        values = workspace["basis_bsp_" + component].astype(np.longdouble)
        count = int(workspace["basis_ns" + component])
        out = np.zeros((values.shape[0], values.shape[1], ns), dtype=np.longdouble)
        for interval in range(values.shape[0]):
            width = min(values.shape[2], max(0, count-interval))
            if width:
                out[interval, :, interval:interval+width] = values[interval, :, :width]
        return out.reshape(-1, ns)[order]

    bp, bq = samples("p"), samples("q")
    rows, densities = {}, {}
    for key, p, q in zip(canonical["radial_keys"], canonical["P"], canonical["Q"], strict=True):
        key = tuple(map(int, key))
        pg, qg = bp @ p.astype(np.longdouble), bq @ q.astype(np.longdouble)
        density = pg**2+qg**2
        norm = density @ w
        if not np.isfinite(norm) or abs(norm-1) > 1e-10:
            raise ValueError("Original radial norm gate failed")
        mass = density*w/norm
        densities[key] = density/norm
        cumulative = np.cumsum(mass)
        median = r[min(int(np.searchsorted(cumulative, .5)), len(r)-1)]
        vec = np.concatenate((p,q)).astype(np.longdouble)
        h, s = geometry["h_" + str(key[1])].astype(np.longdouble), geometry["s_" + str(key[1])].astype(np.longdouble)
        rows[key] = dict(key=list(key), norm=float(norm), mean_r_a0=float(mass @ r), median_quadrature_r_a0=float(median),
                         rms_r_a0=float(np.sqrt(mass @ r**2)), bare_nuclear_Rayleigh_Ha=float(vec @ h @ vec/(vec @ s @ vec)),
                         cumulative_mass={str(radius):float(np.sum(mass[r <= radius])) for radius in (.25,.5,1.,2.,5.,10.)})
    comparison = []
    for virtual, valence in (((6,-3),(5,-3)),((6,2),(5,2)),((7,-2),(6,-2)),((7,-1),(6,-1)),((7,1),(6,1))):
        v, a = rows[virtual], rows[valence]
        median = a["median_quadrature_r_a0"]
        comparison.append(dict(virtual_key=list(virtual), corresponding_valence_key=list(valence),
                               virtual=v,valence=a,mean_radius_virtual_over_valence=v["mean_r_a0"]/a["mean_r_a0"],
                               virtual_mass_below_valence_median=float(np.sum(densities[virtual][r <= median]*w[r <= median])),
                               density_Bhattacharyya_overlap=float(np.sum(np.sqrt(densities[virtual]*densities[valence])*w))))
    result = load(RUN / "result.json")
    stage = next(s for s in result["stages"] if s["name"] == "balanced_virtual")
    baseline = next(s for s in result["stages"] if s["name"] == "baseline")
    matrix = np.load(RUN / "balanced_virtual_H.npz",allow_pickle=False)["matrix_Ha"]
    ci = np.load(RUN / "balanced_virtual_CI.npz",allow_pickle=False)
    old_ci = np.load(RUN / "baseline_CI.npz",allow_pickle=False)
    mapping = stage["nested_old261_indices"]
    new_rows = sorted(set(range(len(matrix)))-set(mapping))
    declaration = load(RUN / "basis_rows.json")["balanced_virtual"]
    full_sectors = declaration["full_sectors"]
    retained_sectors = [tuple(full_sectors[i]) for i in declaration["retained_indices"]]
    diagnostics = []
    for row in stage["rows"]:
        oldrow = next(a for a in baseline["rows"] if a["label"] == row["label"])
        sector = row["twoJ"],row["parity"]
        candidates = [i for i in new_rows if retained_sectors[i] == sector]
        denominators = np.diag(matrix)[candidates]-oldrow["energy_Ha"]
        coupling = matrix[np.ix_(candidates,mapping)] @ old_ci["mixing"][:,oldrow["column"]]
        diagnostics.append(dict(label=row["label"],new_CSFs_in_sector=len(candidates),
                                new_diagonal_excitation_Ha_min=float(np.min(denominators)),
                                new_diagonal_excitation_Ha_median=float(np.median(denominators)),
                                new_diagonal_excitation_Ha_max=float(np.max(denominators)),
                                old_root_to_new_CSF_coupling_norm_Ha=float(np.linalg.norm(coupling)),
                                actual_target_new_CSF_probability=float(np.sum(ci["mixing"][candidates,row["column"]]**2)),
                                diagonal_EN_estimate_Ha=float(-np.sum(coupling**2/denominators)) if np.all(denominators>0) else None,
                                exact_same_family_energy_delta_Ha=row["absolute_delta_upper_Ha"]))
    output=dict(all_radial_norm_checks_passed=True,contract_sha256=sha(HERE/"contract.json"),driver_sha256=sha(Path(__file__).resolve()),
                bound_inputs_sha256=contract["inputs_sha256"],new_H_CI_Dirac_virtual_gradient_optimizer_eigensolves=0,
                original_stationarity_passed=False,experimental_gate_passed=False,radial_completeness_certified=False,
                radial_comparisons=comparison,core_comparisons=[rows[key] for key in ((5,-1),(5,-2),(5,1),(4,-4),(4,3))],
                retained_CI_diagnostics=diagnostics,
                limits='Radial support and bare-nuclear Rayleigh are saved oneelectron diagnostics, not screened orbital energies. Density overlap is not spinor overlap. New-CSF diagonal denominators/diagonal EN are basis-dependent estimates, not Q-spectrum bounds. No screened virtuals constructed, no no-pair boundary change measured.')
    (HERE/'result.json').write_text(json.dumps(output,indent=2)+'\n')
    signal.alarm(0)
    print(json.dumps({'radii':[{'new':x['virtual_key'],'newmean':x['virtual']['mean_r_a0'],'valencemean':x['valence']['mean_r_a0'],'newmedian':x['virtual']['median_quadrature_r_a0'],'valencemedian':x['valence']['median_quadrature_r_a0'],'belowvalencemedian':x['virtual_mass_below_valence_median']} for x in comparison],'CI':diagnostics},indent=2))


if __name__ == '__main__':
    main()
