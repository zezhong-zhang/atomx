"""Versioned persistence for native SCF results.

Schema 1 is the historical energy/orbital archive. Schema 2 also retains
physical identity, resolved calculation inputs and correction accounting.
Missing historical metadata is never inferred from orbital array layout.
"""
import json
from dataclasses import asdict

_METADATA_FIELDS = (
    "Z", "_equation", "provenance", "solver_diagnostics", "operator_stages", "radial_convergence",
    "_breit_provenance", "_nms_provenance", "_sms_provenance",
    "_hydrogenic_mass_remainder_provenance",
)


def _json_scalar(value):
    import numpy as np
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"unsupported result metadata: {type(value).__name__}")


def _encode_metadata(result):
    data = {k: getattr(result, k) for k in _METADATA_FIELDS if hasattr(result, k)}
    spec = getattr(result, "calculation_spec", None)
    data["calculation_spec"] = spec.to_dict() if spec is not None else None
    application = getattr(result, "correction_application", None)
    if application is not None:
        data["correction_application"] = asdict(application)
    if hasattr(result, "_correction_results"):
        data["correction_results"] = {k: asdict(v) for k, v in result._correction_results.items()}
    return json.dumps(data, default=_json_scalar, allow_nan=False)


def _restore_metadata(result, data, version):
    from atomx.calculation_spec import CalculationSpec
    from atomx.corrections.protocol import CorrectionApplication, CorrectionResult
    for key in _METADATA_FIELDS:
        if key in data:
            setattr(result, key, data[key])
    if 'operator_stages' in data:
        result.operator_stages = tuple(tuple(item) for item in data['operator_stages'])
    spec = data.get("calculation_spec")
    if spec is not None:
        if spec.get("schema_version") != 1:
            raise ValueError("unsupported calculation specification schema")
        result.calculation_spec = CalculationSpec(**spec)
    if "correction_application" in data:
        ledger = data["correction_application"]
        ledger["shifts"] = tuple(tuple(x) for x in ledger["shifts"])
        ledger["skipped"] = tuple(tuple(x) for x in ledger["skipped"])
        result.correction_application = CorrectionApplication(**ledger)
    if "correction_results" in data:
        result._correction_results = {k: CorrectionResult(**v)
                                      for k, v in data["correction_results"].items()}
    result.persistence_metadata = {
        "schema_version": version,
        "missing": [k for k in ("Z", "calculation_spec") if data.get(k) is None],
    }


def save_scf_result(self, path):
    """Save result to HDF5 for restart or analysis.

    Creates ``{path}.h5`` containing orbital coefficients, grid
    parameters, energies, and all metadata in a self-describing format.

    The file structure is::

        /metadata/           — convergence, method, theory level
        /energies/           — total, orbital, components
        /subshells/          — (n, kappa, occupation, label) for each
        /orbitals/{label}/P  — large component coefficients
        /orbitals/{label}/Q  — small component coefficients
        /basis/t             — knot sequence (for grid reconstruction)

    Parameters
    ----------
    path : str
        Output path (without extension; ``.h5`` is appended).
    """
    import h5py

    path = str(path)
    metadata_json = _encode_metadata(self)
    snapshot = getattr(self, '_local_central_field', None)
    if snapshot is not None:
        from atomx.potentials.frozen_field import frozen_scf_potential
        frozen_scf_potential(self)  # Validate before replacing a checkpoint.

    filepath = f"{path}.h5" if not path.endswith(".h5") else path
    with h5py.File(filepath, "w") as f:
        f.attrs["atomx_version"] = "0.2.0"
        f.attrs["format"] = "SCFResult"
        f.attrs["schema_version"] = 2
        f.create_dataset("physical_metadata", data=metadata_json)

        # Metadata
        meta = f.create_group("metadata")
        meta.attrs["converged"] = self.converged
        meta.attrs["iterations"] = self.iterations
        meta.attrs["residual"] = self.residual
        if self._theory_level is not None:
            meta.attrs["theory_level"] = self._theory_level
        if self._method_name is not None:
            meta.attrs["method_name"] = self._method_name

        # Energies
        eng = f.create_group("energies")
        eng.attrs["total_energy_hartree"] = self.total_energy
        for label, E in self.orbital_energies.items():
            eng.attrs[f"orbital_{label}"] = E
        comp = eng.create_group("components")
        # Components also carry structured nuclear/operator provenance.
        # HDF5 attributes cannot represent dicts or None without object dtype.
        eng.create_dataset("components_json", data=json.dumps(
            self.energy_components, default=_json_scalar, allow_nan=False))
        for key, val in self.energy_components.items():
            if isinstance(val, (str, int, float, bool)):
                comp.attrs[key] = val

        # Subshells (use safe keys — labels like "2p3/2" contain "/")
        sub = f.create_group("subshells")
        n_sub = len(self.subshells)
        sub.attrs["count"] = n_sub
        sub.create_dataset("n", data=[sh.n for sh in self.subshells])
        sub.create_dataset("kappa", data=[sh.kappa for sh in self.subshells])
        sub.create_dataset("occupation", data=[sh.occupation for sh in self.subshells])
        labels = [sh.label for sh in self.subshells]
        sub.create_dataset("label", data=labels)

        # Orbital coefficients (keyed by safe index).  ``orbitals_raw``
        # per-kappa lists are positionally aligned with the per-kappa
        # order of ``subshells`` (see ``orbitals_by_subshell``) — NOT by
        # ``n - 1 - l``, which silently drops orbitals of gapped configs
        # (e.g. 1s2 3s1) whose quantum-number index exceeds the list.
        orb_map = self.orbitals_by_subshell()
        if self.orbitals_raw:
            missing = [sh.label for sh in self.subshells
                       if (sh.n, sh.kappa) not in orb_map]
            if missing:
                raise ValueError(
                    f"orbitals_raw does not cover subshells {missing}; "
                    "refusing to save an incomplete result (silent "
                    "orbital loss on restart)")
        orb_grp = f.create_group("orbitals")
        for i, sh in enumerate(self.subshells):
            orb = orb_map.get((sh.n, sh.kappa))
            if orb is None:
                continue
            P, Q, occ = orb
            g = orb_grp.create_group(f"orb_{i}")
            g.create_dataset("P", data=P, compression="gzip")
            g.create_dataset("Q", data=Q, compression="gzip")
            g.attrs["n"] = sh.n
            g.attrs["kappa"] = sh.kappa
            g.attrs["occupation"] = occ
            g.attrs["label"] = sh.label

        # Basis parameters (for grid reconstruction)
        if self._bsp_data:
            basis = f.create_group("basis")
            basis.create_dataset("t", data=self._bsp_data["t"])
            basis.attrs["ks"] = int(self._bsp_data["ks"])
            basis.attrs["ksp"] = int(self._bsp_data.get("ksp",
                                     self._bsp_data["ks"]))
            if "tp" in self._bsp_data:
                basis.create_dataset("tp", data=self._bsp_data["tp"])
        if snapshot is not None:
            group = f.create_group('local_central_field')
            group.create_dataset('radius', data=snapshot.radius)
            group.create_dataset('screening_charge', data=snapshot.screening_charge)
            group.create_dataset('subshells', data=self._local_central_field_subshells)
            for key in ('Z', 'electron_count', 'residual_charge', 'exchange',
                        'fac_ihx', 'fac_hxs', 'fac_csi'):
                group.attrs[key] = getattr(snapshot, key)


def load_scf_result(path, *, result_type):
    """Load a previously saved HDF5 result.

    Parameters
    ----------
    path : str
        Path to ``.h5`` file (extension optional).

    Returns
    -------
    SCFResult
        Fully reconstructed result with bsp_data for downstream use.
    """
    import h5py

    from atomx.config.subshell import Subshell

    path = str(path)

    filepath = f"{path}.h5" if not path.endswith(".h5") else path
    with h5py.File(filepath, "r") as f:
        version = int(f.attrs.get("schema_version", 1))
        if version not in (1, 2):
            raise ValueError(f"unsupported SCFResult schema version {version}")
        physical = json.loads(f["physical_metadata"][()]) if version == 2 else {}
        meta = f["metadata"]
        converged = bool(meta.attrs["converged"])
        iterations = int(meta.attrs["iterations"])
        residual = float(meta.attrs["residual"])
        theory_level = meta.attrs.get("theory_level")
        method_name = meta.attrs.get("method_name")

        eng = f["energies"]
        total_energy = float(eng.attrs["total_energy_hartree"])
        orbital_energies = {
            k.replace("orbital_", ""): float(v)
            for k, v in eng.attrs.items()
            if k.startswith("orbital_")
        }
        energy_components = (json.loads(eng["components_json"][()])
                             if "components_json" in eng else
                             dict(eng["components"].attrs) if "components" in eng else {})

        sub_grp = f["subshells"]
        n_arr = sub_grp["n"][:]
        kappa_arr = sub_grp["kappa"][:]
        occ_arr = sub_grp["occupation"][:]
        label_arr = [s.decode() if isinstance(s, bytes) else s
                     for s in sub_grp["label"][:]]
        subshells = [
            Subshell(n=int(n_arr[i]), kappa=int(kappa_arr[i]),
                     occupation=float(occ_arr[i]), label=label_arr[i])
            for i in range(len(n_arr))
        ]

        orbitals_raw = {}
        if "orbitals" in f:
            # Numeric key order — lexicographic sort would place
            # "orb_10" before "orb_2" and scramble per-kappa positional
            # alignment for configs with >9 subshells.
            orb_keys = sorted(f["orbitals"].keys(),
                              key=lambda k: int(k.split("_", 1)[1]))
            for key in orb_keys:
                g = f["orbitals"][key]
                kappa = int(g.attrs["kappa"])
                P = g["P"][:]
                Q = g["Q"][:]
                occ = float(g.attrs["occupation"])
                if kappa not in orbitals_raw:
                    orbitals_raw[kappa] = []
                orbitals_raw[kappa].append((P, Q, occ))

        bsp_data = None
        local_field_data = None
        if 'local_central_field' in f:
            group = f['local_central_field']
            local_field_data = dict(group.attrs)
            local_field_data.update(radius=group['radius'][:],
                                    screening_charge=group['screening_charge'][:])
            local_field_subshells = tuple(
                (int(n), int(k), float(q)) for n, k, q in group['subshells'][:])
        if "basis" in f:
            from atomx.numerics.bspline.basis import prepare_bspline_data
            t = f["basis"]["t"][:]
            ks = int(f["basis"].attrs["ks"])
            ksp = int(f["basis"].attrs.get("ksp", ks))
            ns = len(t) - ks
            bsp_data = prepare_bspline_data(
                t, ns, ks, kinetic_balance=(ksp != ks))

    result = result_type(
        converged=converged,
        iterations=iterations,
        residual=residual,
        total_energy=total_energy,
        orbital_energies=orbital_energies,
        subshells=subshells,
        energy_components=energy_components,
        orbitals_raw=orbitals_raw,
        _theory_level=theory_level,
        _method_name=method_name,
        _bsp_data=bsp_data,
    )

    _restore_metadata(result, physical, version)
    if local_field_data is not None:
        from atomx.potentials.frozen_field import FrozenLocalField, frozen_scf_potential
        from atomx.potentials.nuclear import make_nuclear_potential, nuclear_profile
        if result.calculation_spec is None:
            raise ValueError('saved local field lacks nuclear model identity')
        physics = result.calculation_spec.to_dict()['physics']
        profile = physics['nuclear_profile']
        model = profile['model']
        mass_number = profile.get('requested_mass_number')
        if (physics.get('screening_potential') is not None or
                nuclear_profile(result.Z, model, mass_number) != profile):
            raise ValueError('saved local field nuclear model cannot be restored exactly')
        nuclear = make_nuclear_potential(result.Z, model, mass_number)
        result._nuclear_potential = nuclear
        result._local_central_field = FrozenLocalField(**local_field_data, nuclear=nuclear)
        result._local_central_field_subshells = local_field_subshells
        frozen_scf_potential(result)
    return result



def _validate_shooting_arrays(result):
    """Reject incomplete orbital checkpoints before writing or using them."""
    import numpy as np
    radius = np.asarray(result._grid)
    jacobian = np.asarray(result._difr)
    if (radius.ndim != 1 or len(radius) < 2 or jacobian.shape != radius.shape
            or not np.all(np.isfinite(radius)) or not np.all(np.diff(radius) > 0)
            or radius[0] < 0 or not np.all(np.isfinite(jacobian))
            or not np.all(jacobian > 0)):
        raise ValueError('invalid shooting grid or Jacobian')
    shape = (len(result.subshells), len(radius))
    for name in ('_pa', '_qa'):
        array = np.asarray(getattr(result, name))
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise ValueError(f'incomplete or nonfinite shooting spinors: {name}')
    for name in ('_rv', '_rv_nuc'):
        value = getattr(result, name)
        if value is not None:
            potential = np.asarray(value)
            if potential.shape != radius.shape or not np.all(np.isfinite(potential)):
                raise ValueError(f'invalid shooting potential: {name}')
    identities = [(s.n, s.kappa) for s in result.subshells]
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate shooting orbital identities')
    for config in getattr(result, '_jj_configs', None) or ():
        if [(s.n, s.kappa) for s in config.subshells] != identities:
            raise ValueError('jj configuration does not align with shooting orbitals')


def save_shooting_result(result, path):
    """Atomically save native grid spinors, jj weights and physical metadata.

    This preserves the state as supplied, including any correction ledger.
    Save before finalizing corrections for a reusable bare-orbital checkpoint.
    No B-spline projection or reconstruction of discarded orbital data occurs.
    """
    import os
    import tempfile
    from pathlib import Path

    import h5py
    _validate_shooting_arrays(result)
    metadata = _encode_metadata(result)
    payload = dict(converged=bool(result.converged), iterations=int(result.iterations),
        total_energy=result.total_energy, orbital_energies=result.orbital_energies,
        energy_components=result.energy_components, method=result.method,
        model=result._model, subshells=[asdict(s) for s in result.subshells],
        jj_configs=([asdict(c) for c in result._jj_configs]
                    if getattr(result, '_jj_configs', None) is not None else None))
    encoded = json.dumps(payload, default=_json_scalar, allow_nan=False)
    path = Path(path)
    if not str(path).endswith('.h5'):
        path = Path(str(path)+'.h5')
    descriptor, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    os.close(descriptor)
    try:
        with h5py.File(temporary, 'w') as f:
            f.attrs['format'] = 'ShootingResult'
            f.attrs['schema_version'] = 1
            f.create_dataset('physical_metadata', data=metadata)
            f.create_dataset('state', data=encoded)
            for name in ('_grid', '_difr', '_pa', '_qa', '_rv', '_rv_nuc'):
                value = getattr(result, name)
                if value is not None:
                    f.create_dataset(name, data=value, compression='gzip', fletcher32=True)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load_shooting_result(path):
    """Restore a shooting result without solving or changing its radial metric."""
    from pathlib import Path

    import h5py

    from atomx.config.jj_coupling import JJConfig
    from atomx.config.subshell import Subshell
    from atomx.results import ShootingResult
    path = Path(path)
    if not str(path).endswith('.h5'):
        path = Path(str(path)+'.h5')
    with h5py.File(path, 'r') as f:
        if f.attrs.get('format') != 'ShootingResult' or f.attrs.get('schema_version') != 1:
            raise ValueError('unsupported shooting result format or schema')
        state = json.loads(f['state'][()])
        physical = json.loads(f['physical_metadata'][()])
        configs = state.pop('jj_configs')
        state['subshells'] = [Subshell(**s) for s in state['subshells']]
        state['_model'] = state.pop('model')
        result = ShootingResult(**state, **{name: f[name][:] if name in f else None
            for name in ('_grid', '_difr', '_pa', '_qa', '_rv', '_rv_nuc')})
    if configs is not None:
        result._jj_configs = [JJConfig(subshells=[Subshell(**s) for s in c['subshells']],
                                      weight=c['weight']) for c in configs]
    _restore_metadata(result, physical, 1)
    _validate_shooting_arrays(result)
    return result
