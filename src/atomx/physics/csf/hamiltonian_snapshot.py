"""Owned complete real CI matrices bound to their actual operators and inputs.

The public producer builds once through the ordinary physical matrix builder.
It is not an arbitrary matrix injection or a cache of different orbital points.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Integral, Real

import numpy as np

from atomx.physics.csf.effective_one_body import CIReferenceBinding

__all__ = ['CIHamiltonianSnapshot', 'build_ci_hamiltonian_snapshot']


def _keys(values):
    result = []
    for key in values:
        if (not isinstance(key, (tuple, list, np.ndarray)) or len(key) != 2
                or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in key)
                or key[0] < 1 or key[1] == 0):
            raise ValueError('CI Hamiltonian snapshot requires valid integer orbital keys')
        result.append(tuple(int(v) for v in key))
    if not result or len(set(result)) != len(result):
        raise ValueError('CI Hamiltonian snapshot requires unique nonempty orbital keys')
    return tuple(result)


def _model(csfs):
    from atomx.angular.csf_hamiltonian import csf_signature

    identities = []
    for csf in csfs:
        parity = csf.parity
        if isinstance(parity, (bool, np.bool_)) or not isinstance(parity, Integral) or parity not in (-1, 1):
            raise ValueError('CI Hamiltonian snapshot requires explicit integer CSF parity')
        signature = csf_signature(csf)
        identities.append((int(csf.n_electrons), int(parity), signature.key))
    if not identities or len(set(identities)) != len(identities):
        raise ValueError('CI Hamiltonian snapshot requires a nonempty distinct ordered CSF basis')
    return tuple(identities)


def _cap(value):
    from atomx.angular.csf_hamiltonian import MAX_BLOCK_DETERMINANTS

    if value is None:
        return MAX_BLOCK_DETERMINANTS
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError('CI Hamiltonian snapshot requires a positive integer determinant cap')
    return int(value)


def _representation(workspace, Z, ns):
    from atomx.physics.csf.matrix import GridCIWorkspace

    if isinstance(workspace, GridCIWorkspace):
        raise ValueError('retained CI Hamiltonians require a B-spline workspace')
    if (not isinstance(workspace, dict) and not hasattr(workspace, 'items')):
        raise ValueError('retained CI Hamiltonians require a B-spline workspace')
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < 1 for v in (Z, ns)):
        raise ValueError('CI Hamiltonian snapshot requires positive integer Z and ns')
    if workspace.get('ns') != ns:
        raise ValueError('CI Hamiltonian snapshot ns does not match its workspace')


def _coverage(orbitals, include_breit, breit_orbital_energies, one_body_shifts, one_body_operator):
    """Canonical identity of one explicit, unshifted real operator recipe."""
    if not isinstance(include_breit, bool):
        raise TypeError('retained CI include_breit must be bool')
    if one_body_shifts is not None or one_body_operator is not None:
        raise ValueError('Hamiltonian retention requires the complete dense unshifted spectrum')
    if not include_breit:
        if breit_orbital_energies is not None:
            raise ValueError('retained breit_orbital_energies requires include_breit=True')
        return ('dirac-coulomb',), (), b''
    if breit_orbital_energies is None:
        return ('dirac-coulomb', 'static-breit'), (), b''
    if not isinstance(breit_orbital_energies, Mapping):
        raise ValueError('retained finite-frequency Breit requires an orbital energy mapping')
    keys = tuple(sorted(_keys(orbitals)))
    if set(_keys(breit_orbital_energies)) != set(keys):
        raise ValueError('retained finite-frequency Breit needs exactly every provided orbital key')
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
           for v in breit_orbital_energies.values()):
        raise ValueError('retained finite-frequency Breit orbital energies must be finite real scalars')
    from atomx.physics.csf.transverse import validate_orbital_energies

    energies = validate_orbital_energies(orbitals, breit_orbital_energies)
    hexadecimal = tuple((key, energies[key].hex()) for key in keys)
    binary = np.asarray([energies[key] for key in keys], dtype='<f8').tobytes()
    return ('dirac-coulomb', 'finite-frequency-transverse'), hexadecimal, binary


@dataclass(frozen=True, init=False, eq=False)
class CIHamiltonianSnapshot:
    """Immutable complete real dense H with checked signed basis/point identity.

    Build using :func:`build_ci_hamiltonian_snapshot`. ``matrix_for`` checks
    the actual orbitals, workspace, nuclear quadrature, ordered signed CSFs,
    orbital keys and determinant cap. Byte-backed storage cannot be made
    writable. State-average weights and optimizer metrics remain separate.
    Local objects are not persistence certificates; archive their header and
    complete arrays, and rebuild using physical inputs rather than unpickle.
    """

    matrix_Ha: np.ndarray
    radial_binding: CIReferenceBinding
    orbital_keys: tuple
    orbital_basis_keys: tuple
    csf_identities: tuple
    max_determinants: int
    input_sha256: str
    matrix_sha256: str
    representation: str = 'dense-real-full-CSF'
    units: str = 'Hartree'
    hamiltonian_terms: tuple = ('dirac-coulomb',)
    schema_version: int = 1
    breit_orbital_energies_hex: tuple = ()
    breit_orbital_energies_bytes: bytes = b''

    def __init__(self, *args, **kwargs):
        raise TypeError('CIHamiltonianSnapshot must be built from public physical inputs')

    def __reduce_ex__(self, protocol):
        raise TypeError('CIHamiltonianSnapshot is local; archive physical inputs and rebuild')

    def identity_payload(self):
        """Return a fresh serializable header, separate from matrix storage."""
        payload = dict(schema_version=self.schema_version, representation=self.representation,
                       units=self.units, hamiltonian_terms=self.hamiltonian_terms,
                       orbital_keys=self.orbital_keys, orbital_basis_keys=self.orbital_basis_keys,
                       CSFs=self.csf_identities, max_determinants=self.max_determinants,
                       radial_binding=asdict(self.radial_binding))
        # Keep the established DC schema/header exactly unchanged.
        if self.hamiltonian_terms != ('dirac-coulomb',):
            operator = dict(mode=self.hamiltonian_terms[1],
                            finite_replaces_static_breit=bool(self.breit_orbital_energies_hex))
            if self.breit_orbital_energies_hex:
                from atomx.physics.csf.transverse import frequency_provenance

                energies = {key: float.fromhex(value) for key, value in self.breit_orbital_energies_hex}
                operator.update(frequency_provenance(energies))
                operator.update(
                    orbital_energies_hex=self.breit_orbital_energies_hex,
                    orbital_energies_encoding='IEEE-754 binary64 little-endian, sorted orbital keys',
                    orbital_energies_binary64_le_hex=self.breit_orbital_energies_bytes.hex(),
                    orbital_energies_binary64_le_sha256=hashlib.sha256(
                        self.breit_orbital_energies_bytes).hexdigest())
            payload['transverse_operator'] = operator
        return payload

    def matrix_for(self, csfs, orbitals, orbital_basis, workspace, Z, ns, *,
                   nuclear_potential=None, max_determinants=None,
                   include_breit=False, breit_orbital_energies=None,
                   one_body_shifts=None, one_body_operator=None):
        """Return owned H only for the same complete physical input binding."""
        operator = _coverage(orbitals, include_breit, breit_orbital_energies,
                             one_body_shifts, one_body_operator)
        if operator != (self.hamiltonian_terms, self.breit_orbital_energies_hex,
                        self.breit_orbital_energies_bytes):
            raise ValueError('CI Hamiltonian snapshot operator/frequency mismatch')
        _representation(workspace, Z, ns)
        keys, basis = tuple(sorted(_keys(orbitals))), _keys(orbital_basis)
        if (keys != self.orbital_keys or basis != self.orbital_basis_keys
                or _model(csfs) != self.csf_identities or _cap(max_determinants) != self.max_determinants):
            raise ValueError('CI Hamiltonian snapshot signed basis/keys/cap mismatch')
        self.radial_binding.validate(keys, orbitals, workspace, int(Z), int(ns), nuclear_potential)
        return self.matrix_Ha


def build_ci_hamiltonian_snapshot(csfs, orbitals, orbital_basis, workspace, Z, ns, *,
                                  nuclear_potential=None, convol_cache=None,
                                  max_determinants=None, radial_snapshot=None,
                                  include_breit=False, breit_orbital_energies=None,
                                  one_body_shifts=None, one_body_operator=None):
    """Build one complete B-spline DC/static/finite H through the public builder.

    No matrix input, symmetrization, coordinate repair, truncation or operator
    substitution is accepted. Validate input contents before and after build.
    Concurrent mutations during a mutable-input build are unsupported; use
    the existing owned ``CIRadialSnapshot.solve`` for immutable radial inputs.
    """
    from atomx.physics.csf.matrix import build_ci_matrix

    operator = _coverage(orbitals, include_breit, breit_orbital_energies,
                         one_body_shifts, one_body_operator)
    energies = ({key: float.fromhex(value) for key, value in operator[1]}
                if operator[1] else None)
    _representation(workspace, Z, ns)
    keys, basis, identities = tuple(sorted(_keys(orbitals))), _keys(orbital_basis), _model(csfs)
    cap = _cap(max_determinants)
    binding = CIReferenceBinding.capture([(*orbitals[k], k[1]) for k in keys], {},
                                         workspace, int(Z), int(ns), nuclear_potential)
    h = build_ci_matrix(csfs, orbitals, orbital_basis, workspace, Z, ns,
                        nuclear_potential=nuclear_potential, convol_cache=convol_cache,
                        max_determinants=max_determinants, radial_snapshot=radial_snapshot,
                        include_breit=include_breit, breit_orbital_energies=energies)
    if (tuple(sorted(_keys(orbitals))) != keys or _keys(orbital_basis) != basis
            or _model(csfs) != identities):
        raise ValueError('CI Hamiltonian snapshot inputs changed during build')
    if (_coverage(orbitals, include_breit, breit_orbital_energies,
                  one_body_shifts, one_body_operator) != operator
            or _coverage(orbitals, include_breit, energies,
                         one_body_shifts, one_body_operator) != operator):
        raise ValueError('CI Hamiltonian snapshot operator/frequency inputs changed during build')
    binding.validate(keys, orbitals, workspace, int(Z), int(ns), nuclear_potential)
    if (not isinstance(h, np.ndarray) or h.dtype != np.dtype(np.float64)
            or h.shape != (len(identities), len(identities)) or not np.isfinite(h).all()):
        raise ValueError('retained CI Hamiltonian requires the original finite float64 dense matrix')
    storage = h.tobytes(order='C')
    matrix = np.frombuffer(storage, dtype=np.float64).reshape(h.shape)
    snapshot = object.__new__(CIHamiltonianSnapshot)
    values = dict(matrix_Ha=matrix, radial_binding=binding, orbital_keys=keys,
                  orbital_basis_keys=basis, csf_identities=identities, max_determinants=cap,
                  representation='dense-real-full-CSF', units='Hartree',
                  hamiltonian_terms=operator[0], schema_version=1 if not include_breit else 2,
                  breit_orbital_energies_hex=operator[1], breit_orbital_energies_bytes=operator[2])
    for name, value in values.items():
        object.__setattr__(snapshot, name, value)
    payload = json.dumps(snapshot.identity_payload(), sort_keys=True, separators=(',', ':')).encode()
    object.__setattr__(snapshot, 'input_sha256', hashlib.sha256(payload).hexdigest())
    matrix_frame = repr((h.shape, h.dtype.str)).encode() + storage
    object.__setattr__(snapshot, 'matrix_sha256', hashlib.sha256(matrix_frame).hexdigest())
    return snapshot
