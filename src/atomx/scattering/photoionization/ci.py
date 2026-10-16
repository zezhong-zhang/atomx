"""Explicit adjacent CI sectors in a common frozen-orbital PI model.

Consumes complete native blocked Dirac-Coulomb spectra, including closed
spectators. No residual Fock space is inferred. The continuum uses the final
level's spherical direct Hartree field, with projection against every supplied
bound orbital. Nonlocal exchange, interchannel coupling, Auger and EII are
outside this operator. Omitted daughter configurations are missing channels.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from math import comb

import numpy as np

from atomx.angular.clebsch import clebsch_gordan
from atomx.angular.coefficients import ckm
from atomx.angular.csf_hamiltonian import (
    csf_signature,
    pattern_m_counts,
)
from atomx.angular.dyson import (
    dyson_annihilation,
    reduced_dyson_annihilation,
    scalar_radial_density,
)
from atomx.angular.one_body_tensor import magnetic_csf_projection
from atomx.config.csf import get_shared_orbital_basis
from atomx.constants import (
    ALPHA,
    BOHR_RADIUS_M,
    CODATA,
    DEFAULT_CODATA_VERSION,
    HARTREE_EV,
)
from atomx.methods.ci_blocks import BlockedCISpectrum
from atomx.numerics.bspline.evaluate import eval_P, eval_Q
from atomx.numerics.radial_workspace import make_grid_workspace
from atomx.numerics.shooting.dirac_continuum import solve_dirac_continuum_many
from atomx.numerics.yk import compute_yk
from atomx.physics.csf.radial_snapshot import _freeze
from atomx.properties.transitions.multipole import radial_eL
from atomx.scattering.models import ContinuumModel


def _owned(a):
    a = np.asarray(a)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True)
class CIPhotoLevel:
    electrons: int
    two_J: int
    parity: int
    root: int
    energy_Ha: float
    state_id: str

    @property
    def g(self):
        return self.two_J + 1


class ExplicitCIPhotoionization:
    """Signed, unpolarized, finite-frequency E1 partial PI cross sections.

    ``sectors`` maps total electron number to ``(csfs, BlockedCISpectrum)``.
    Every supplied block root is retained by default. ``state_selection`` can
    explicitly request (2J, parity, root) tuples for every sector; complete
    exactly degenerate subspaces and every selected state's M partners remain.
    Inputs must share the same native
    orbital/geometry/nuclear Hamiltonian binding; absolute CI energies define
    every threshold. State indices refer only to this object's ``levels``.
    Cost caps reject before magnetic expansion or continuum allocation.

    ``dipole_mode='reduced-dyson'`` exactly reconstructs the complete E1
    magnetic trace from highest pure-J components. It permits highest-only
    snapshots while preserving all signed radial interferences and true g
    factors. The default ``'complete'`` explicitly sums supplied M vectors.
    ``dyson_engine='compiled'`` selects exact finite bitset removal/support
    kernels for at most63 magnetic spinors; the default remains Python.
    This numerical engine is recorded in the plan, with unchanged physical
    source identity and explicit source hashes in measurement workflows.
    """

    def __init__(self, sectors, orbitals, workspace, Z, *, nuclear_potential,
                 continuum_model, max_levels=128, max_magnetic_entries=100000,
                 max_waves=5000, coverage, state_selection=None,
                 magnetic_mode='complete', projection_engine='csr',
                 dipole_mode='complete', dyson_engine='python'):
        if magnetic_mode not in ('complete', 'highest-parent', 'highest-all') or projection_engine not in ('csr', 'streamed'):
            raise ValueError("explicit magnetic mode and CSR/streamed projection required")
        if dipole_mode not in ('complete', 'reduced-dyson'):
            raise ValueError("explicit complete or reduced-dyson dipole mode required")
        if dyson_engine not in ('python', 'compiled'):
            raise ValueError('explicit python or compiled Dyson engine required')
        if (not isinstance(continuum_model, ContinuumModel) or continuum_model.method != 'shooting'
                or continuum_model.potential is not None or continuum_model.orbital_removal != 'none'
                or continuum_model.shooting_n_grid < 1000 or not coverage):
            raise ValueError("explicit final-ion shooting model and channel coverage required")
        for cap in (max_levels, max_magnetic_entries, max_waves):
            if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
                raise ValueError("positive integer resource caps required")
        if (not sectors or any(not isinstance(n, int) or not 1 <= n <= Z for n in sectors)
                or any(n - 1 not in sectors for n in sectors if n != min(sectors))):
            raise ValueError("explicit contiguous positive-electron CI sectors required")
        selected = {}
        if state_selection is not None and set(state_selection) != set(sectors):
            raise ValueError("explicit state selection required for every supplied sector")
        for n, (_, spec) in sectors.items():
            if not isinstance(spec, BlockedCISpectrum):
                raise ValueError("native blocked CI spectra required")
            available = {(b.sector[0], b.sector[1], r) for b in spec.blocks for r in range(len(b.indices))}
            tokens = available if state_selection is None else set(state_selection[n])
            if (not tokens or not tokens <= available or (state_selection is not None and len(tokens) != len(state_selection[n]))):
                raise ValueError("unique, present, nonempty selected CI root identities required")
            for b in spec.blocks:
                roots = [r for r in range(len(b.indices)) if (*b.sector, r) in tokens]
                excluded = [r for r in range(len(b.indices)) if (*b.sector, r) not in tokens]
                if roots and excluded and np.any(abs(b.energies_Ha[roots, None]-b.energies_Ha[excluded]) < 1e-10):
                    raise ValueError("state selection splits an exactly degenerate subspace")
            selected[n] = tokens
        nlev = sum(map(len, selected.values()))
        if nlev > max_levels:
            raise MemoryError("complete supplied CI level inventory exceeds cap")
        # Conservative count: each state can use every determinant of each
        # configuration pattern in its block, for every magnetic partner.
        entries, projection_entries = 0, 0
        for n, (csfs, spectrum) in sectors.items():
            if not isinstance(spectrum, BlockedCISpectrum) or any(c.n_electrons != n for c in csfs):
                raise ValueError("matched complete native blocked CI sectors required")
            if tuple(csf_signature(c).key for c in csfs) != spectrum.csf_identities:
                raise ValueError("CI basis identity or ordering mismatch")
            keys = get_shared_orbital_basis(csfs)
            spectrum.radial_binding.validate(keys, orbitals, workspace, Z, workspace['ns'], nuclear_potential)
            all_indices = [int(i) for block in spectrum.blocks for i in block.indices]
            if sorted(all_indices) != list(range(len(csfs))):
                raise ValueError("every CI basis coordinate must occur exactly once")
            for block in spectrum.blocks:
                patterns = {csf_signature(csfs[i]).pattern for i in block.indices}
                dets = sum(np.prod([comb(2*abs(k), q) for _, k, q in pattern], dtype=object)
                           for pattern in patterns)
                count = sum((*block.sector, r) in selected[n] for r in range(len(block.indices)))
                magnetic = ((block.sector[0],) if magnetic_mode == 'highest-all' or (magnetic_mode == 'highest-parent' and n == max(sectors))
                            else range(-block.sector[0], block.sector[0]+1, 2))
                entries += count*sum(sum(dict(pattern_m_counts(p)).get(m, 0) for p in patterns)
                                     for m in magnetic)
                if count:
                    if projection_engine == 'csr':
                        projection_entries += sum(sum(dict(pattern_m_counts(csf_signature(csfs[i]).pattern)).get(m, 0)
                                                  for i in block.indices) for m in magnetic)
                    else:
                        projection_entries += sum(count*sum(dict(pattern_m_counts(p)).get(m, 0) for p in patterns)
                                                  for m in magnetic)
        if max(entries, projection_entries) > max_magnetic_entries:
            raise MemoryError("complete signed magnetic expansion exceeds cap")
        self.plan = dict(levels=nlev, magnetic_entry_bound=entries, max_waves=max_waves,
                         CSF_projection_entry_bound=projection_entries,
                         wave_storage_bound_bytes=16*len(workspace['gr'].ravel())*max_waves)
        self.Z, self.model, self.coverage = int(Z), continuum_model, str(coverage)
        self.magnetic_mode, self.projection_engine = magnetic_mode, projection_engine
        self.dipole_mode = dipole_mode
        self.dyson_engine = dyson_engine
        self.data, self.orbitals = _freeze(workspace), _freeze(orbitals)
        self.keys = tuple(sorted(set().union(*(set(get_shared_orbital_basis(csfs)) for csfs, _ in sectors.values()))))
        self.spinors = tuple((n, k, m) for n, k in self.keys
                            for m in range(-2*abs(k)+1, 2*abs(k), 2))
        if dyson_engine == 'compiled' and len(self.spinors) > 63:
            raise MemoryError('compiled removal exceeds63 magnetic spinors; no orbital removed')
        self.plan['dyson_engine'] = dyson_engine
        self.radius, self.weights = self.data['gr'].ravel(), self.data['grw'].ravel()
        self.workspace = make_grid_workspace(self.radius, self.weights)
        self.bound = {key: (_owned(eval_P(p, self.data).ravel()), _owned(eval_Q(q, self.data).ravel()))
                      for key, (p, q) in self.orbitals.items() if key in self.keys}
        gram = np.array([[np.sum(self.weights*(self.bound[a][0]*self.bound[b][0]
                         + self.bound[a][1]*self.bound[b][1])) if a[1] == b[1] else 0.
                          for b in self.keys] for a in self.keys])
        if not np.isfinite(gram).all() or np.max(abs(gram-np.eye(len(self.keys)))) > 1e-7:
            raise ValueError("common CI orbitals must be orthonormal")
        self._projectors = {}
        for k in {key[1] for key in self.keys}:
            indices = [i for i, key in enumerate(self.keys) if key[1] == k]
            self._projectors[k] = (tuple(self.bound[self.keys[i]] for i in indices), gram[np.ix_(indices, indices)])
        self.levels, self._vectors, densities = [], [], []
        digest = hashlib.sha256()
        digest.update(json.dumps(dict(Z=Z, model=asdict(continuum_model), coverage=coverage), sort_keys=True).encode())
        if magnetic_mode != 'complete':
            digest.update(magnetic_mode.encode())
        if dipole_mode != 'complete':
            digest.update(dipole_mode.encode())
        nuclear = (-Z/self.radius if nuclear_potential is None
                   else np.array([nuclear_potential(r) for r in self.radius]))
        # The off-grid nuclear model cannot be reconstructed from quadrature
        # values. Keep its declared callable and detect mutation on each field
        # evaluation against these integration-node values.
        self._nuclear, self._nuclear_values = nuclear_potential, _owned(nuclear)
        for n, (csfs, spectrum) in sorted(sectors.items()):
            digest.update(repr((n, spectrum.radial_binding, spectrum.csf_identities)).encode())
            for block in spectrum.blocks:
                h, c, e = block.hamiltonian_Ha, block.coefficients, block.energies_Ha
                dim = len(block.indices)
                roots = [r for r in range(dim) if (*block.sector, r) in selected[n]]
                if not roots:
                    continue
                cr = c if len(roots) == dim else c[:, roots]
                er = e if len(roots) == dim else e[roots]
                if (h.shape != (dim, dim) or c.shape != (dim, dim) or e.shape != (dim,)
                        or not all(np.isrealobj(x) and np.isfinite(x).all() for x in (h, c, e))
                        or np.max(abs(h-h.T)) > 1e-12 or np.max(abs(cr.T@cr-np.eye(len(roots)))) > 1e-10
                        or np.max(abs(h@cr-cr*er)) > 1e-8
                        or any((csfs[i].total_J2, csfs[i].parity) != block.sector for i in block.indices)):
                    raise ValueError("CI block residual, metric or symmetry mismatch")
                digest.update(h.tobytes())
                digest.update(cr.tobytes())
                digest.update(er.tobytes())
                if state_selection is not None:
                    digest.update(repr(roots).encode())
                subset = [csfs[i] for i in block.indices]
                sig = tuple(csf_signature(csf) for csf in subset)
                j, parity = block.sector
                magnetic = ((j,) if magnetic_mode == 'highest-all' or (magnetic_mode == 'highest-parent' and n == max(sectors)) else range(-j, j+1, 2))
                projected = {}
                for m in magnetic:
                    dets, _, amplitudes = magnetic_csf_projection(sig, m, self.keys, cr, engine=projection_engine)
                    projected[m] = (dets, amplitudes)
                for root in roots:
                    energy = e[root]
                    self.levels.append(CIPhotoLevel(n, j, parity, root, float(energy),
                                                    f'N{n}:J2{j}:p{parity}:r{root}'))
                    vectors = {}
                    for m in magnetic:
                        dets, amplitudes = projected[m]
                        vector = {mask: value for mask, value in zip(dets, amplitudes[:, roots.index(root)], strict=True)
                                  if value != 0}
                        if abs(sum(v*v for v in vector.values())-1) > 1e-10:
                            raise ValueError("signed CI magnetic norm failed")
                        vectors[m] = vector
                    self._vectors.append(vectors)
                    # Only lower sectors can be final ions. A parent-only
                    # density is never used by these adjacent loss operators.
                    if n == max(sectors):
                        densities.append(None)
                        continue
                    density_keys, rho = scalar_radial_density(vectors[j], self.spinors)
                    if density_keys != self.keys:
                        raise ValueError("scalar density orbital layout mismatch")
                    rho = (rho+rho.T)*.5
                    if abs(np.trace(rho)-n) > 1e-9:
                        raise ValueError("final CI density has wrong electron number")
                    densities.append(_owned(rho))
        self.levels, self._densities = tuple(self.levels), tuple(densities)
        self.source_id = digest.hexdigest()
        self._fields, self._waves, self._angular, self._removal = {}, {}, {}, {}
        self.projection_diagnostics = []

    def threshold_eV(self, initial, final):
        if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= i < len(self.levels)
               for i in (initial, final)):
            raise ValueError("explicit in-range CI level indices required")
        a, b = self.levels[initial], self.levels[final]
        if a.electrons != b.electrons+1:
            raise ValueError("PI requires adjacent electron sectors")
        threshold = (b.energy_Ha-a.energy_Ha)*HARTREE_EV
        if threshold <= 0:
            raise ValueError("PI bound state requires positive matched CI threshold; resonances need Auger")
        return float(threshold)

    def _field(self, final):
        if final not in self._fields:
            rho = self._densities[final]
            if rho is None:
                raise ValueError("no supplied parent sector for this final-ion field")
            density = np.zeros_like(self.radius)
            for i, a in enumerate(self.keys):
                for j, b in enumerate(self.keys):
                    if a[1] == b[1] and rho[i, j] != 0:
                        density += rho[i, j]*(self.bound[a][0]*self.bound[b][0]+self.bound[a][1]*self.bound[b][1])
            if abs(np.sum(self.weights*density)-self.levels[final].electrons) > 1e-7:
                raise ValueError("final-ion radial density normalization failed")
            hartree = compute_yk(density.reshape(self.data['gr'].shape), self.data['gr'], self.data['grw'],
                                 self.data['nv'], self.data['gr'].shape[1], 0).ravel()
            number = self.levels[final].electrons

            def field(r):
                r = np.asarray(r)
                direct = np.interp(r, self.radius, hartree)
                direct = np.where(r > self.radius[-1], number/r, direct)
                attraction = (-self.Z/r if self._nuclear is None else np.vectorize(self._nuclear, otypes=[float])(r))
                return attraction+direct

            self._fields[final] = field
        return self._fields[final]

    def _angular_tensor(self, initial, final, kappa):
        token = initial, final, kappa
        if token not in self._angular:
            rows = []
            if self.dipole_mode == 'reduced-dyson':
                ji, jf = self.levels[initial].two_J, self.levels[final].two_J
                if (initial, final) not in self._removal:
                    keys, coefficients = reduced_dyson_annihilation(
                        self._vectors[final][jf], self._vectors[initial][ji],
                        self.spinors, ji, jf, engine=self.dyson_engine)
                    if keys != self.keys:
                        raise ValueError('reduced Dyson radial layout mismatch')
                    self._removal[initial, final] = _owned(coefficients)
                coefficients = self._removal[initial, final]
                for me in range(-2*abs(kappa)+1, 2*abs(kappa), 2):
                    for mf in range(-jf, jf+1, 2):
                        for mi in range(-ji, ji+1, 2):
                            if me+mf != mi:
                                continue
                            row = np.zeros(len(self.keys))
                            for i, (_, k) in enumerate(self.keys):
                                j = 2*abs(k)-1
                                if abs(me) <= j and coefficients[i] != 0:
                                    row[i] = (coefficients[i]
                                        * float(clebsch_gordan(jf, mf, j, me, ji, mi))
                                        * ckm(kappa, me, k, me, 1, 0))
                            rows.append(row)
                self._angular[token] = _owned(np.asarray(rows).reshape(-1, len(self.keys)))
                return self._angular[token]
            for me in range(-2*abs(kappa)+1, 2*abs(kappa), 2):
                for mf, bra in self._vectors[final].items():
                    for mi, ket in self._vectors[initial].items():
                        if me+mf != mi:  # q=0; unpolarized initial multiplet
                            continue
                        dyson = dyson_annihilation(bra, ket, len(self.spinors), engine=self.dyson_engine)
                        row = np.zeros(len(self.keys))
                        lookup = {key: i for i, key in enumerate(self.keys)}
                        for value, (n, k, m) in zip(dyson, self.spinors, strict=True):
                            if value != 0:
                                row[lookup[n, k]] += value*ckm(kappa, me, k, m, 1, 0)
                        rows.append(row)
            self._angular[token] = _owned(np.asarray(rows).reshape(-1, len(self.keys)))
        return self._angular[token]

    def cross_section_cm2(self, initial, final, photon_energies_eV):
        """Partial sigma averaged over initial M and summed over final M/electron.

        Scalar or 1-D energies are accepted; below/at threshold returns zero.
        No interpolation, threshold substitution or extrapolation is performed.
        """
        threshold = self.threshold_eV(initial, final)
        photons = np.asarray(photon_energies_eV, dtype=float)
        if photons.ndim > 1 or not np.isfinite(photons).all() or np.any(photons <= 0):
            raise ValueError("positive finite scalar or 1-D photon energies required")
        return self._cross_section(initial, final, photons, (photons-threshold)/HARTREE_EV)

    def _cross_section(self, initial, final, photons, kinetic_Ha):
        from atomx.scattering.radiative_recombination import _final_e1_kappas

        if self.dipole_mode == 'complete' and (
                len(self._vectors[initial]) != self.levels[initial].g
                or len(self._vectors[final]) != self.levels[final].g):
            raise ValueError("unpolarized PI/RR requires complete initial magnetic partners")
        flat, out = photons.reshape(-1), np.zeros(photons.size)
        kappas = sorted({k for _, bound_k in self.keys for k in _final_e1_kappas(bound_k)
                         if np.any(self._angular_tensor(initial, final, k))})
        active = [(i, float(energy)) for i, energy in enumerate(np.asarray(kinetic_Ha).reshape(-1)) if energy > 0]
        self.prepare_continuum_waves(final, [(k, energy) for _, energy in active for k in kappas])
        for i, energy in active:
            omega = float(flat[i]/HARTREE_EV)
            for k in kappas:
                p, q = self._waves[final, k, energy]
                angular = self._angular_tensor(initial, final, k)
                radial = np.array([radial_eL(*self.bound[key], p, q, self.workspace, 1,
                                            kappa_i=key[1], kappa_f=k, omega=omega)
                                   if np.any(angular[:, j]) else 0. for j, key in enumerate(self.keys)])
                out[i] += 4*np.pi**2*ALPHA*omega*(BOHR_RADIUS_M*100)**2*np.sum((angular@radial)**2)/self.levels[initial].g
        return float(out[0]) if photons.ndim == 0 else out

    def prepare_continuum_waves(self, final, requests):
        """Prepare exact (kappa, kinetic Hartree) waves in a selected final ion.

        Shared by PI and full h+V Auger consumers; no energy override or
        alternative field is introduced. Existing resource/Pauli gates apply.
        """
        if (isinstance(final, bool) or not isinstance(final, (int, np.integer)) or not 0 <= final < len(self.levels)
                or any(not isinstance(k, (int, np.integer)) or k == 0 or not np.isfinite(e) or e <= 0
                       for k, e in requests)):
            raise ValueError("valid final ion and positive finite continuum kinetic energies required")
        missing = list(dict.fromkeys((int(k), float(energy)) for k, energy in requests
                                    if (final, k, energy) not in self._waves))
        if len(self._waves)+len(missing) > self.plan['max_waves']:
            raise MemoryError("continuum wave cap exceeded; no physical channel removed")
        current_nuclear = (-self.Z/self.radius if self._nuclear is None
                           else np.array([self._nuclear(r) for r in self.radius]))
        if not np.array_equal(current_nuclear, self._nuclear_values):
            raise ValueError("nuclear callable changed after CI snapshot")
        if missing:
            rmax = self.model.shooting_r_max or max(50., float(self.radius[-1])*1.01)
            if rmax < self.radius[-1]:
                raise ValueError("continuum box must contain the complete bound quadrature")
            _, waves = solve_dirac_continuum_many(missing, self._field(final), n_grid=self.model.shooting_n_grid,
                        r_max=rmax, Z_residual=self.Z-self.levels[final].electrons,
                        interp_grid=self.radius, workers=1)
            for k, energy in missing:
                p, q = waves[k, energy]
                basis, gram = self._projectors.get(k, ((), None))
                if basis:
                    overlap = np.array([np.sum(self.weights*(a*p+b*q)) for a, b in basis])
                    for coefficient, (a, b) in zip(np.linalg.solve(gram, overlap), basis, strict=True):
                        p -= coefficient*a
                        q -= coefficient*b
                    defect = float(max(abs(np.sum(self.weights*(a*p+b*q))) for a, b in basis))
                    if defect > 1e-10:
                        raise ValueError("continuum Pauli projection failed")
                    self.projection_diagnostics.append(defect)
                self._waves[final, k, energy] = (_owned(p), _owned(q))

    def recombination_moments(self, initial, final, temperature_eV, *, order=32):
        """Spontaneous Maxwellian Milne RR, directly on Gauss-Laguerre nodes.

        Returns alpha in cm3/s and photon moment in eV cm3/s. The continuum
        kinetic energy nodes are T*x, shared across initial levels to the same
        residual ion; no fitted PI tail or interpolated table is needed.
        Compare quadrature orders before using these partial rates.
        """
        if not np.isfinite(temperature_eV) or temperature_eV <= 0 or not isinstance(order, int) or not 2 <= order <= 128:
            raise ValueError("positive temperature and Gauss order2..128 required")
        from scipy.special import roots_laguerre

        threshold = self.threshold_eV(initial, final)
        x, w = roots_laguerre(order)
        photons = threshold+temperature_eV*x
        sigma = self._cross_section(initial, final, photons, temperature_eV*x/HARTREE_EV)
        constants = CODATA[DEFAULT_CODATA_VERSION]
        me = constants['_me']*1000
        ev = constants['_e']*1e7
        c = constants['_c']*100
        factor = (np.sqrt(8/(np.pi*me))*(temperature_eV*ev)**-1.5*ev**3
                  * (self.levels[initial].g/self.levels[final].g)/(2*me*c*c))
        return dict(alpha_cm3_s=float(factor*temperature_eV*np.dot(w, sigma*photons**2)),
                    photon_eV_cm3_s=float(factor*temperature_eV*np.dot(w, sigma*photons**3)), order=order)
