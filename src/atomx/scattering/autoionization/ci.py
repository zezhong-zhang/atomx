"""Explicit common-orbital CI Auger partial widths with full signed h+V."""

import hashlib
import json
from functools import lru_cache

import numpy as np

from atomx.angular.clebsch import clebsch_gordan
from atomx.angular.csf_hamiltonian import continuum_hamiltonian_terms, two_body_angular
from atomx.constants import AU_RATE_PER_S, HARTREE_EV
from atomx.numerics.bspline.dirac_assembly import apply_dirac_hamiltonian
from atomx.numerics.bspline.slater import AVAILABLE_METHODS, slater_rk
from atomx.numerics.continuum_actions import ContinuumFockAction
from atomx.scattering.dielectronic_recombination.solver import (
    dielectronic_capture_rate_cm3_s,
)
from atomx.scattering.photoionization.ci import ExplicitCIPhotoionization


class ExplicitCIAutoionization:
    """Partial outgoing widths to explicitly supplied adjacent CI eigenstates.

    The bound eigenstates, energy zero, nuclear operator, orbital layout and
    final-ion projected Hartree waves are shared with ExplicitCIPhotoionization.
    Every retained signed h_D and Coulomb amplitude is summed before squaring.
    Missing final-ion levels remain missing channels, not zero widths.
    """

    def __init__(self, snapshot, *, radial_method='poisson', max_pair_products=2000000,
                 engine='magnetic-pairs', max_kernel_steps=100000000):
        if not isinstance(snapshot, ExplicitCIPhotoionization) or radial_method not in AVAILABLE_METHODS:
            raise ValueError("explicit native CI snapshot and supported radial integral method required")
        if isinstance(max_pair_products, bool) or not isinstance(max_pair_products, int) or max_pair_products < 1:
            raise ValueError("positive integer angular pair-product cap required")
        if (engine not in ('magnetic-pairs', 'compiled-reduced') or isinstance(max_kernel_steps, bool)
                or not isinstance(max_kernel_steps, int) or max_kernel_steps < 1):
            raise ValueError("explicit magnetic-pairs/compiled-reduced engine and positive kernel work cap required")
        if snapshot.magnetic_mode == 'highest-all' and engine != 'compiled-reduced':
            raise ValueError("highest-all snapshots require exact coupled scalar reduction")
        self.snapshot, self.radial_method, self.max_pair_products = snapshot, radial_method, max_pair_products
        self.engine, self.max_kernel_steps = engine, max_kernel_steps
        self.source_id = hashlib.sha256(json.dumps(dict(snapshot=snapshot.source_id,
                     process='complete signed h_D+Coulomb amplitude', radial_method=radial_method, engine=engine), sort_keys=True).encode()).hexdigest()
        nuclear = snapshot._nuclear_values.reshape(snapshot.data['gr'].shape)
        self._h_bound = {key: tuple(value.ravel() for value in apply_dirac_hamiltonian(
                         *snapshot.orbitals[key], key[1], snapshot.data, potential=nuclear)) for key in snapshot.keys}
        self._angular = {}
        self._actions = {}

    def partial_width(self, initial, final):
        """Return unpolarized partial Gamma [s^-1], signed components and h/V ledger.

        Components carry amplitudes in sqrt(s^-1). Full magnetic rows are
        averaged over represented initial M; coupled scalar channel rows
        already contain the CG reduction and are summed without another g.
        Zero total amplitudes with nonzero h or V
        remain in the ledger. No spectroscopic factor or occupation is added.
        """
        s = self.snapshot
        # Reuse checked indices/adjacency; Auger has the opposite energy sign
        # to positive bound PI thresholds, so it needs its own opening test.
        if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= i < len(s.levels)
               for i in (initial, final)):
            raise ValueError("explicit in-range CI level indices required")
        a, b = s.levels[initial], s.levels[final]
        if a.electrons != b.electrons+1:
            raise ValueError("Auger requires adjacent electron sectors")
        energy = a.energy_Ha-b.energy_Ha
        kappas = []
        for je in range(abs(a.two_J-b.two_J), a.two_J+b.two_J+1, 2):
            if je % 2 != 1:
                continue
            for k in (-(je+1)//2, (je+1)//2):
                orbital_l = k if k > 0 else -k-1
                if b.parity*(-1)**orbital_l == a.parity:
                    kappas.append(k)
        if self.engine == 'compiled-reduced':
            return self._compiled_width(initial, final, energy, kappas)
        rows = []
        plan = [(k, me, mf, mi) for k in kappas for me in range(-2*abs(k)+1, 2*abs(k), 2)
                for mf in s._vectors[final] for mi in s._vectors[initial] if mf+me == mi]
        products = sum(len(s._vectors[final][mf])*len(s._vectors[initial][mi]) for _, _, mf, mi in plan)
        if products > self.max_pair_products:
            raise MemoryError("complete Auger angular pair-product cap exceeded; no terms screened")
        if energy > 0:
            s.prepare_continuum_waves(final, [(k, energy) for k in kappas])
            for k in kappas:
                p, q = s._waves[final, k, energy]
                continuum_key = (-1, k)
                waves = dict(s.bound)
                waves[continuum_key] = p, q

                @lru_cache(maxsize=None)
                def evaluate(key, k=k, p=p, q=q, continuum_key=continuum_key, waves=waves):
                    if key[0] == 'h':
                        _, na, ka, nb = key
                        if na != -1 or ka != k:
                            raise ValueError("incorrect continuum one-body role")
                        hp, hq = self._h_bound[nb, ka]
                        return float(np.sum(s.weights*(p*hp+q*hq)))
                    _, rank, aa, c, bb, d = key
                    if continuum_key in (aa, c):
                        aa, c, bb, d = bb, d, aa, c
                    pa, qa = waves[aa]
                    pc, qc = waves[c]
                    pb, qb = waves[bb]
                    pd, qd = waves[d]
                    return float(slater_rk(pa*pc+qa*qc, pb*pd+qb*qd, s.data, rank, method=self.radial_method))

                for kk, me, mf, mi in plan:
                    if kk != k:
                        continue
                    token = initial, final, k, me, mf, mi
                    if token not in self._angular:
                        self._angular[token] = continuum_hamiltonian_terms(
                            s._vectors[final][mf], s._vectors[initial][mi], s.spinors, k, me)
                    h, v = 0., 0.
                    for key, angular in self._angular[token]:
                        value = angular*evaluate(key)
                        if key[0] == 'h':
                            h += value
                        else:
                            v += value
                    if h != 0 or v != 0:
                        scale = np.sqrt(2*np.pi*AU_RATE_PER_S)
                        rows.append(dict(kappa=k, electron_m2=me, final_m2=mf, initial_m2=mi,
                                         h_sqrt_per_s=float(h*scale), V_sqrt_per_s=float(v*scale)))
        # A scalar width summed over the full final-ion/electron multiplet
        # is identical for each initial M. Highest-weight evaluation is an
        # exact alternative to the complete unpolarized average, not a cut.
        denominator = len(s._vectors[initial])
        if len(s._vectors[final]) != b.g or denominator not in (1, a.g):
            raise ValueError("scalar Auger widths require a complete final multiplet and complete/highest initial mode")
        gh = sum(r['h_sqrt_per_s']**2 for r in rows)/denominator
        gv = sum(r['V_sqrt_per_s']**2 for r in rows)/denominator
        interference = sum(2*r['h_sqrt_per_s']*r['V_sqrt_per_s'] for r in rows)/denominator
        gamma = sum((r['h_sqrt_per_s']+r['V_sqrt_per_s'])**2 for r in rows)/denominator
        return dict(initial=initial, final=final, electron_energy_eV=float(energy*HARTREE_EV),
                    open=energy > 0, partial_rate_per_s=float(gamma), h_only_per_s=float(gh),
                    V_only_per_s=float(gv), interference_per_s=float(interference),
                    angular_pair_products=products, components=rows, source_id=self.source_id,
                    final_channel_scope=s.coverage, total_width_certified=False)

    def _action(self, initial, final):
        """Exact normal ordering of common occupied spinors and owned vectors."""
        key = initial, final
        if key not in self._actions:
            s = self.snapshot
            vi = s._vectors[initial][s.levels[initial].two_J]
            vf = s._vectors[final][s.levels[final].two_J]
            frozen = next(iter(vi))
            for mask in (*vi, *vf):
                frozen &= mask
            frozen_bits = [i for i in range(len(s.spinors)) if frozen&(1 << i)]
            active_bits = [i for i in range(len(s.spinors)) if not frozen&(1 << i)]
            if len(active_bits) > 63:
                raise MemoryError('compiled continuum action exceeds63 active spinors; no orbital removed')
            def compress(vector):
                masks, values = [], []
                for mask, value in vector.items():
                    if mask&frozen != frozen:
                        raise ValueError('state opens a contracted continuum spectator')
                    swaps = sum((mask&~frozen&((1 << i)-1)).bit_count() for i in frozen_bits)
                    masks.append(sum(1 << j for j, i in enumerate(active_bits) if mask&(1 << i)))
                    values.append(value*(-1 if swaps%2 else 1))
                return np.array(masks,dtype=np.uint64), np.array(values)
            im, ci = compress(vi)
            fm, cf = compress(vf)
            n = len(active_bits)
            occupied = int(im[0]).bit_count()
            step_bound = len(im)*(occupied+occupied*(occupied-1)//2*n)
            if step_bound > self.max_kernel_steps:
                raise MemoryError('complete compiled continuum action exceeds declared kernel work cap')
            action = ContinuumFockAction(im, ci, fm, cf, n)
            self._actions[key] = action, tuple(s.spinors[i] for i in active_bits), tuple(s.spinors[i] for i in frozen_bits)
        return self._actions[key]

    def _compiled_width(self, initial, final, energy, kappas):
        s = self.snapshot
        a, b = s.levels[initial], s.levels[final]
        action, labels, core = self._action(initial, final)
        rows = []
        if energy > 0:
            s.prepare_continuum_waves(final, [(k, energy) for k in kappas])
            me = a.two_J-b.two_J
            for k in kappas:
                je = 2*abs(k)-1
                cg = float(clebsch_gordan(b.two_J, b.two_J, je, me, a.two_J, a.two_J))
                if cg == 0:
                    raise ValueError('highest-weight scalar channel has zero Clebsch-Gordan coefficient')
                p, q = s._waves[final, k, energy]
                continuum = (-1, k)
                waves = dict(s.bound)
                waves[continuum] = p, q
                electron = (-1, k, me)
                @lru_cache(maxsize=None)
                def radial(rank, aa, cc, bb, dd, continuum=continuum, waves=waves):
                    if continuum in (aa, cc):
                        aa, cc, bb, dd = bb, dd, aa, cc
                    pa, qa = waves[aa]
                    pc, qc = waves[cc]
                    pb, qb = waves[bb]
                    pd, qd = waves[dd]
                    return float(slater_rk(pa*pc+qa*qc,pb*pd+qb*qd,s.data,rank,method=self.radial_method))
                def direct(aa, bb, cc, dd, radial=radial):
                    return sum(v*radial(rank,aa[:2],cc[:2],bb[:2],dd[:2]) for rank,v in
                              two_body_angular(aa[1],aa[2],bb[1],bb[2],cc[1],cc[2],dd[1],dd[2]))
                def eri(aa,bb,cc,dd,direct=direct):
                    if aa[2]+bb[2] != cc[2]+dd[2]:
                        return 0.
                    return direct(aa,bb,cc,dd)-direct(aa,bb,dd,cc)
                n = len(labels)
                h, frozen_row, g = np.zeros(n),np.zeros(n),np.zeros((n,n,n))
                for i, label in enumerate(labels):
                    if label[1:] == electron[1:]:
                        hp,hq = self._h_bound[label[:2]]
                        h[i] = float(np.sum(s.weights*(p*hp+q*hq)))
                    frozen_row[i] = sum(eri(electron,c,label,c) for c in core)
                    for j in range(n):
                        for r in range(j+1,n):
                            value = eri(electron,label,labels[j],labels[r])
                            g[i,j,r],g[i,r,j] = value,-value
                ah,av = action.amplitudes(h,frozen_row,g)
                scale = np.sqrt(2*np.pi*AU_RATE_PER_S)/cg
                if ah != 0 or av != 0:
                    rows.append(dict(kappa=k,electron_m2=me,final_m2=b.two_J,initial_m2=a.two_J,
                         Clebsch_Gordan=cg,amplitude_basis='coupled scalar channel, unit CG sum',
                         h_sqrt_per_s=float(ah*scale),V_sqrt_per_s=float(av*scale)))
        gh=sum(r['h_sqrt_per_s']**2 for r in rows)
        gv=sum(r['V_sqrt_per_s']**2 for r in rows)
        cross=sum(2*r['h_sqrt_per_s']*r['V_sqrt_per_s'] for r in rows)
        gamma=sum((r['h_sqrt_per_s']+r['V_sqrt_per_s'])**2 for r in rows)
        return dict(initial=initial,final=final,electron_energy_eV=float(energy*HARTREE_EV),open=energy>0,
              partial_rate_per_s=float(gamma),h_only_per_s=float(gh),V_only_per_s=float(gv),interference_per_s=float(cross),
              kernel_step_bound=action.step_bound,components=rows,source_id=self.source_id,
              final_channel_scope=s.coverage,total_width_certified=False)

    def inverse_capture_cm3_s(self, initial, final, temperature_eV):
        """Unbranched narrow-resonance inverse of this same partial Auger width.

        No radiative stabilisation factor: a kinetic model must supply every
        competing lifetime separately. Narrow/isolation validity and omitted
        outgoing channels are caller-side physical convergence requirements.
        """
        width = self.partial_width(initial, final)
        if not width['open']:
            return 0.
        return dielectronic_capture_rate_cm3_s(temperature_eV, width['electron_energy_eV'],
                    width['partial_rate_per_s'], g_resonance=self.snapshot.levels[initial].g,
                    g_continuum=self.snapshot.levels[final].g)
