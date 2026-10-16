"""Partial spectator-ion DW EII from explicit signed adjacent CI eigenstates."""

import hashlib
import json

from atomx.angular.dyson import reduced_dyson_annihilation
from atomx.scattering.electron_ionization import _eii_dw_channels
from atomx.scattering.electron_ionization.channels import EIIDWBoundChannel
from atomx.scattering.photoionization.ci import ExplicitCIPhotoionization


class ExplicitCIElectronIonization:
    """Matched-threshold CI removal in an explicitly scoped DW approximation.

    The shared snapshot supplies complete signed CI vectors, both energies,
    orbitals, nucleus and the selected final-ion direct Hartree field. The
    incident and two outgoing electrons use that same field and Pauli
    projection against every supplied bound radial spinor. The retained
    interaction is instantaneous Coulomb with direct-minus-exchange amplitudes.
    Nonlocal distortion exchange, interchannel coupling, postcollision effects,
    transverse exchange and indirect excitation-autoionization are absent.

    An unpolarized complete ionic magnetic trace leaves a diagonal kappa/m
    removal density. Same-kappa radial Dyson amplitudes remain coherent and
    unnormalized. The shared one-electron DW prefactor is therefore 1/(2j+1)
    with occupation one, without another initial/final CI statistical factor.
    Partial-wave, sharing, radial-box and shooting convergence remain separate.
    """

    def __init__(self, snapshot, *, coverage, max_channel_work=2000000):
        if not isinstance(snapshot, ExplicitCIPhotoionization) or not isinstance(coverage, str) or not coverage:
            raise ValueError('explicit native adjacent CI snapshot and partial channel coverage required')
        if (isinstance(max_channel_work, bool) or not isinstance(max_channel_work, int)
                or max_channel_work < 1):
            raise ValueError('positive integer full DW channel-work cap required')
        self.snapshot, self.coverage = snapshot, coverage
        self.max_channel_work = max_channel_work
        self.source_id = hashlib.sha256(json.dumps(dict(snapshot=snapshot.source_id,
            coverage=coverage, process='coherent CI removal spectator-ion Coulomb DW',
            continuum='selected final-ion direct Hartree; all supplied bound Pauli projectors'),
            sort_keys=True).encode()).hexdigest()

    def bound_channels(self, initial, final):
        """Return all nonzero coherent kappa channels with the actual CI threshold."""
        s = self.snapshot
        s.threshold_eV(initial, final)
        a, b = s.levels[initial], s.levels[final]
        threshold = b.energy_Ha-a.energy_Ha
        keys, coefficients = reduced_dyson_annihilation(
            s._vectors[final][b.two_J], s._vectors[initial][a.two_J],
            s.spinors, a.two_J, b.two_J, engine=s.dyson_engine)
        if keys != s.keys:
            raise ValueError('complete CI removal radial-layout mismatch')
        channels = []
        for kappa in sorted({k for _, k in keys}):
            selected = [(key, weight) for key, weight in zip(keys, coefficients, strict=True)
                        if key[1] == kappa and weight != 0]
            if not selected:
                continue  # Exact zero only, including forbidden parity/triangles.
            if b.parity*(-1)**(kappa if kappa > 0 else -kappa-1) != a.parity:
                raise ValueError('nonzero CI removal violates fixed state parity')
            p = sum(weight*s.orbitals[key][0] for key, weight in selected)
            q = sum(weight*s.orbitals[key][1] for key, weight in selected)
            channels.append(EIIDWBoundChannel(f'{a.state_id}->{b.state_id}:k{kappa}',
                            kappa, threshold, 1., p, q))
        return tuple(channels)

    def cross_section(self, initial, final, energies_eV, *, kappa_max=2,
                      sharing_order=6, radial_refinement=1):
        """Partial total EII cross section [cm²], without a fitted threshold/tail.

        ``kappa_max`` follows the existing DW orbital-l cutoff convention.
        Full magnetic and same-kappa radial interference are retained. Resource
        admission rejects a full requested calculation rather than trimming it.
        """
        s = self.snapshot
        if (isinstance(kappa_max, bool) or not isinstance(kappa_max, int) or kappa_max < 1
                or isinstance(sharing_order, bool) or not isinstance(sharing_order, int) or sharing_order < 3):
            raise ValueError('positive partial-wave cutoff and sharing order>=3 required')
        channels = self.bound_channels(initial, final)
        work = (2*kappa_max+1)**3*sum(2*abs(c.kappa)+2*kappa_max+2 for c in channels)
        if work > self.max_channel_work:
            raise MemoryError('full CI DW channel-work cap exceeded; no term removed')
        # Revalidate the bound nuclear snapshot even when no PI wave is needed.
        s.prepare_continuum_waves(final, [])
        projectors = {k: tuple(s.orbitals[key] for key in s.keys if key[1] == k)
                      for k in sorted({key[1] for key in s.keys})}
        result = _eii_dw_channels(channels, s.data, s.Z, energies_eV,
            V_func=s._field(final), Z_res=s.Z-s.levels[final].electrons,
            potential_kind='selected final CI-ion direct Hartree', pauli_projectors=projectors,
            kappa_max=kappa_max, n_outgoing_energies=sharing_order,
            continuum_model=s.model, continuum_failure_mode='raise',
            partial_wave_tail='none', radial_refinement=radial_refinement)
        result.provenance.update(source_id=self.source_id, snapshot_id=s.source_id,
            initial_state=s.levels[initial].state_id, final_state=s.levels[final].state_id,
            matched_threshold_eV=s.threshold_eV(initial, final), coverage=self.coverage,
            channel_work_bound=work, full_ionic_EII_certified=False,
            model='first-order spectator-ion Coulomb DW with coherent signed CI removal',
            experimental_accepted=False)
        return result

    def rate_coefficient(self, initial, final, temperature_eV, *, order=16,
                         kappa_max=16, sharing_order=12, radial_refinement=1):
        """Direct-node partial Maxwellian EII rate, with complete sampled data.

        Numerical convergence of this integral and of its cross sections is
        separate from channel closure or experimental source validation. All
        nodes remain in the returned audit; no high-energy tail is fitted.
        """
        from atomx.plasma.rates import maxwellian_rate_direct

        captured = []
        def cross_section(energy):
            result = self.cross_section(initial, final, energy, kappa_max=kappa_max,
                sharing_order=sharing_order, radial_refinement=radial_refinement)
            captured.append(result)
            return result.sigma_total_cm2
        threshold = self.snapshot.threshold_eV(initial, final)
        rate = maxwellian_rate_direct(cross_section, threshold, temperature_eV, order=order)
        result, = captured
        return dict(rate_cm3_s=rate, temperature_eV=float(temperature_eV),
            threshold_eV=threshold, quadrature_order=int(order), source_id=self.source_id,
            energies_eV=result.energies_eV.tolist(), cross_section_cm2=result.sigma_total_cm2.tolist(),
            cross_section_provenance=result.provenance, full_ionic_EII_certified=False,
            experimental_accepted=False)
