"""Native electric lines between exact finite CI Bloch invariant spaces.

This is a bounded finite-space consumer, not a metric inferred for SD/CC.
It shares the CI tensor phases, native radial photon operator and rate
normalization with the existing transition machinery. No bare P O P,
occupation average, fitted energy shift or orbital-frequency substitution.
"""
from dataclasses import dataclass

import numpy as np

from atomx.angular.csf_hamiltonian import (
    csf_signature,
    one_body_tensor_terms,
    pattern_m_counts,
)
from atomx.config.csf import get_shared_orbital_basis
from atomx.methods.ci_effective import (
    FiniteBlochReduction,
    FiniteCIModel,
    FiniteCITransitionOperator,
    _transition_storage_guard,
)
from atomx.numerics.radial_workspace import workspace_arrays
from atomx.physics.csf.effective_one_body import CIReferenceBinding
from atomx.physics.csf.matrix import _check_orbital_metric
from atomx.properties.transitions.angular import angular_eL, selection_rules
from atomx.properties.transitions.multipole import radial_eL, radial_eL_velocity
from atomx.properties.transitions.rates import einstein_A_eL, oscillator_strength

__all__ = ["FiniteCIElectricLine", "build_finite_ci_electric_operator", "finite_ci_electric_line"]


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"positive integer {name} required")


def _native_sector(model, csfs, orbitals, workspace, Z, ns, nucleus):
    if not isinstance(model, FiniteCIModel) or not isinstance(model.radial_binding, CIReferenceBinding):
        raise TypeError("native finite CI parent with physical radial binding required")
    signatures = tuple(csf_signature(c) for c in csfs)
    if tuple(s.key for s in signatures) != model.csf_identities:
        raise ValueError("native transition CSF order does not match its parent snapshot")
    sectors = {(c.n_electrons, c.total_J2, c.parity) for c in csfs}
    if len(sectors) != 1:
        raise ValueError("one native N/J/parity sector required at each endpoint")
    keys = get_shared_orbital_basis(csfs)
    model.radial_binding.validate(keys, orbitals, workspace, Z, ns, nucleus)
    return signatures, next(iter(sectors))


def build_finite_ci_electric_operator(bra, ket, bra_csfs, ket_csfs, orbitals_dict,
                                      bsp_data, Z, ns, *, omega_Ha, rank=1,
                                      gauge="length", photon_approximation="finite_wavelength",
                                      nuclear_potential=None, dimension_cap=256,
                                      max_determinants=2000, angular_work_cap=10_000_000,
                                      memory_bytes=64*1024**2):
    """Full signed rectangular <bra||E_rank(omega)||ket> in native CSF order.

    ``omega_Ha`` is the signed MANY-ELECTRON E_bra-E_ket, fixed for every
    orbital pair. Use one operator per line. Reverse endpoints and omega
    together. Long wavelength is explicit; velocity retains its own kernel.
    The union of endpoint orbitals must be orthonormal in one native metric.
    Independently optimized nonorthogonal endpoints are rejected here.

    Complete CSF dimensions, determinant upper counts, angular work and
    storage are checked before angular-vector or radial construction. Caps
    reject; they never discard a coupling tree or small amplitude.
    """
    for name, value in (("rank", rank), ("dimension cap", dimension_cap),
                        ("determinant cap", max_determinants),
                        ("angular work cap", angular_work_cap), ("memory cap", memory_bytes)):
        _positive_integer(value, name)
    if (isinstance(omega_Ha, (bool, np.bool_)) or not np.isscalar(omega_Ha)
            or not np.isrealobj(omega_Ha) or not np.isfinite(omega_Ha) or omega_Ha == 0):
        raise ValueError("finite nonzero signed many-electron photon energy required")
    if gauge not in ("length", "velocity"):
        raise ValueError("explicit length or velocity electric gauge required")
    if photon_approximation not in ("finite_wavelength", "long_wavelength"):
        raise ValueError("explicit finite_wavelength or long_wavelength photon operator required")
    bra_csfs, ket_csfs = tuple(bra_csfs), tuple(ket_csfs)
    if not bra_csfs or not ket_csfs or max(len(bra_csfs), len(ket_csfs)) > dimension_cap:
        raise ValueError("complete native transition space exceeds dimension cap")
    if not isinstance(bra, FiniteCIModel) or not isinstance(ket, FiniteCIModel):
        raise TypeError("two native finite CI parents required")
    if (bra.hamiltonian_terms != ket.hamiltonian_terms
            or bra.hamiltonian_terms not in (("dirac-coulomb",), ("dirac-coulomb", "static-breit"))):
        raise ValueError("common native Hamiltonian operator coverage required for both endpoints")
    bs, bsector = _native_sector(bra, bra_csfs, orbitals_dict, bsp_data, Z, ns, nuclear_potential)
    ks, ksector = _native_sector(ket, ket_csfs, orbitals_dict, bsp_data, Z, ns, nuclear_potential)
    if bsector[0] != ksector[0]:
        raise ValueError("one-body photon operator requires equal electron counts")
    counts = [dict(pattern_m_counts(s.pattern)).get(s.total_J2, 0) for s in bs+ks]
    if not all(0 < n <= max_determinants for n in counts):
        raise ValueError("complete native magnetic determinant space exceeds cap")
    union_keys = get_shared_orbital_basis(bra_csfs+ket_csfs)
    spinors = sum(2*abs(k) for _, k in union_keys)
    work = len(bs)*sum(counts[len(bs):])*bsector[0]*spinors + sum(counts)
    if work > angular_work_cap:
        raise ValueError("complete native angular transition work exceeds cap")
    quadrature_points = len(workspace_arrays(bsp_data)[0])
    required = (_transition_storage_guard(bra, ket, 8, memory_bytes)
                + 256*sum(counts) + 128*len(bs)*len(ks)*len(union_keys)**2
                + 512*quadrature_points + 64*len(union_keys)**2)
    if required > memory_bytes:
        raise MemoryError(f"complete native electric transition work {required} bytes exceeds cap {memory_bytes}")
    _check_orbital_metric(bra_csfs+ket_csfs, orbitals_dict, bsp_data, Z, ns)
    matrix = np.zeros((len(bs), len(ks)))
    jf, ji = bsector[1], ksector[1]
    allowed = (abs(jf-ji) <= 2*rank <= jf+ji and bsector[2]*ksector[2] == (-1)**rank)
    radial_cache = {}
    if allowed:
        for i, a in enumerate(bs):
            for j, b in enumerate(ks):
                for destination, source, coefficient in one_body_tensor_terms(a, b, rank):
                    ki, kf = source[1], destination[1]
                    if not selection_rules(ki, kf, rank, "E"):
                        continue
                    key = (destination, source)
                    if key not in radial_cache:
                        pi, qi = orbitals_dict[source]
                        pf, qf = orbitals_dict[destination]
                        if gauge == "velocity":
                            radial = radial_eL_velocity(
                                pi, qi, pf, qf, bsp_data, rank, kappa_i=ki, kappa_f=kf,
                                omega=float(omega_Ha),
                                long_wavelength=photon_approximation == "long_wavelength")
                        else:
                            radial = radial_eL(
                                pi, qi, pf, qf, bsp_data, rank, kappa_i=ki, kappa_f=kf,
                                omega=float(omega_Ha) if photon_approximation == "finite_wavelength" else None)
                        radial_cache[key] = angular_eL(ki, kf, rank)*radial
                    matrix[i, j] += coefficient*radial_cache[key]
    identity = (f"native:E{rank}:{gauge}:{photon_approximation}:"
                f"signed_Ebra_minus_Eket_Ha={float(omega_Ha).hex()}")
    return FiniteCITransitionOperator(matrix, bra, ket, operator_identity=identity,
                                      unit=f"a0^{rank}", memory_bytes=memory_bytes)


@dataclass(frozen=True)
class FiniteCIElectricLine:
    """One line with signed <upper||E||lower> and normalized physical strength."""
    upper_root_index: int
    lower_root_index: int
    upper_energy_Ha: float
    lower_energy_Ha: float
    upper_two_J: int
    lower_two_J: int
    omega_Ha: float
    reduced_matrix_element_au: float
    line_strength: float
    A_rate_per_s: float
    f_value: float
    gf_value: float
    rank: int
    gauge: str
    photon_approximation: str
    representation: str
    full_operator: FiniteCITransitionOperator


def finite_ci_electric_line(upper, lower, upper_csfs, lower_csfs, orbitals_dict,
                            bsp_data, Z, ns, *, upper_state=0, lower_state=0,
                            representation, rank=1, gauge="length",
                            photon_approximation="finite_wavelength", nuclear_potential=None,
                            dimension_cap=256, max_determinants=2000,
                            angular_work_cap=10_000_000, memory_bytes=64*1024**2):
    """Consume exact normalized effective states with a line-specific photon.

    State indices are positions in each reduction's explicit selected-root
    order, not CSF indices. A uses the upper statistical weight; absorption
    f uses the lower weight. No total lifetime is inferred from one line.
    """
    if not isinstance(upper, FiniteBlochReduction) or not isinstance(lower, FiniteBlochReduction):
        raise TypeError("exact finite upper/lower Bloch reductions required")
    for index, reduction in ((upper_state, upper), (lower_state, lower)):
        if (isinstance(index, bool) or not isinstance(index, (int, np.integer))
                or not 0 <= index < len(reduction.root_indices)):
            raise ValueError("selected finite root state index required")
    if representation not in ("bloch", "orthonormal"):
        raise ValueError("explicit bloch or orthonormal operator representation required")
    eu, el = float(upper.energies_Ha[upper_state]), float(lower.energies_Ha[lower_state])
    omega = eu-el
    if omega <= 0 or not np.isfinite(omega):
        raise ValueError("upper root must have a positive emitted photon energy above lower root")
    upper_csfs, lower_csfs = tuple(upper_csfs), tuple(lower_csfs)
    operator = build_finite_ci_electric_operator(
        upper.parent, lower.parent, upper_csfs, lower_csfs, orbitals_dict, bsp_data, Z, ns,
        omega_Ha=omega, rank=rank, gauge=gauge, photon_approximation=photon_approximation,
        nuclear_potential=nuclear_potential, dimension_cap=dimension_cap,
        max_determinants=max_determinants, angular_work_cap=angular_work_cap, memory_bytes=memory_bytes)
    amplitudes = upper.transition_amplitudes(operator, lower, representation=representation,
                                             memory_bytes=memory_bytes)
    amplitude = float(amplitudes[upper_state, lower_state])
    strength = amplitude**2
    ju, jl = upper_csfs[0].total_J2, lower_csfs[0].total_J2
    rate = float(einstein_A_eL(omega, strength, rank, ju))
    f, gf = oscillator_strength(omega, strength, rank, jl, ju)
    if not np.all(np.isfinite([amplitude, strength, rate, f, gf])):
        raise RuntimeError("nonfinite physical electric line")
    return FiniteCIElectricLine(upper.root_indices[upper_state], lower.root_indices[lower_state],
                                eu, el, ju, jl, omega, amplitude, strength, rate,
                                float(f), float(gf), rank, gauge, photon_approximation,
                                representation, operator)
