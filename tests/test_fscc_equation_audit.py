"""Independent finite signed gates for the noncanonical attachment audit."""

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
PROBE = REPO / "docs/atomic_pci_sd_native_2026_10_09/parallel_fscc012/probe.py"
spec = importlib.util.spec_from_file_location("fscc_equation_audit", PROBE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def elementary_matrix(masks, terms):
    """Independent occupation-list action, distinct from oracle.apply."""
    index = {mask: i for i, mask in enumerate(masks)}
    matrix = np.zeros((len(masks), len(masks)), complex)
    for col, mask in enumerate(masks):
        for operations, value in terms:
            occupied = [i for i in range(6) if mask >> i & 1]
            phase = 1
            for spinor, create in reversed(operations):
                if (spinor in occupied) == create:
                    break
                below = sum(i < spinor for i in occupied)
                phase *= (-1) ** below
                if create:
                    occupied.append(spinor)
                else:
                    occupied.remove(spinor)
            else:
                target = sum(1 << i for i in occupied)
                matrix[index[target], col] += value * phase
    return matrix


def test_open_fov_t2_has_one_antisymmetric_copy_with_complex_phases():
    """Extract the exact two-body open coefficient of [Fdeex,T2]."""
    rng = np.random.default_rng(314519)
    fock = np.zeros((6, 6), complex)
    fock[:2, 2:] = rng.normal(size=(2, 4)) + 1j * rng.normal(size=(2, 4))
    fock[2:, :2] = fock[:2, 2:].conj().T
    bra_t2 = np.zeros((6,) * 4, complex)
    independent_terms = []
    for a, b in itertools.combinations(range(2, 6), 2):
        ket = rng.normal() + 1j * rng.normal()
        bra_t2[0, 1, a, b] = bra_t2[1, 0, b, a] = ket.conjugate()
        bra_t2[1, 0, a, b] = bra_t2[0, 1, b, a] = -ket.conjugate()
        independent_terms.append((((a, True), (b, True), (1, False), (0, False)), ket))
    masks = [sum(1 << i for i in pair) for pair in itertools.combinations(range(6), 2)]
    t2_exact = elementary_matrix(masks, independent_terms)
    t2_lift = elementary_matrix(masks, audit.cluster_terms([np.zeros((6, 6)), bra_t2]))
    assert np.max(abs(t2_exact - t2_lift)) < 1e-14
    f = elementary_matrix(
        masks,
        [(((i, True), (v, False)), fock[i, v]) for i in range(2) for v in range(2, 6)],
    )
    commutator = f @ t2_exact - t2_exact @ f
    bra_open = np.einsum("vj,jiab->viab", fock, bra_t2)
    bra_open[:2] = 0
    open_lift = elementary_matrix(
        masks, audit.cluster_terms([np.zeros((6, 6)), bra_open], attachment=True)
    )
    rows = [i for i, mask in enumerate(masks) if mask & 3 == 0]
    cols = [i for i, mask in enumerate(masks) if (mask & 3).bit_count() == 1]
    assert np.max(abs((commutator + open_lift)[np.ix_(rows, cols)])) < 1e-14
    doubled = np.max(abs((commutator + 2 * open_lift)[np.ix_(rows, cols)]))
    assert doubled > 0.1  # This independent complex case distinguishes 1 from 2.


def test_complete_saved_actual_amplitudes_locate_open_fov_t2_residual():
    metrics, arrays = audit.audit(audit.DEFAULT_CASE)
    assert metrics["decoder_and_core_gate_pass"]
    assert metrics["exact_oracle_gate_pass"]
    assert arrays["backend_omega"].shape == (120, 8)
    assert 4e-7 < metrics["backend_wave_bloch_residual_max_Ha"] < 4.4e-7
    assert metrics["residual_minus_open_Fov_T2_max_Ha"] < 1e-11
    assert not metrics["strict_backend_gate_pass"]


def test_decoder_rejects_unknown_magic_truncation_and_trailing_bytes(tmp_path):
    original = (audit.DEFAULT_CASE / "scratch/t1c.dg").read_bytes()
    magic_at = original.index(b"ATXDG002")
    for name, content in (
        ("magic", original[:magic_at] + b"OTHER002" + original[magic_at + 8 :]),
        ("truncated", original[:-1]),
        ("trailing", original + b"\0"),
    ):
        path = tmp_path / name
        path.write_bytes(content)
        with pytest.raises(ValueError):
            audit.decode_diagram(path)


def test_decoder_complete_signed_permutation_and_integral_convention():
    t2, metadata = audit.decode_diagram(audit.DEFAULT_CASE / "scratch/t2c.dg")
    assert metadata["rank"] == 4 and metadata["qparts"] == list(map(ord, "hhpp"))
    assert np.max(abs(t2 + t2.swapaxes(0, 1))) < 1e-14
    assert np.max(abs(t2 + t2.swapaxes(2, 3))) < 1e-14
    assert np.count_nonzero(t2) > 0
    with pytest.raises(ValueError, match="declared bounds"):
        audit.decode_diagram(audit.DEFAULT_CASE / "scratch/t2c.dg", nspinors=100)
