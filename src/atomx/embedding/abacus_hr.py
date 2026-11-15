"""ABACUS LCAO real-space Hamiltonian -> k-space, Loewdin basis, correlated d projector.

Input: ``data-HR-sparse_SPIN0.csr`` / ``data-SR-sparse_SPIN0.csr`` written with
``out_mat_hs2 1`` (H in Ry, S dimensionless), plus STRU and the .orb files to
recover the orbital layout.

CSR block format (ABACUS 3.7): header lines ``STEP``, ``Matrix Dimension``,
``Matrix number``; then per R: ``Rx Ry Rz nnz``; if nnz > 0 three lines follow:
values, column indices, row pointers (length N+1).

ABACUS LCAO orbital order: for each atom (in STRU order), for L = 0..Lmax,
for each zeta N, for m in ABACUS order; for L = 2 the m order is
dz2, dxz, dyz, dx2-y2, dxy (same convention as ``abacus_cf``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RY_EV = 13.605693122994


def read_csr(path) -> dict:
    lines = Path(path).read_text().split("\n")
    i = 0
    dim = None
    out = {}
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("Matrix Dimension"):
            dim = int(ln.split(":")[1])
            i += 1
            continue
        if not ln or ln.startswith("STEP") or ln.startswith("Matrix number"):
            i += 1
            continue
        parts = ln.split()
        if len(parts) == 4 and dim is not None:
            R = tuple(int(x) for x in parts[:3])
            nnz = int(parts[3])
            M = np.zeros((dim, dim))
            if nnz > 0:
                vals = np.array(lines[i + 1].split(), float)
                cols = np.array(lines[i + 2].split(), int)
                ptr = np.array(lines[i + 3].split(), int)
                for r in range(dim):
                    sl = slice(ptr[r], ptr[r + 1])
                    M[r, cols[sl]] = vals[sl]
                i += 4
            else:
                i += 1
            out[R] = M
            continue
        i += 1
    return out


def orbital_counts(orb_file) -> list[int]:
    """Number of zetas per L from an ABACUS .orb header."""
    txt = Path(orb_file).read_text()
    counts = []
    for name in "SPDFGH":
        m = re.search(rf"Number of {name}orbital-->\s*(\d+)", txt)
        if not m:
            break
        counts.append(int(m.group(1)))
    return counts


@dataclass
class Layout:
    labels: list  # one entry per basis function: (atom_index, species, L, zeta, m)
    atoms: list  # [(species, frac_pos)]

    def index(self, atom, L, zeta=None):
        return [
            i
            for i, (a, _, l, z, _) in enumerate(self.labels)
            if a == atom and l == L and (zeta is None or z == zeta)
        ]


def parse_layout(stru, orbital_dir) -> Layout:
    txt = Path(stru).read_text().split("\n")
    species, orbs = [], []
    sec = None
    atoms = []
    i = 0
    while i < len(txt):
        ln = txt[i].strip()
        if ln in (
            "ATOMIC_SPECIES",
            "NUMERICAL_ORBITAL",
            "LATTICE_CONSTANT",
            "LATTICE_VECTORS",
            "ATOMIC_POSITIONS",
        ):
            sec = ln
            i += 1
            continue
        if sec == "ATOMIC_SPECIES" and ln:
            species.append(ln.split()[0])
        elif sec == "NUMERICAL_ORBITAL" and ln:
            orbs.append(ln.split()[0])
        elif sec == "ATOMIC_POSITIONS" and ln and ln not in ("Direct", "Cartesian"):
            sp = ln
            if sp in species:
                nat = int(txt[i + 2].split()[0])
                for k in range(nat):
                    pos = np.array(txt[i + 3 + k].split()[:3], float)
                    atoms.append((sp, pos))
                i += 3 + nat
                continue
        i += 1
    counts = {sp: orbital_counts(Path(orbital_dir) / o) for sp, o in zip(species, orbs, strict=True)}
    labels = []
    for a, (sp, _) in enumerate(atoms):
        for L, nz in enumerate(counts[sp]):
            for z in range(nz):
                for m in range(2 * L + 1):
                    labels.append((a, sp, L, z, m))
    return Layout(labels=labels, atoms=atoms)


def kmesh(n: int) -> np.ndarray:
    g = np.arange(n) / n
    return np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)


def fourier(MR: dict, ks: np.ndarray) -> np.ndarray:
    Rs = np.array(list(MR.keys()), float)
    Ms = np.stack(list(MR.values()))
    ph = np.exp(2j * np.pi * ks @ Rs.T)  # (nk, nR)
    return np.einsum("kr,rij->kij", ph, Ms, optimize=True)


def lowdin_hk(HR: dict, SR: dict, ks: np.ndarray):
    """Orthonormal (Loewdin) H(k) in eV, plus S^{-1/2} for projections."""
    Hk = fourier(HR, ks) * RY_EV
    Sk = fourier(SR, ks)
    Hk = 0.5 * (Hk + Hk.conj().transpose(0, 2, 1))
    Sk = 0.5 * (Sk + Sk.conj().transpose(0, 2, 1))
    w, U = np.linalg.eigh(Sk)
    if w.min() <= 0:
        raise ValueError(f"overlap not positive definite (min eig {w.min():.2e})")
    Sm12 = np.einsum("kij,kj,klj->kil", U, w**-0.5, U.conj(), optimize=True)
    Hl = Sm12 @ Hk @ Sm12
    return 0.5 * (Hl + Hl.conj().transpose(0, 2, 1)), Sm12


def natural_d_projector(
    Hl: np.ndarray, layout: Layout, atom: int, n_electrons: float, nspin_deg: int = 2
):
    """Correlated d subspace: for each cubic m, the zeta combination with the largest
    PBE occupation (natural orbital of the 2x2 zeta block). Returns W (nbasis x 5),
    columns in ABACUS cubic order (dz2, dxz, dyz, dx2-y2, dxy), plus mu_PBE and n_d."""
    E, V = np.linalg.eigh(Hl)
    mu = fermi_level(E, n_electrons, nspin_deg)
    occ = (mu > E).astype(float)
    rho = np.einsum("kin,kn,kjn->ij", V, occ, V.conj(), optimize=True) / len(Hl)
    nb = Hl.shape[1]
    W = np.zeros((nb, 5), complex)
    zetas = sorted({z for (a, _, L, z, _) in layout.labels if a == atom and L == 2})
    for m in range(5):
        idx = [
            i
            for i, (a, _, L, z, mm) in enumerate(layout.labels)
            if a == atom and L == 2 and mm == m
        ]
        blk = rho[np.ix_(idx, idx)]
        ev, ec = np.linalg.eigh(blk)
        W[idx, m] = ec[:, -1]
    n_d = nspin_deg * np.real(np.trace(W.conj().T @ rho @ W))
    return W, mu, float(n_d), len(zetas)


def fermi_level(E: np.ndarray, n_electrons: float, nspin_deg: int = 2) -> float:
    e = np.sort(E.ravel())
    nk = E.shape[0]
    n_states = int(round(n_electrons / nspin_deg * nk))
    return 0.5 * (e[n_states - 1] + e[n_states])
