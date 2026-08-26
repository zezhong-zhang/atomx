#!/usr/bin/env python3
"""Condensed-matter local-environment convergence study (HYP-909).

Rock-salt MgO: the central Mg 1s core level is tracked through the
local-orbital bridge while the quantum cluster (``quantum_shells``) and the
point-charge embedding radius (``mm_shells``, Evjen fractional boundary
charges) grow toward the infinite-lattice Madelung limit, plus an
embedding-charge sensitivity sweep (``charge_scale``).  The reference is
the isolated Mg atom computed with the same basis and the same ΔSCF
protocol, so every shift carries a declared, traceable energy zero
(``mo_energy`` for orbital shifts, ``delta_scf_total_energy`` for binding
shifts).

Usage::

    python benchmarks/bench_pyscf_embedding_convergence.py [--csv OUT]

Requires the optional ``atomx[pyscf]`` extra.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

HARTREE_TO_EV = 27.211386245988

# Rock-salt MgO (B1): nearest-neighbour distance 2.106 Angstrom, formal
# charges +/-2.  The infinite-lattice Madelung shift of a cation core level
# is ~ alpha_M * q / r0 = 1.7476 * 2 / 3.98 Bohr ~ 23.9 eV.
MGO_NN_ANGSTROM = 2.106


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=None, help="write rows here")
    parser.add_argument("--basis", default="sto-3g")
    args = parser.parse_args()

    try:
        import pyscf  # noqa: F401
    except ImportError:
        print("PySCF is required: pip install 'atomx[pyscf]'", file=sys.stderr)
        return 2

    from atomx.providers.pyscf_embedding import study_cluster_convergence

    # Cluster-size / embedding-radius convergence at formal charges.
    study = study_cluster_convergence(
        cation="Mg",
        anion="O",
        q_cation=2.0,
        q_anion=-2.0,
        nearest_neighbor_angstrom=MGO_NN_ANGSTROM,
        basis=args.basis,
        core_shell="1s",
        quantum_shells=(0, 1),
        mm_shells=(1, 2, 3),
        evjen=True,
        charge_scales=(1.0,),
    )
    # Embedding-charge sensitivity at the largest embedding radius.
    sensitivity = study_cluster_convergence(
        cation="Mg",
        anion="O",
        q_cation=2.0,
        q_anion=-2.0,
        nearest_neighbor_angstrom=MGO_NN_ANGSTROM,
        basis=args.basis,
        core_shell="1s",
        quantum_shells=(0,),
        mm_shells=(3,),
        evjen=True,
        charge_scales=(0.5, 1.0, 1.5),
    )

    rows = study.to_table() + sensitivity.to_table()
    print(f"reference (isolated Mg, {args.basis}):")
    print(f"  Mg 1s epsilon   = {study.reference_identity.energy_hartree:.8f} Ha")
    print(f"  Mg 1s binding   = {study.reference_binding_hartree:.8f} Ha "
          f"({study.reference_binding_hartree * HARTREE_TO_EV:.3f} eV)")
    header = (
        "label", "n_qm", "n_mm", "scale", "dominance",
        "d_eps(eV)", "d_bind(eV)",
    )
    print(" | ".join(f"{h:>24s}" for h in header))
    for row in rows:
        print(
            " | ".join(
                [
                    f"{row['label']:>24s}",
                    f"{row['n_quantum_atoms']:>24d}",
                    f"{row['n_mm_charges']:>24d}",
                    f"{row['charge_scale']:>24.2f}",
                    f"{row['core_dominance']:>24.6f}",
                    f"{row['orbital_shift_hartree'] * HARTREE_TO_EV:>24.4f}",
                    f"{row['binding_shift_hartree'] * HARTREE_TO_EV:>24.4f}",
                ]
            )
        )
    if args.csv is not None:
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
