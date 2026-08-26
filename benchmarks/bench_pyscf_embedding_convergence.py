#!/usr/bin/env python3
"""Condensed-matter local-environment convergence study (HYP-909).

Rock-salt MgO: the central Mg 1s core level is tracked through the
local-orbital bridge while the point-charge embedding radius grows, plus an
embedding-charge sensitivity sweep (``charge_scale``).  The reference is
the isolated Mg atom computed with the same basis and the same ΔSCF
protocol, so every shift carries a declared, traceable energy zero
(``mo_energy`` for orbital shifts, ``delta_scf_total_energy`` for binding
shifts).

Two distinct physical models — do NOT read them as one curve
-------------------------------------------------------------
* **Point-ion embedding (default, ``quantum_shells=(0,)``)**: only the
  central Mg is quantum; the Evjen-neutral embedding carries net charge
  ``-q_Mg`` (anion-dominated), so the cation core level is *destabilized*
  toward the bulk Madelung limit (~+24 eV orbital shift) as ``mm_shells``
  grows.  This is the series that converges to the infinite-lattice
  point-charge limit.
* **Charged quantum cluster (``--with-quantum-cluster``, qm=1)**: the
  quantum region is the non-stoichiometric 27-site Mg13O14 cube whose
  formal charge is *negative*; the embedding then carries ``+q_Mg`` and the
  central cation sees an **opposite-sign** potential, so the shifts
  reverse.  This series answers a different question — convergence of the
  quantum cluster itself (screening / charge transfer from real
  neighbours) — and must not be plotted as a continuation of the qm0
  curve.

Usage::

    python benchmarks/bench_pyscf_embedding_convergence.py [--csv OUT]
    python benchmarks/bench_pyscf_embedding_convergence.py --with-quantum-cluster

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

#: The 27-atom (qm=1) clusters need ~100 SCF cycles; PySCF's default
#: max_cycle=50 fails closed before the study table is produced.
MAX_CYCLE = 200

#: Human-readable series labels; the two series are distinct physical
#: models and must never be read as a single convergence curve.
SERIES_LABELS = {
    0: "point-ion embedding (qm0)",
    1: "charged quantum cluster (qm1)",
}


def _print_table(title: str, rows: list[dict]) -> None:
    print(f"== {title} ==")  # noqa: T201
    header = (
        "label", "n_qm", "n_mm", "scale", "dominance",
        "d_eps(eV)", "d_bind(eV)",
    )
    print(" | ".join(f"{h:>24s}" for h in header))  # noqa: T201
    for row in rows:
        print(  # noqa: T201
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=None, help="write rows here")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument(
        "--with-quantum-cluster",
        action="store_true",
        help="also run the qm=1 charged-cluster (Mg13O14) sensitivity series "
        "(adds several minutes; a distinct physical model from the default "
        "point-ion embedding series — see the module docstring)",
    )
    args = parser.parse_args()

    try:
        import pyscf  # noqa: F401
    except ImportError:
        print("PySCF is required: pip install 'atomx[pyscf]'", file=sys.stderr)  # noqa: T201
        return 2

    from atomx.providers.pyscf_embedding import study_cluster_convergence

    common = dict(
        cation="Mg",
        anion="O",
        q_cation=2.0,
        q_anion=-2.0,
        nearest_neighbor_angstrom=MGO_NN_ANGSTROM,
        basis=args.basis,
        core_shell="1s",
        evjen=True,
        max_cycle=MAX_CYCLE,
    )

    # Default series: point-ion embedding, cluster-size (mm_shells)
    # convergence toward the bulk Madelung limit.
    study = study_cluster_convergence(
        **common,
        quantum_shells=(0,),
        mm_shells=(1, 2, 3),
        charge_scales=(1.0,),
    )
    # Embedding-charge sensitivity at the largest embedding radius.
    sensitivity = study_cluster_convergence(
        **common,
        quantum_shells=(0,),
        mm_shells=(3,),
        charge_scales=(0.5, 1.0, 1.5),
    )

    series: dict[str, list[dict]] = {
        "point-ion embedding (qm0): Madelung convergence": study.to_table(),
        "point-ion embedding (qm0): charge sensitivity": sensitivity.to_table(),
    }
    reference_identity = study.reference_identity
    reference_binding = study.reference_binding_hartree

    if args.with_quantum_cluster:
        cluster = study_cluster_convergence(
            **common,
            quantum_shells=(1,),
            mm_shells=(2, 3),
            charge_scales=(1.0,),
        )
        series[
            "charged quantum cluster (qm1, Mg13O14): OPPOSITE-SIGN potential"
            " — not a continuation of the qm0 curve"
        ] = cluster.to_table()

    print(f"reference (isolated Mg, {args.basis}):")  # noqa: T201
    print(f"  Mg 1s epsilon   = {reference_identity.energy_hartree:.8f} Ha")  # noqa: T201
    print(f"  Mg 1s binding   = {reference_binding:.8f} Ha "  # noqa: T201
          f"({reference_binding * HARTREE_TO_EV:.3f} eV)")
    rows = []
    for title, table in series.items():
        _print_table(title, table)
        rows.extend(table)
    if args.csv is not None:
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
