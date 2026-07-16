#!/usr/bin/env python3
"""Inventory FAC AngularZxZ0 branch coverage in AtomX CSF panels.

This diagnostic is intentionally independent of the radial solver.  It answers
which angular-recouping branch a CSF interaction would exercise, and whether
AtomX currently evaluates that interaction with a FAC-style angular coefficient
or with a conservative fallback.  Use it before tuning against individual FAC
level residuals; unsupported branch counts are stronger evidence than one small
case.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atomx.angular.coefficients import exchange_k_range, exchange_lambda  # noqa: E402
from atomx.angular.csf_recoupling import (  # noqa: E402
    aligned_subshell_keys_from_simple_csfs,
    fac_angular_zxz0_branch_case,
    fac_simple_axw_wxa_zxz0_postprocess,
    single_state_for_subshell,
)
from atomx.methods.mcdhf.config import enumerate_jj_csfs  # noqa: E402
from atomx.methods.mcdhf.matrix import (  # noqa: E402
    _chain_one_diff_spectator_exact_scalar,
    _fac_inner_shell_index,
    _fac_one_diff_open_spectator_exchange_coef,
    _fac_two_diff_branch9_element,
    _has_explicit_J,
)

FE17_CONFIGS = (
    "1s2 2s2 2p6",
    "1s2 2s2 2p5 3s1",
    "1s2 2s2 2p5 3p1",
    "1s2 2s2 2p5 3d1",
    "1s2 2s1 2p6 3s1",
    "1s2 2s1 2p6 3p1",
    "1s2 2s1 2p6 3d1",
)

DEFAULT_PANELS = {
    "Be-like odd": ("1s2 2s1 2p1",),
    "Ne-like hole": (
        "1s2 2s2 2p6",
        "1s2 2s2 2p5 3s1",
        "1s2 2s2 2p5 3p1",
        "1s2 2s2 2p5 3d1",
    ),
    "Fe17 L-shell": FE17_CONFIGS,
}


@dataclass(frozen=True)
class BranchRow:
    panel: str
    interaction: str
    support: str
    branch: int
    order: str
    phase: int
    j2: int
    parity: int
    bra: str
    ket: str
    shells: str
    ranks: str
    fallback_unit_coeff: float = 0.0
    exact_chain_scalar: str = ""
    composite_seed: str = ""
    note: str = ""

    def as_dict(self) -> dict[str, str | int]:
        return {
            "panel": self.panel,
            "interaction": self.interaction,
            "support": self.support,
            "branch": self.branch,
            "order": self.order,
            "phase": self.phase,
            "J2": self.j2,
            "parity": self.parity,
            "bra": self.bra,
            "ket": self.ket,
            "shells": self.shells,
            "ranks": self.ranks,
            "fallback_unit_coeff": f"{self.fallback_unit_coeff:.12g}",
            "exact_chain_scalar": self.exact_chain_scalar,
            "composite_seed": self.composite_seed,
            "note": self.note,
        }


def _key_text(key: tuple[int, int]) -> str:
    return f"{key[0]}:{key[1]}"


def _keys_text(keys: Iterable[tuple[int, int]]) -> str:
    return " ".join(_key_text(key) for key in keys)


def _same_block(csf_a, csf_b) -> bool:
    return (
        int(getattr(csf_a, "total_J2", 0)) == int(getattr(csf_b, "total_J2", 0))
        and int(getattr(csf_a, "parity", 1)) == int(getattr(csf_b, "parity", 1))
    )


def _occ(csf) -> dict[tuple[int, int], int]:
    return {(int(sh.n), int(sh.kappa)): int(round(sh.occupation)) for sh in csf.subshells}


def _branch_for_keys(chain_keys, keys: tuple[tuple[int, int], ...]):
    indices = tuple(_fac_inner_shell_index(chain_keys, key) for key in keys)
    if any(index is None for index in indices):
        return None
    order, phase, branch = fac_angular_zxz0_branch_case(indices)
    return indices, order, phase, branch


def _one_diff_rows(panel: str, bra, ket) -> list[BranchRow]:
    diffs = bra.differs_from(ket)
    lost = [(key, qs, qo) for key, qs, qo in diffs if qs > qo]
    gained = [(key, qs, qo) for key, qs, qo in diffs if qs < qo]
    if len(lost) != 1 or len(gained) != 1:
        return []

    chain_keys = aligned_subshell_keys_from_simple_csfs(bra, ket)
    if chain_keys is None:
        return []
    occ_bra = _occ(bra)
    occ_ket = _occ(ket)
    key_a = lost[0][0]
    key_b = gained[0][0]
    rows: list[BranchRow] = []
    for key_c in sorted((set(occ_bra) | set(occ_ket)) - {key_a, key_b}):
        q_common = min(occ_bra.get(key_c, 0), occ_ket.get(key_c, 0))
        if q_common <= 0:
            continue
        k_values = sorted(
            set(exchange_k_range(key_a[1], key_c[1]))
            & set(exchange_k_range(key_c[1], key_b[1]))
        )
        k_values = [
            k for k in k_values
            if exchange_lambda(key_a[1], key_c[1], k) > 1e-15
            and exchange_lambda(key_c[1], key_b[1], k) > 1e-15
        ]
        if not k_values:
            continue
        branch_info = _branch_for_keys(chain_keys, (key_a, key_b, key_c, key_c))
        if branch_info is None:
            continue
        indices, order, phase, branch = branch_info
        supported_ranks = []
        zero_ranks = []
        exact_chain_values = []
        composite_seed_values = []
        fallback_unit_coeff = 0.0
        tj_c = 2 * abs(key_c[1]) - 1
        cap_c = 2 * abs(key_c[1])
        closed_spectator = int(round(q_common)) == cap_c
        fac_inner_keys = tuple(
            sorted(chain_keys, key=lambda item: (int(item[0]), abs(int(item[1])), int(item[1])))
        )
        for k in k_values:
            fac_supported = _fac_one_diff_open_spectator_exchange_coef(
                bra, ket, key_a, key_b, key_c, k
            ) is not None
            production_zero = closed_spectator and k > 0 and key_a[1] != key_b[1]
            if fac_supported:
                supported_ranks.append(k)
            elif production_zero:
                # Mirrors _one_diff_element's production skip: a closed
                # spectator has no non-scalar W^K tensor, so different-kappa
                # exchange through it is a computed zero, not an averaged
                # fallback term.
                zero_ranks.append(k)
            else:
                exact = None
                if branch in (7, 8):
                    exact = _chain_one_diff_spectator_exact_scalar(
                        bra, ket, fac_inner_keys, key_a, key_b, key_c, k
                    )
                if exact is not None:
                    exact_chain_values.append(f"k={k}:{exact:.12g}")
                if branch in (7, 8):
                    bra_active = single_state_for_subshell(2 * abs(key_a[1]) - 1, occ_bra.get(key_a, 0))
                    ket_active = single_state_for_subshell(2 * abs(key_a[1]) - 1, occ_ket.get(key_a, 0))
                    kind = "WxA" if branch == 8 else "AxW"
                    if bra_active is not None and ket_active is not None:
                        seed = fac_simple_axw_wxa_zxz0_postprocess(
                            kind,
                            bra_active,
                            ket_active,
                            R2=2 * abs(key_a[1]) - 1,
                            recouple_operators=branch,
                            order=order,
                            phase=phase,
                            shell_j2=(
                                2 * abs(key_a[1]) - 1,
                                2 * abs(key_b[1]) - 1,
                                2 * abs(key_c[1]) - 1,
                                2 * abs(key_c[1]) - 1,
                            ),
                            kk=(2 * k,),
                            kk1=(2 * k,),
                        )
                        if seed is not None:
                            composite_seed_values.append(f"{kind}:k={k}:{seed[0]:.12g}")
                lam_ac = exchange_lambda(key_a[1], key_c[1], k)
                lam_cb = exchange_lambda(key_c[1], key_b[1], k)
                if lam_ac > 1.0e-15 and lam_cb > 1.0e-15:
                    ang_k = lam_ac if key_a[1] == key_b[1] else math.sqrt(lam_ac * lam_cb)
                    fallback_unit_coeff += abs(q_common / (tj_c + 1.0) * ang_k)
        if len(supported_ranks) + len(zero_ranks) == len(k_values):
            support = "fac-angular"
            if supported_ranks and zero_ranks:
                note = "production FAC coefficient plus closed-spectator zero"
            elif zero_ranks:
                note = "production closed-spectator FAC zero"
            else:
                note = "production FAC coefficient"
        elif branch == 8 and int(round(q_common)) == cap_c and key_a[1] == key_b[1]:
            support = "closed-spectator-fallback"
            note = "closed-shell same-kappa tensor needs full branch-8 recoupling"
        elif branch == 8 and int(round(q_common)) == cap_c - 1:
            support = "closed-spectator-fallback"
            note = "single-hole branch-8 decouple-zero kept fallback pending coupling metadata"
        elif branch in (7, 8) and q_common != 1:
            support = "closed-spectator-fallback"
            note = "branch known; remaining multi-occupancy spectator ranks use fallback"
        elif branch in (7, 8) and key_a[1] == key_b[1]:
            support = "same-kappa-fallback"
            note = "same-kappa one-diff term uses the averaged fallback"
        elif branch == 7:
            support = "fac-branch-shape-missing"
            note = "branch-7 decouple-zero kept fallback pending full parentage"
        elif branch == 8:
            support = "fac-branch-shape-missing"
            note = "branch known but parentage/order unsupported"
        else:
            support = "fallback"
            note = "no production AngularZxZ0 path"
        rows.append(BranchRow(
            panel=panel,
            interaction="one-diff",
            support=support,
            branch=branch,
            order=" ".join(str(i) for i in order),
            phase=phase,
            j2=int(bra.total_J2),
            parity=int(bra.parity),
            bra=bra.label,
            ket=ket.label,
            shells=_keys_text((key_a, key_b, key_c, key_c)),
            ranks=" ".join(str(k) for k in k_values),
            fallback_unit_coeff=float(fallback_unit_coeff),
            exact_chain_scalar=" ".join(exact_chain_values),
            composite_seed=" ".join(composite_seed_values),
            note=note,
        ))
    return rows


def _two_diff_rows(panel: str, bra, ket) -> list[BranchRow]:
    diffs = bra.differs_from(ket)
    lost = [(key, qs, qo) for key, qs, qo in diffs if qs > qo]
    gained = [(key, qs, qo) for key, qs, qo in diffs if qs < qo]
    if len(lost) != 2 or len(gained) != 2:
        return []

    chain_keys = aligned_subshell_keys_from_simple_csfs(bra, ket)
    if chain_keys is None:
        return []
    key_a, key_b = lost[0][0], lost[1][0]
    key_c, key_d = gained[0][0], gained[1][0]
    # FAC stores two-diff operators as s0=a, s1=c, s2=b, s3=d.
    fac_keys = (key_a, key_c, key_b, key_d)
    branch_info = _branch_for_keys(chain_keys, fac_keys)
    if branch_info is None:
        return []
    _indices, order, phase, branch = branch_info
    fac_value = _fac_two_diff_branch9_element(
        bra, ket, key_a, key_b, key_c, key_d, {}, {}, 1, unit_radials=True
    )
    support = "fac-branch9-pending" if fac_value is not None else "fallback"
    note = (
        "branch-9 shell shape is identified; full FAC SumCoeff coefficient is pending"
        if fac_value is not None
        else "two-diff production still uses averaged Slater fallback"
    )

    rows: list[BranchRow] = []
    for label, keys in (
        ("two-diff-exchange", (key_a, key_b, key_d, key_c)),
        ("two-diff-direct", (key_a, key_b, key_c, key_d)),
    ):
        if label.endswith("exchange"):
            ranks = sorted(
                set(exchange_k_range(key_a[1], key_d[1]))
                & set(exchange_k_range(key_b[1], key_c[1]))
            )
        else:
            ranks = sorted(
                set(exchange_k_range(key_a[1], key_c[1]))
                & set(exchange_k_range(key_b[1], key_d[1]))
            )
        rows.append(BranchRow(
            panel=panel,
            interaction=label,
            support=support,
            branch=branch,
            order=" ".join(str(i) for i in order),
            phase=phase,
            j2=int(bra.total_J2),
            parity=int(bra.parity),
            bra=bra.label,
            ket=ket.label,
            shells=_keys_text(keys),
            ranks=" ".join(str(k) for k in ranks),
            note=note,
        ))
    return rows


def _diagonal_rows(panel: str, csf) -> list[BranchRow]:
    if not _has_explicit_J(csf):
        return []
    chain_keys = aligned_subshell_keys_from_simple_csfs(csf, csf)
    if chain_keys is None:
        return []
    occupied = sorted(_occ(csf))
    rows: list[BranchRow] = []
    for i, key_a in enumerate(occupied):
        for key_b in occupied[i:]:
            branch_info = _branch_for_keys(chain_keys, (key_a, key_b, key_a, key_b))
            if branch_info is None:
                continue
            _indices, order, phase, branch = branch_info
            support = "fac-angular" if branch in (0, 6) else "fallback"
            rows.append(BranchRow(
                panel=panel,
                interaction="diagonal",
                support=support,
                branch=branch,
                order=" ".join(str(i) for i in order),
                phase=phase,
                j2=int(csf.total_J2),
                parity=int(csf.parity),
                bra=csf.label,
                ket=csf.label,
                shells=_keys_text((key_a, key_b, key_a, key_b)),
                ranks="",
                note="diagonal pair inventory",
            ))
    return rows


def inventory_panel(panel: str, configs: Iterable[str], *, include_diagonal: bool = True) -> list[BranchRow]:
    csfs = []
    for config in configs:
        csfs.extend(enumerate_jj_csfs(config))

    rows: list[BranchRow] = []
    if include_diagonal:
        for csf in csfs:
            rows.extend(_diagonal_rows(panel, csf))

    for i, bra in enumerate(csfs):
        for ket in csfs[i + 1:]:
            if not _same_block(bra, ket):
                continue
            n_diff = bra.n_electron_diff(ket)
            if n_diff == 1:
                rows.extend(_one_diff_rows(panel, bra, ket))
                rows.extend(_one_diff_rows(panel, ket, bra))
            elif n_diff == 2:
                rows.extend(_two_diff_rows(panel, bra, ket))
                rows.extend(_two_diff_rows(panel, ket, bra))
    return rows


def print_summary(rows: Iterable[BranchRow]) -> None:
    rows = list(rows)
    by_interaction = Counter(row.interaction for row in rows)
    by_branch = Counter((row.interaction, row.branch, row.support) for row in rows)
    unsupported = [row for row in rows if row.support != "fac-angular"]

    print("Interaction counts:")
    for key, count in sorted(by_interaction.items()):
        print(f"  {key}: {count}")
    print("\nBranch/support counts:")
    for (interaction, branch, support), count in sorted(by_branch.items()):
        print(f"  {interaction:17s} branch={branch:<2d} {support:26s} {count}")
    print(f"\nRows without production FAC angular coefficient: {len(unsupported)} / {len(rows)}")

    if unsupported:
        print("\nUnsupported fallback unit-coefficient weight:")
        grouped: dict[tuple[str, int, str], list[BranchRow]] = {}
        for row in unsupported:
            grouped.setdefault((row.interaction, row.branch, row.support), []).append(row)
        for (interaction, branch, support), group in sorted(grouped.items()):
            total = sum(row.fallback_unit_coeff for row in group)
            peak = max(row.fallback_unit_coeff for row in group)
            print(
                f"  {interaction:17s} branch={branch:<2d} {support:26s} "
                f"count={len(group):<3d} sum={total:.12g} max={peak:.12g}"
            )

        print("\nUnsupported blocker notes:")
        by_note: dict[tuple[str, int, str], list[BranchRow]] = {}
        for row in unsupported:
            by_note.setdefault((row.interaction, row.branch, row.note), []).append(row)
        for (interaction, branch, note), group in sorted(by_note.items()):
            total = sum(row.fallback_unit_coeff for row in group)
            peak = max(row.fallback_unit_coeff for row in group)
            print(
                f"  {interaction:17s} branch={branch:<2d} count={len(group):<3d} "
                f"sum={total:.12g} max={peak:.12g} note={note}"
            )

        seed_rows = [row for row in unsupported if row.composite_seed]
        if seed_rows:
            print("\nUnsupported FAC composite-seed diagnostics:")
            grouped_seed: dict[tuple[str, int, str, str], list[BranchRow]] = {}
            for row in seed_rows:
                grouped_seed.setdefault(
                    (row.interaction, row.branch, row.support, row.composite_seed),
                    [],
                ).append(row)
            for (interaction, branch, support, seed), group in sorted(grouped_seed.items()):
                total = sum(row.fallback_unit_coeff for row in group)
                print(
                    f"  {interaction:17s} branch={branch:<2d} {support:26s} "
                    f"count={len(group):<3d} sum={total:.12g} seed={seed}"
                )

        exact_rows = [row for row in unsupported if row.exact_chain_scalar]
        if exact_rows:
            print("\nUnsupported exact-chain scalar diagnostics:")
            grouped_exact: dict[tuple[str, int, str, str], list[BranchRow]] = {}
            for row in exact_rows:
                grouped_exact.setdefault(
                    (row.interaction, row.branch, row.support, row.exact_chain_scalar),
                    [],
                ).append(row)
            for (interaction, branch, support, scalar), group in sorted(grouped_exact.items()):
                total = sum(row.fallback_unit_coeff for row in group)
                print(
                    f"  {interaction:17s} branch={branch:<2d} {support:26s} "
                    f"count={len(group):<3d} sum={total:.12g} exact={scalar}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=sorted(DEFAULT_PANELS), action="append")
    parser.add_argument("--config", action="append", help="custom configuration; may be repeated")
    parser.add_argument("--name", default="custom", help="custom panel name with --config")
    parser.add_argument("--no-diagonal", action="store_true")
    parser.add_argument("--out", type=Path, help="optional CSV output")
    args = parser.parse_args()

    selected: dict[str, tuple[str, ...]] = {}
    if args.config:
        selected[args.name] = tuple(args.config)
    else:
        names = args.panel or list(DEFAULT_PANELS)
        selected = {name: DEFAULT_PANELS[name] for name in names}

    rows: list[BranchRow] = []
    for name, configs in selected.items():
        rows.extend(inventory_panel(name, configs, include_diagonal=not args.no_diagonal))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].as_dict()) if rows else list(BranchRow.__dataclass_fields__))
            writer.writeheader()
            for row in rows:
                writer.writerow(row.as_dict())
    print_summary(rows)


if __name__ == "__main__":
    main()
