"""Export a read-only NIST snapshot and explicitly classified saved calculations.

No archive mutation, implicit download, state calculation, or accuracy promotion.
"""

import argparse
import csv
import gzip
import hashlib
import json
import resource
import shutil
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from atomx.reference_data.nist import Level, normalise_configuration

REPO = Path(__file__).resolve().parents[1]
SYMBOLS = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
    )


def positions():
    result = {}
    for period, numbers in enumerate(
        [
            [1] + [0] * 16 + [2],
            [3, 4] + [0] * 10 + list(range(5, 11)),
            [11, 12] + [0] * 10 + list(range(13, 19)),
            list(range(19, 37)),
            list(range(37, 55)),
            [55, 56, 0] + list(range(72, 87)),
            [87, 88, 0] + list(range(104, 119)),
        ],
        1,
    ):
        for group, z in enumerate(numbers, 1):
            if z:
                result[z] = [period, group]
    for start, row in [(57, 9), (89, 10)]:
        for offset in range(15):
            result[start + offset] = [row, offset + 3]
    assert set(result) == set(range(1, 119))
    return result


def level_record(r):
    return Level(
        ion="",
        z=r["z"],
        charge=r["charge"],
        index=r["row_index"],
        row=r["source_row"] or 0,
        configuration=r["configuration"] or "",
        configuration_nist=r["configuration_nist"] or "",
        term=r["term"] or "",
        j_text=r["j_text"] or "",
        energy_cm1=r["energy_cm1"],
        uncertainty_cm1=r["uncertainty_cm1"],
        reference=r["reference"] or "",
        flags=tuple(json.loads(r["flags"] or "[]")),
        bracketed=bool(r["bracketed"]),
        energy_kind=r["energy_kind"] or "missing",
    )


def match_yb(rows, levels, *, dynamic=False):
    """Unique nominal branches only; do not certify heavily mixed wavefunctions."""
    zero = [
        r
        for r in levels
        if normalise_configuration(r["configuration"]) == "4f14 6s1"
        and level_record(r).two_j == 1
        and level_record(r).parity == 1
        and r["energy_kind"] == "evaluated"
    ]
    if len(zero) != 1 or zero[0]["energy_cm1"] != 0:
        raise ValueError("Yb II common evaluated 6s zero is not unique")
    out = []
    for row in rows:
        if row["key"] == [6, -1]:
            continue
        n, k = row["key"]
        l = abs(k) - (k < 0)
        configuration = f"4f14 {n}{'spdf'[l]}1"
        parity = row["parity"] if dynamic else (-1 if row["parity"] == "odd" else 1)
        two_j = row["two_J"] if dynamic else row["twoJ"]
        candidates = [
            r
            for r in levels
            if normalise_configuration(r["configuration"]) == configuration
            and level_record(r).two_j == two_j
            and level_record(r).parity == parity
            and r["energy_kind"] == "evaluated"
            and r["energy_cm1"] is not None
            and not r["bracketed"]
            and not level_record(r).flags
            and "?" not in (r["configuration_nist"] or "") + (r["term"] or "")
        ]
        value = row["excitation_cm1"]["dynamic"] if dynamic else row["computed_cm1"]
        entry = dict(
            configuration=configuration,
            two_j=two_j,
            parity=parity,
            calculated_cm1=value,
            matching="unique_nominal_branch"
            if len(candidates) == 1
            else "ambiguous_or_missing",
            candidates=len(candidates),
            handbook_cm1=row["experimental_cm1"],
        )
        if len(candidates) == 1:
            r = candidates[0]
            reference = r["energy_cm1"]
            error = value - reference
            entry.update(
                nist_row=r["row_index"],
                level_id=r["level_id"],
                reference_cm1=reference,
                uncertainty_cm1=r["uncertainty_cm1"],
                reference_kind=r["energy_kind"],
                reference_code=r["reference"],
                error_cm1=error,
                error_percent=100 * error / reference if reference else None,
                relative_goal=0.05 if dynamic else 0.01,
                goal_pass=abs(error / reference) <= (0.05 if dynamic else 0.01),
                asd_minus_handbook_cm1=reference - row["experimental_cm1"],
            )
        out.append(entry)
    return out


def export(archive, output, evidence, snapshot):
    if output.exists():
        raise FileExistsError(
            "Use a new output directory; retained exports are immutable"
        )
    evidence.mkdir(parents=True, exist_ok=True)
    if snapshot.exists():
        raise FileExistsError(snapshot)
    source = sqlite3.connect(archive.resolve().as_uri() + "?mode=ro", uri=True)
    with sqlite3.connect(snapshot) as frozen:
        source.backup(frozen)
    source.close()
    db = sqlite3.connect(snapshot.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    db.execute("BEGIN")
    metadata = dict(db.execute("select key,value from metadata"))
    counts = {
        table: db.execute(f"select count(*) from {table}").fetchone()[0]
        for table in ["levels", "lines", "nist_ionization", "source_files"]
    }

    def payload(z, charge):
        result = {}
        for table in counts:
            category = {"nist_ionization": "ionization", "source_files": "sources"}.get(
                table, table
            )
            records = []
            for raw in db.execute(
                f"select * from {table} where z=? and charge=?", (z, charge)
            ):
                r = dict(raw)
                for key in ["source_fields_json", "leading_percentages"]:
                    r.pop(key, None)
                records.append(r)
            result[category] = records
        result["history"] = [
            json.loads(r[0])
            for r in ledger.execute(
                "select record from records where z=? and charge=? order by row_index",
                (z, charge),
            )
        ]
        return result

    history = REPO / "output/nist_spectra/theory_comparison/all_levels_dhf_nist.csv"
    history_hash = digest(history)
    ledger = sqlite3.connect(evidence / "history_snapshot.sqlite")
    ledger.execute(
        "create table records(z int,charge int,row_index int,record text,primary key(z,charge,row_index))"
    )
    history_statuses = Counter()
    with history.open() as f:
        for r in csv.DictReader(f):
            ledger.execute(
                "insert into records values(?,?,?,?)",
                (int(r["z"]), int(r["charge"]), int(r["row_index"]), json.dumps(r)),
            )
            history_statuses[r["status"]] += 1
    ledger.commit()
    if digest(history) != history_hash:
        raise ValueError("historical CSV changed during export")
    yb = payload(70, 1)
    owner = REPO / "docs/atomic_pci_sd_native_2026_10_09"
    models = []
    files = []
    for name, dynamic in [
        ("yb_monovalent_experiment003/comparison.json", False),
        ("yb_dynamic_breit003/result.json", True),
    ]:
        path = owner / name
        raw = path.read_bytes()
        result = json.loads(raw)
        models.append(
            dict(
                title="冻结轨道 CI + 动态横向交换"
                if dynamic
                else "原生 Dirac–Coulomb 线性全阶 SD",
                saved_run="yb_dynamic_breit003" if dynamic else result["computed_run"],
                source=str(path.relative_to(REPO)),
                sha256=hashlib.sha256(raw).hexdigest(),
                scope=result["scope"],
                highest_level_reached=False,
                rows=match_yb(result["rows"], yb["levels"], dynamic=dynamic),
            )
        )
        files.append(
            dict(
                path=str(path.relative_to(REPO)), sha256=hashlib.sha256(raw).hexdigest()
            )
        )
    yb["models"] = models
    ion_metadata = defaultdict(list)
    numeric_by_element = Counter()
    for r in db.execute("select * from species order by z,charge"):
        row = dict(r)
        key = row["z"], row["charge"]
        p = yb if key == (70, 1) else payload(*key)
        row["history_statuses"] = dict(Counter(x["status"] for x in p["history"]))
        row["saved_unique_comparisons"] = row["history_statuses"].get(
            "unique_level_comparison", 0
        )
        numeric_by_element[row["z"]] += row["saved_unique_comparisons"]
        row["ionization"] = p["ionization"]
        row["asset"] = (
            f"ions/{key[0]}-{key[1]}.json.gz"
            if p["levels"] or p["lines"] or p["history"] or p.get("models")
            else None
        )
        if row["asset"]:
            asset = output / row["asset"]
            asset.parent.mkdir(parents=True, exist_ok=True)
            asset.write_bytes(
                gzip.compress(
                    json.dumps(
                        p, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                    ).encode(),
                    mtime=0,
                )
            )
        ion_metadata[row["z"]].append(row)
    pos = positions()
    elements = []
    for z, symbol in enumerate(SYMBOLS, 1):
        ions = ion_metadata[z]
        elements.append(
            dict(
                z=z,
                symbol=symbol,
                position=pos[z],
                ions=ions,
                levels=sum(i["level_rows"] for i in ions),
                lines=sum(i["line_rows"] for i in ions),
                unique_history=numeric_by_element[z],
                highest_certified=False,
            )
        )
    mismatches = [
        dict(r) for r in db.execute("select * from source_count_discrepancies")
    ]
    index = dict(
        generated_at=datetime.now(timezone.utc).isoformat(),
        metadata=metadata,
        elements=elements,
        counts=dict(counts),
        history_statuses=dict(history_statuses),
        source_count_discrepancies=mismatches,
        provenance=dict(
            snapshot_sha256=digest(snapshot),
            history_sha256=history_hash,
            exporter_sha256=digest(Path(__file__)),
            model_files=files,
        ),
        scope="本地档案快照；历史 DHF 计算未按当前源码重新认证。NIST 含评估、理论、半经验及缺失记录。",
        highest_level_reached=False,
    )
    write(output / "index.json", index)
    (output / "index.html").write_bytes((REPO / "tools/nist_atlas.html").read_bytes())
    db.close()
    ledger.close()
    manifest = dict(
        counts=dict(counts),
        history_statuses=dict(history_statuses),
        elements=118,
        asset_files=len(list(output.rglob("*.json.gz"))) + 1,
        source_count_discrepancies=mismatches,
        snapshot_sha256=digest(snapshot),
        index_sha256=digest(output / "index.json"),
        models=models,
        source_files=files,
        highest_level_reached=False,
    )
    write(evidence / "export.json", manifest)
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in ["counts", "elements", "asset_files", "highest_level_reached"]
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output.parent).free < 2 * 1024**3:
        raise MemoryError("export requires 2 GiB free disk for snapshot and assets")
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    export(args.archive, args.output, args.evidence, args.snapshot)


if __name__ == "__main__":
    main()
