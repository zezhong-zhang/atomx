"""Official NIST exports can omit levels in one ordering or both tables."""

import csv
import sqlite3
from pathlib import Path

from scripts.build_nist_spectra import SCHEMA
from scripts.import_nist_level_details import prepare_database
from scripts.import_nist_line_only_levels import import_missing
from scripts.import_nist_term_order_levels import SOURCE_SCHEMA, import_one


def _write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_term_order_recovers_unique_level_id_without_changing_existing_row(tmp_path):
    db = sqlite3.connect(":memory:")
    db.executescript(SCHEMA)
    prepare_database(db)
    db.execute("ALTER TABLE levels ADD COLUMN source_variant TEXT DEFAULT 'energy_order'")
    db.execute(SOURCE_SCHEMA)
    db.execute("""INSERT INTO species(z,charge,symbol,label,line_listed,
        level_listed,line_status,level_status,line_rows,usable_wavelengths,
        observed_wavelengths,ritz_only_wavelengths,aki_rows,level_rows,
        line_parse_errors,level_parse_errors)
        VALUES (35,3,'Br','Br IV',0,1,'missing','parsed',0,0,0,0,0,1,0,0)""")
    db.execute("""INSERT INTO levels(z,charge,row_index,configuration_nist,
        term,j_text,energy_cm1,bracketed,level_id)
        VALUES (35,3,0,'4s2.4p.6s','1P*','1',265438,0,'id1')""")
    db.execute("INSERT INTO source_count_discrepancies VALUES ('levels',35,3,2,1)")
    first = {"Configuration": "4s2.4p.6s", "Term": "1P*", "J": "1", "g": "3",
             "Prefix": "", "Level (cm-1)": "265438", "Suffix": "",
             "Uncertainty (cm-1)": "", "Splitting": "", "Lande": "",
             "Leading percentages": "", "Level ID": "id1", "Reference": ""}
    second = {**first, "Configuration": "", "Term": "1*", "J": "2",
              "g": "5", "Level (cm-1)": "265961", "Level ID": "id2"}
    name = "asd.035003.tsv"
    _write_tsv(tmp_path / "official_level_details" / name, [first])
    _write_tsv(tmp_path / "official_level_term_order" / name, [first, second])

    assert import_one(db, tmp_path, 35, 3)["new_rows"] == 1
    assert db.execute("""SELECT row_index,configuration_nist,energy_cm1,
        source_variant FROM levels WHERE level_id='id2'""").fetchone() == (
            1, "", 265961, "term_order")
    assert import_one(db, tmp_path, 35, 3)["new_rows"] == 0
    assert db.execute("SELECT level_rows FROM species").fetchone()[0] == 2


def test_line_only_level_keeps_unreported_energy_and_j_null():
    db = sqlite3.connect(":memory:")
    db.executescript(SCHEMA)
    db.execute("ALTER TABLE levels ADD COLUMN level_id TEXT")
    db.execute("ALTER TABLE lines ADD COLUMN lower_level_id TEXT")
    db.execute("ALTER TABLE lines ADD COLUMN upper_level_id TEXT")
    db.execute("""INSERT INTO species(z,charge,symbol,label,line_listed,
        level_listed,line_status,level_status,line_rows,usable_wavelengths,
        observed_wavelengths,ritz_only_wavelengths,aki_rows,level_rows,
        line_parse_errors,level_parse_errors)
        VALUES (56,24,'Ba','Ba XXV',1,1,'parsed','parsed',1,1,1,0,0,0,0,0)""")
    db.execute("""INSERT INTO lines(z,charge,row_index,source_row,
        lower_level_id,lower_configuration,line_reference)
        VALUES (56,24,0,123,'line-only','3d10 4s2 4p2','L11790')""")

    assert import_missing(db)["added"] == 1
    assert db.execute("""SELECT configuration,energy_cm1,j_text,
        source_variant FROM levels WHERE level_id='line-only'""").fetchone() == (
            "3d10 4s2 4p2", None, None, "line_endpoint")
    assert db.execute("""SELECT line_row_index,source_row FROM
        nist_line_only_level_sources""").fetchone() == (0, 123)
    assert import_missing(db)["added"] == 0
