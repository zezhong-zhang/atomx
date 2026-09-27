"""Cached NIST diagnostic failures must retain their original HTTP status."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def test_offline_diagnostic_audit_preserves_http_error(tmp_path, monkeypatch):
    script_dir = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(script_dir))
    from scripts.audit_nist_diagnostics import PROBES

    raw_dir = tmp_path / "official_diagnostics_probe"
    raw_dir.mkdir()
    previous = []
    for name, _spectrum, fmt, _line_out, _include in PROBES:
        body = b"<html>cached NIST response</html>"
        suffix = "tsv" if fmt == "3" else "html"
        (raw_dir / f"{name}.{suffix}").write_bytes(body)
        previous.append({"name": name, "sha256": hashlib.sha256(body).hexdigest(),
                         "http_status": 500 if name == "c_iii_minimal_diag_html"
                         else 200})
    report = tmp_path / "nist_diagnostics_audit.json"
    report.write_text(json.dumps({"probes": previous}))
    script = script_dir / "audit_nist_diagnostics.py"
    result = subprocess.run([sys.executable, str(script), "--offline",
                             "--out", str(tmp_path)],
                            capture_output=True, text=True, check=True)
    current = json.loads(report.read_text())
    statuses = {row["name"]: row["http_status"]
                for row in current["probes"]}
    assert statuses["c_iii_minimal_diag_html"] == 500
    assert "c_iii_minimal_diag_html 500" in result.stdout
