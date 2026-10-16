"""Hash current owner reports and preserved failures after numerical review."""

import hashlib
import json
import re
from pathlib import Path

OWNER = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(part)
    return digest.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    for target in re.findall(r"\]\(([^)]+)\)", (OWNER / "README.md").read_text()):
        assert (OWNER / target).exists(), target
    assert all(
        not line.rstrip("\n").endswith((" ", "\t"))
        for path in OWNER.glob("*.md")
        for line in path.read_text().splitlines(keepends=True)
    )
    # Existing run result/contract/output manifests are unchanged; review
    # manifests are appended in the owner, not written into archived epochs.
    failures = {}
    for epoch in ("run001", "run002", "thread_preflight001", "thread_preflight002"):
        hashes = {
            str(path.relative_to(OWNER)): sha(path)
            for path in sorted((OWNER / epoch).rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
        }
        failures[epoch] = {
            "files": hashes,
            "disposition": {
                "run001": "Requested-only thread budget failed; original archived scientific outputs retained; requested p4/p8 performance rejected/null.",
                "run002": "Scientific arrays/source gates retained; public-path latency observations retained; backend algorithm timing/efficiency inference withdrawn by separate review.",
                "thread_preflight001": "Stricter total thread count=requested team rejection at p2; not falsely labelled parent 8-thread excess.",
                "thread_preflight002": "Original parent thread cap rejected p4 at 9 actual threads; p1/p2 complete signed output passed.",
            }[epoch],
        }
    dump(OWNER / "failed_attempts_manifest.json", failures)
    metadata = {
        "status": "owner_evidence_sealed",
        "authoritative_summary": "result.json",
        "authoritative_run": "run003",
        "full_validation": "validation.json",
        "historical_failure_manifest": "failed_attempts_manifest.json",
        "run002_review_scope": "Deployed public timeout-polling latency is real and retained; inferential backend cost/parallel-efficiency claim is withdrawn. run003 uses declared private blocking-wait instrumentation.",
        "scope": "Owner directory only; no production/reference/default edits or shared guide/trace mutation.",
        "highest_level_reached": False,
        "performance_ranking": None,
    }
    dump(OWNER / "handoff.json", metadata)
    files = {
        str(path.relative_to(OWNER)): sha(path)
        for path in sorted(OWNER.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.name != "final_owner_sha256.json"
    }
    dump(OWNER / "final_owner_sha256.json", files)
    assert all(sha(OWNER / path) == digest for path, digest in files.items())
    print(
        json.dumps(
            {
                "status": metadata["status"],
                "verified_final_owner_files": len(files),
                "final_owner_manifest_sha256": sha(OWNER / "final_owner_sha256.json"),
                "result_sha256": files["result.json"],
                "failure_manifest_sha256": files["failed_attempts_manifest.json"],
            }
        )
    )


if __name__ == "__main__":
    main()
