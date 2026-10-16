#!/usr/bin/env python3
"""Prepare an identified working-tree software snapshot for Zenodo, offline.

Copies selected files into a fresh directory, regenerates only the copy's
project navigation, builds a wheel, and writes a ZIP plus SHA-256 manifests.
Never stages, commits, deletes original files, uploads, or publishes anything.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

ALLOWED_ROOTS = {"src", "tests", "scripts", "tools", "docs", "benchmarks", ".github"}
ALLOWED_ROOT_FILES = {
    ".gitignore", ".mailmap", ".pre-commit-config.yaml", ".python-version",
    ".zenodo.json", "AGENTS.md", "CHANGELOG.md", "CITATION.cff", "LICENSE",
    "PROJECT_INDEX.json", "PROJECT_INDEX.md", "README.md", "conftest.py",
    "pyproject.toml", "uv.lock",
}
GENERATED_PARTS = {
    "__pycache__", ".git", ".venv", "build", "dist", ".pytest_cache",
    ".ruff_cache", "CMakeFiles", "openblas", "numba_cuda_runtime",
    "scratch", "extracted_bundle",
}
RAW_SUFFIXES = {
    ".npz", ".npy", ".h5", ".hdf5", ".dg", ".pkl", ".pstats",
    ".13", ".4", ".b", ".basis",
}
BINARY_SUFFIXES = {
    ".pyc", ".pyo", ".nbc", ".nbi", ".o", ".a", ".so", ".mod",
    ".x", ".cubin", ".deb", ".xz",
}


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(
            hashlib, "file_digest"
        ) else digest_stream(stream)


def digest_stream(stream) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def exclusion(name: str, tracked: set[str]) -> str | None:
    path = PurePosixPath(name)
    parts = path.parts
    if parts[0] not in ALLOWED_ROOTS and name not in ALLOWED_ROOT_FILES:
        return "outside software snapshot roots"
    # Explicitly tracked scientific fixtures can live under historical scratch
    # or source directories. Preserve those exact inputs before filtering local
    # build products; otherwise a clean source archive can lose runnable gates.
    if name in tracked:
        return None
    if any(part in GENERATED_PARTS or part.endswith("-build") for part in parts):
        return "generated build, cache, dependency, or extracted bundle"
    if path.suffix in BINARY_SUFFIXES:
        return "compiled binary, compiler cache, or local toolchain"
    # New experiment outputs are not implicit additions to the software deposit.
    if any(part.startswith("expt_") and "source" in part for part in parts[:-1]):
        return "copied optional upstream source tree"
    if "source" in parts or any(part.startswith("source_") for part in parts):
        return "untracked historical source snapshot"
    if path.suffix in RAW_SUFFIXES or path.name.startswith("ATOMX_"):
        return "untracked numerical research data"
    if (path.suffix in {".f", ".f90", ".c", ".h"} and "docs" in parts
            and any(part.startswith("backend") for part in parts)):
        return "generated case-specific backend source"
    return None


def inventory(root: Path) -> tuple[set[str], list[str]]:
    tracked = {p.decode() for p in git(root, "ls-files", "-z").split(b"\0") if p}
    candidates = {
        p.decode() for p in git(
            root, "ls-files", "--cached", "--others", "--exclude-standard", "-z"
        ).split(b"\0") if p
    }
    return tracked, sorted(candidates)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare(root: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    # An output under an included root could recursively package earlier runs.
    if output.is_relative_to(root) and output.relative_to(root).parts[0] != "dist":
        raise ValueError("Repository-local output must be under ignored dist/")
    version_source = (root / "src/atomx/__init__.py").read_text()
    version = re.search(r'^__version__ = "([^"]+)"', version_source, re.M)[1]
    metadata = json.loads((root / ".zenodo.json").read_text())
    if metadata["version"] != version:
        raise ValueError("Zenodo metadata and package version differ")
    tracked, names = inventory(root)
    before_status = git(root, "status", "--porcelain=v1", "-z")
    selected, excluded, identities = [], [], {}
    for name in names:
        source = root / name
        if source.is_symlink():
            if name in tracked or not name.startswith("docs/"):
                raise ValueError(f"Review symlink before packaging: {name}")
            excluded.append(dict(path=name, reason="untracked experiment symlink", bytes=0))
            continue
        if not source.exists():
            excluded.append(dict(path=name, reason="deleted from working tree", bytes=0))
            continue
        if not source.is_file():
            raise ValueError(f"Unsupported repository entry: {name}")
        info = source.stat()
        reason = exclusion(name, tracked)
        if reason:
            excluded.append(dict(path=name, reason=reason, bytes=info.st_size))
        else:
            selected.append(name)
            identities[name] = (info.st_size, info.st_mtime_ns, info.st_ino)
    required = {n for n in names if n.startswith(("src/", "tests/", "tools/"))
                and (root / n).is_file() and Path(n).suffix not in BINARY_SUFFIXES
                and "__pycache__" not in PurePosixPath(n).parts}
    if not required.issubset(selected):
        raise ValueError(f"Missing runtime/test inputs: {sorted(required - set(selected))}")
    totals = collections.defaultdict(lambda: dict(files=0, bytes=0))
    for record in excluded:
        totals[record["reason"]]["files"] += 1
        totals[record["reason"]]["bytes"] += record["bytes"]
    print(json.dumps(dict(selected_files=len(selected), selected_bytes=sum(
        identities[n][0] for n in selected
    ), excluded_summary=dict(totals)), indent=2), flush=True)
    output.mkdir(parents=True)
    snapshot = output / f"atomx-{version}"
    snapshot.mkdir()
    for name in selected:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, destination)
    # Regenerate against the actual deposited files, without touching the
    # user's navigation or Git index and without scanning omitted source copies.
    spec = importlib.util.spec_from_file_location(
        "release_navigation", snapshot / "scripts/build_project_index.py"
    )
    navigation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(navigation)
    inputs = {n: (snapshot / n).read_bytes() for n in selected if navigation.kind(n)}
    for name, content in navigation.build(inputs).items():
        (snapshot / name).write_bytes(content)
    records = []
    for name in selected:
        destination = snapshot / name
        record = dict(path=name, bytes=destination.stat().st_size, sha256=sha256(destination),
                      origin="tracked working tree" if name in tracked else "untracked working tree")
        if name in navigation.OUTPUTS:
            record.update(origin="regenerated navigation", original_sha256=sha256(root / name))
        records.append(record)
    manifest = dict(
        schema_version=1, package="atomx", version=version,
        repository="https://github.com/zezhong-zhang/atomx",
        base_commit=git(root, "rev-parse", "HEAD").decode().strip(),
        source_kind="working-tree snapshot including uncommitted files",
        git_status=before_status.decode().split("\0")[:-1],
        files=records, excluded_summary=dict(totals), excluded_files=excluded,
        policy="scripts/prepare_zenodo.py; no original files were deleted",
        manifest_scope="All copied files; RELEASE_MANIFEST.json and SOURCE_SHA256SUMS are generated separately.",
    )
    write_json(snapshot / "RELEASE_MANIFEST.json", manifest)
    sums = [f"{r['sha256']}  {r['path']}\n" for r in records]
    sums.append(f"{sha256(snapshot / 'RELEASE_MANIFEST.json')}  RELEASE_MANIFEST.json\n")
    (snapshot / "SOURCE_SHA256SUMS").write_text("".join(sums))
    # Build only from the isolated copy; the normal backend selects the wheel's
    # src/atomx payload. The ZIP is the complete curated repository distribution.
    with (output / "build.log").open("w") as log:
        subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(output)],
            cwd=snapshot, check=True, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "SOURCE_DATE_EPOCH": "315532800"},
        )
    # Reject concurrent software changes. New experiment sidecars may appear in
    # other sessions; they are outside the inventory-time snapshot and listed
    # explicitly, rather than retroactively added to a partly copied archive.
    after_tracked, after_names = inventory(root)
    added = sorted(set(after_names) - set(names))
    removed = sorted(set(names) - set(after_names))
    def software_input(name):
        path = PurePosixPath(name)
        return (path.parts[0] in {"src", "tests", "scripts", "tools", ".github"}
                or name in ALLOWED_ROOT_FILES
                or (path.parent == PurePosixPath("docs") and path.suffix == ".md"))
    if after_tracked != tracked or any(software_input(n) for n in added + removed):
        raise RuntimeError("Software file set changed during packaging; use a fresh output")
    copied_hashes = {r["path"]: r["sha256"] for r in records}
    for name, expected in identities.items():
        info = (root / name).stat()
        if (info.st_size, info.st_mtime_ns, info.st_ino) != expected:
            raise RuntimeError(f"Source changed during packaging: {name}")
        if name not in navigation.OUTPUTS and sha256(root / name) != copied_hashes[name]:
            raise RuntimeError(f"Source bytes changed during packaging: {name}")
    manifest["files_appeared_after_inventory"] = added
    manifest["files_disappeared_after_inventory"] = removed
    write_json(snapshot / "RELEASE_MANIFEST.json", manifest)
    sums[-1] = f"{sha256(snapshot / 'RELEASE_MANIFEST.json')}  RELEASE_MANIFEST.json\n"
    (snapshot / "SOURCE_SHA256SUMS").write_text("".join(sums))
    archive = output / f"atomx-{version}-source.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for relative in sorted([*selected, "RELEASE_MANIFEST.json", "SOURCE_SHA256SUMS"]):
            path = snapshot / relative
            name = path.relative_to(output).as_posix()
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if path.stat().st_mode & 0o111 else 0o100644) << 16
            with path.open("rb") as source, bundle.open(info, "w", force_zip64=True) as target:
                shutil.copyfileobj(source, target)
    shutil.copy2(snapshot / "RELEASE_MANIFEST.json", output / "RELEASE_MANIFEST.json")
    shutil.copy2(root / ".zenodo.json", output / "zenodo-metadata.json")
    write_json(output / "zenodo-deposition.json", {"metadata": metadata})
    shutil.copy2(root / "docs/ZENODO_RELEASE.md", output / "ZENODO_PAGE.md")
    artifacts = sorted(p for p in output.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (output / "SHA256SUMS").write_text("".join(f"{sha256(p)}  {p.name}\n" for p in artifacts))
    print(f"Prepared {archive} ({archive.stat().st_size:,} bytes)")
    print(f"Verify: cd {output} && sha256sum -c SHA256SUMS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory; use dist/ inside the repo")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    prepare(root, args.output.resolve())


if __name__ == "__main__":
    main()
