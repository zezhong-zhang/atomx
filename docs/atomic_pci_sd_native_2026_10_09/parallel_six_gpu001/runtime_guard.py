"""Fixed dependency identity checks before CUDA import and after use."""

import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def verify(manifest):
    manifest = Path(manifest)
    entries = json.loads(manifest.read_text())["dependencies"]
    for name, identity in entries.items():
        path = Path(name)
        if (
            path.stat().st_size != identity["bytes"]
            or digest(path) != identity["sha256"]
        ):
            raise RuntimeError(
                f"fixed CUDA implementation/runtime dependency changed: {name}"
            )
    return dict(
        manifest_sha256=digest(manifest),
        complete_dependency_count=len(entries),
        matched=True,
    )
