"""Private atomic storage of complete, uncertified numerical restart guesses."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np


def plain(value):
    if hasattr(value, "items"):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical(value):
    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            result.update(chunk)
    return result.hexdigest()


class RestartStorage:
    """Immutable NPZ generations, committed by an atomic stage-specific head."""

    def __init__(self, directory, binding, *, cap_bytes=8 * 1024**3):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.binding = plain(binding)
        self.cap = cap_bytes
        self.used = sum(p.stat().st_size for p in self.directory.glob("*.npz"))

    def __call__(self, snapshot):
        metadata = plain({k: v for k, v in snapshot.items() if k != "numerators"})
        stage = metadata["stage"]
        if stage not in ("root", "response") or metadata["certified"] is not False:
            raise ValueError("uncertified root/response numerical guess required")
        vector = snapshot["numerators"]
        if (
            vector.ndim != 1
            or vector.dtype != np.float64
            or not np.all(np.isfinite(vector))
        ):
            raise ValueError("complete finite real numerical guess required")
        envelope = dict(binding=self.binding, snapshot=metadata)
        estimate = vector.nbytes + len(canonical(envelope).encode()) + 65536
        if (
            self.used + estimate > self.cap
            or shutil.disk_usage(self.directory).free < 2 * estimate
        ):
            raise MemoryError(
                "complete restart storage exceeds declared cap/free space"
            )
        name = f"{stage}_{metadata['ordinal']:06d}.npz"
        target = self.directory / name
        if target.exists():
            raise ValueError("immutable restart generation already exists")
        temporary = target.with_suffix(".npz.partial")
        with temporary.open("xb") as stream:
            np.savez(stream, numerators=vector, metadata=np.array(canonical(envelope)))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        self.used += target.stat().st_size
        head = dict(file=name, sha256=digest(target))
        temporary = self.directory / f"{stage}.json.partial"
        with temporary.open("w") as stream:
            stream.write(json.dumps(head, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.directory / f"{stage}.json")
        # The payload was committed before its pointer. Persist both renames.
        descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def load(self, stage, expected_identity, count):
        head_path = self.directory / f"{stage}.json"
        if not head_path.exists():
            return None
        head = json.loads(head_path.read_text())
        name = head["file"]
        if (
            Path(name).name != name
            or not name.startswith(stage + "_")
            or not name.endswith(".npz")
        ):
            raise ValueError("bound restart generation filename required")
        payload = self.directory / name
        if payload.stat().st_size > self.cap:
            raise MemoryError("restart archive exceeds declared cap")
        if digest(payload) != head["sha256"]:
            raise ValueError("restart generation hash differs")
        with np.load(payload, allow_pickle=False) as data:
            envelope = json.loads(str(data["metadata"].item()))
            vector = data["numerators"].copy()
        metadata = envelope["snapshot"]
        if canonical(envelope["binding"]) != canonical(self.binding) or any(
            canonical(metadata[k]) != canonical(v) for k, v in expected_identity.items()
        ):
            raise ValueError(
                "complete restart physical/source/coordinate identity differs"
            )
        if metadata["stage"] != stage or metadata["certified"] is not False:
            raise ValueError("uncertified same-stage numerical guess required")
        if (
            vector.shape != (count,)
            or vector.dtype != np.float64
            or not np.all(np.isfinite(vector))
        ):
            raise ValueError("complete finite real saved numerical guess required")
        return vector
