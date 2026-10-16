"""Write or verify the owner-only archive identity without touching scientific files."""
import hashlib
import json
import sys
from pathlib import Path

OWNER = Path(__file__).resolve().parent
MANIFEST = OWNER / "artifact_sha256.json"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def members():
    return sorted(path for path in OWNER.rglob("*") if path.is_file()
                  and path != MANIFEST and "__pycache__" not in path.parts
                  and path.suffix != ".pyc")


def main():
    if "--verify" in sys.argv[1:]:
        files = json.loads(MANIFEST.read_text())["files"]
        actual = {str(path.relative_to(OWNER)) for path in members()}
        if actual != set(files):
            raise AssertionError("owner archive member set changed")
        for name, identity in files.items():
            path = OWNER / name
            if path.stat().st_size != identity["bytes"] or digest(path) != identity["sha256"]:
                raise AssertionError(f"owner archive identity changed: {name}")
        print(json.dumps({"files": len(files), "verified": True}))
        return
    files = {str(path.relative_to(OWNER)): {"bytes": path.stat().st_size, "sha256": digest(path)}
             for path in members()}
    MANIFEST.write_text(json.dumps({"scope": "Every owner file except this self-referential manifest and generated Python caches; no external artifact is silently claimed archived.", "files": files}, indent=2) + "\n")
    print(json.dumps({"files": len(files), "bytes": sum(item["bytes"] for item in files.values())}))


if __name__ == "__main__":
    main()
