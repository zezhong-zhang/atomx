"""Package case evidence plus the actual dirty Atomx source; exclude caches only."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import zipfile

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    primary=json.loads((ROOT/"results/v68_box90_newton/source_identity.json").read_text())
    for name,value in primary["source_sha256"].items():
        assert sha(REPO/name)==value, name
    files=[p for p in (REPO/"src/atomx").rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc",".nbc",".nbi")]
    dependencies={name:importlib.metadata.version(name) for name in ["numpy","scipy","numba","matplotlib","mpmath","h5py","atomx"]}
    (ROOT/"calculation_environment.json").write_text(json.dumps({"python":platform.python_version(),"packages":dependencies,"reproduction_policy":"Use isolated source copy; original dirty library never overwritten."},indent=2)+"\n")
    source_manifest={str(p.relative_to(REPO)):sha(p) for p in files}
    for name in ("pyproject.toml","uv.lock","LICENSE","README.md"):
        source_manifest[name]=sha(REPO/name)
    (ROOT/"source_archive_manifest.json").write_text(json.dumps(source_manifest,ensure_ascii=False,indent=2)+"\n")
    archive=ROOT/"Atomx_source_replay.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,Path("atomx_replay")/p.relative_to(REPO))
        for name in ("pyproject.toml","uv.lock","LICENSE","README.md"):z.write(REPO/name,Path("atomx_replay")/name)
        for p in ROOT.rglob("*"):
            if p.is_file() and p.suffix==".py" and "__pycache__" not in p.parts:
                z.write(p,Path("atomx_replay/docs")/ROOT.name/p.relative_to(ROOT))
    notes=json.loads((ROOT/"provenance_notes.json").read_text())
    notes["script_hashes_at_delivery"]={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob("*.py") if "__pycache__" not in p.parts}
    (ROOT/"provenance_notes.json").write_text(json.dumps(notes,ensure_ascii=False,indent=2)+"\n")
    manifest={str(p.relative_to(ROOT)):{"sha256":sha(p),"bytes":p.stat().st_size} for p in ROOT.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.name not in ("artifact_manifest.json","Yb谱线计算与实验对照_复现数据.zip","delivery_receipts.json","bundle_build.log")}
    (ROOT/"artifact_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    bundle=ROOT/"Yb谱线计算与实验对照_复现数据.zip"
    with zipfile.ZipFile(bundle,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for name in manifest:z.write(ROOT/name,Path("yb_case")/name)
        z.write(ROOT/"artifact_manifest.json","yb_case/artifact_manifest.json")
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None
        for name,info in manifest.items():
            assert hashlib.sha256(z.read("yb_case/"+name)).hexdigest()==info["sha256"],name
    print(json.dumps({"bundle":bundle.name,"bytes":bundle.stat().st_size,"sha256":sha(bundle),"source_files":len(source_manifest),"artifacts":len(manifest)}))


if __name__=="__main__":main()
