"""Create a curated document bundle without scientific caches or private receipts."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def build():
    refs=ROOT/"references";refs.mkdir(exist_ok=True)
    snapshots=[]
    for name in ("ARCHITECTURE_CAPABILITIES.md","CRYSTAL_FIELD_SPEC.md"):
        src=REPO/"docs"/name
        dst=refs/name
        shutil.copyfile(src,dst)
        snapshots.append({"source_repo_path":f"docs/{name}","bundle_path":f"references/{name}","sha256":hashlib.sha256(src.read_bytes()).hexdigest(),"status":"dated offline copy; source repository guide is authoritative"})
    (refs/"README.md").write_text("# 离线参考快照\n\n本目录保留 2026-10-06 编写研究体系时所查 Atomx 文档的原样副本。它们用于离线阅读和检查当时的能力/表示约定；当前权威指南仍是原仓库 docs/ 中的对应文件，后续版本不由此副本代表。源路径与 SHA-256 见 ../source_index.json。\n\n- [能力指南](ARCHITECTURE_CAPABILITIES.md)\n- [晶场约定](CRYSTAL_FIELD_SPEC.md)\n\n这些文档内部的跨仓库链接需要原 Atomx 仓库，不是完整源码交付。\n",encoding="utf-8")
    text=(ROOT/"研究体系与案例设计.md").read_text(encoding="utf-8")
    section=text.split("## 12. 原始来源与软件入口",1)[1]
    sources=[]
    for line in section.splitlines():
        m=re.match(r"\| \[([^\]]+)\]\((https?://[^)]+)\) \| (.+?) \|$",line)
        if m:sources.append({"title":m[1],"url":m[2],"used_for":m[3],"checked_date":"2026-10-06","kind":"primary paper, author software documentation, official facility or platform"})
    assert len(sources)==23
    dump(ROOT/"source_index.json",{
        "date":"2026-10-06","sources":sources,"reference_snapshots":snapshots,
        "reference_policy":"Original synthesis and proposed setups; sources support motivation or method, not certification of this project's unexecuted cases.",
        "HEPS_annual_report_check":{"method":"downloaded official PDF and extracted facility table locally","source_sha256":hashlib.sha256(Path('/tmp/atomx-heps-report-2024.pdf').read_bytes()).hexdigest(),"values_confirmed":{"XAS_keV":[4.8,45],"high_resolution_spectroscopy_keV":[7,25],"nano_electronic_structure_keV":[0.1,2]},"status":"published 2024 table; not a live beamtime or operational capability guarantee"},
        "competition_contract":"platform entry verified; complete current rubric, submission format and deadline not verified; no submission performed",
    })
    layout=json.loads((ROOT/"layout_check.json").read_text())
    dump(ROOT/"review_record.json",{
        "main_pdf_pages":layout["main_pages"],"summary_pdf_pages":layout["summary_pages"],
        "all_case_ids_in_pdf":layout["all_case_ids_in_pdf"],"out_of_page_text_count":len(layout["out_of_page_text"]),
        "visual_review":"Main contact sheet, full first page and final reference page inspected; initial one-paragraph trailing page removed; final reference page fits.",
        "consistency_checks":json.loads((ROOT/"verification_report.json").read_text())["check_count"],"physics_scope":"three bounded software demonstrations only; no complete science case certified",
        "sealed_Ne_manifest_sha256":"bbe79f26b9c354f371c749a2002a607a79f9af6ab0e25dff1ad996eb4b2b15e2",
        "sealed_manifest_rechecked_unchanged":hashlib.sha256((REPO/'docs/xfel_tail_breit_pole_2026_10_06/manifest.json').read_bytes()).hexdigest()=="bbe79f26b9c354f371c749a2002a607a79f9af6ab0e25dff1ad996eb4b2b15e2",
        "documentation_disposition":"navigation updated; earlier Ne dossier retained as dated supporting material; no new physical insight promoted",
    })
    selected=[]
    for suffix in ("*.md","*.py","*.json","*.html","*.pdf","*.docx"):
        selected.extend(ROOT.glob(suffix))
    for folder in ("assets","foundation_results","references"):
        selected.extend(p for p in (ROOT/folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    selected=sorted(set(selected)-{ROOT/'artifact_manifest.json',ROOT/'package_check.json'},key=lambda p:p.relative_to(ROOT).as_posix())
    manifest={"format":"atomx-science-document-artifact-v1","date":"2026-10-06","scope":"curated documents and interface demonstrations; excludes delivery receipts, previews, Atomx full source, scientific caches and third-party article PDFs","files":[{"path":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in selected]}
    dump(ROOT/"artifact_manifest.json",manifest)
    delivery=ROOT/"delivery";delivery.mkdir(exist_ok=True)
    bundle=delivery/"Atomx_谱学研究体系_案例与代码.zip"
    with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as z:
        for p in selected+[ROOT/"artifact_manifest.json"]:z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(bundle) as z:
        assert z.testzip() is None
        for entry in manifest["files"]:
            assert hashlib.sha256(z.read(entry["path"])).hexdigest()==entry["sha256"]
        check={"passed":True,"member_count":len(z.namelist()),"manifested_files":len(manifest["files"]),"archive_bytes":bundle.stat().st_size,"archive_sha256":hashlib.sha256(bundle.read_bytes()).hexdigest(),"private_or_full_physics_data_included":False}
    dump(ROOT/"package_check.json",check)
    for src,dst in [("研究体系与案例设计.pdf","Atomx_原子分子晶体谱学与未来产业.pdf"),("研究体系与案例设计.docx","Atomx_原子分子晶体谱学与未来产业.docx"),("故事线摘要.pdf","Atomx_谱学故事线摘要.pdf")]:
        shutil.copyfile(ROOT/src,delivery/dst)
    print(json.dumps(check,ensure_ascii=False))


if __name__=="__main__":build()
