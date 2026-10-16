"""Render in a separate environment, leaving physics dependencies unchanged."""
from pathlib import Path
import importlib.metadata
import json
import runpy
import markdown
from weasyprint import HTML

ROOT=Path(__file__).resolve().parent
CSS=runpy.run_path(str(ROOT.parent/"rare_earth_yb_spectrum_2026_10_08/render_report.py"))["CSS"]
CSS=CSS.replace("谱线计算与实验对照","相关尾部与径向表示")
CSS+="\nbody {line-height:1.5;} p {margin:2mm 0;} table {font-size:8.3pt;}\n"


def main():
    content=markdown.markdown((ROOT/"README.md").read_text(),extensions=["tables","fenced_code"])
    document=f'<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8"><title>Yb⁺ 相关尾部与径向表示</title><style>{CSS}</style><body>{content}</body></html>'
    (ROOT/"Yb相关尾部与径向表示.html").write_text(document)
    HTML(string=document,base_url=str(ROOT)).write_pdf(ROOT/"Yb相关尾部与径向表示.pdf")
    (ROOT/"render_dependencies.json").write_text(json.dumps({n:importlib.metadata.version(n) for n in ("markdown","weasyprint")},indent=2)+"\n")


if __name__=="__main__":main()
