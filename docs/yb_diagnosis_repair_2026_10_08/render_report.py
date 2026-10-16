"""Render in a separate uv environment; keep numerical dependencies fixed."""
import importlib.metadata
from pathlib import Path
import runpy
import json
import markdown
from weasyprint import HTML

ROOT=Path(__file__).resolve().parent
CSS=runpy.run_path(str(ROOT.parent/"rare_earth_yb_spectrum_2026_10_08/render_report.py"))["CSS"]
CSS=CSS.replace("谱线计算与实验对照", "谱线偏差定位与修复")
content=markdown.markdown((ROOT/"README.md").read_text(),extensions=["tables","fenced_code"])
document=f'<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8"><title>Yb⁺ 谱线偏差定位与修复</title><style>{CSS}</style><body>{content}</body></html>'
(ROOT/"Yb谱线偏差定位与修复.html").write_text(document)
HTML(string=document,base_url=str(ROOT)).write_pdf(ROOT/"Yb谱线偏差定位与修复.pdf")
(ROOT/"render_dependencies.json").write_text(json.dumps({n:importlib.metadata.version(n) for n in ["markdown","weasyprint"]},indent=2)+"\n")
