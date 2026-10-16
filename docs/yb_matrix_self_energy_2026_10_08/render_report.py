"""Render the completed scientific report in an isolated uv environment."""
from pathlib import Path
import importlib.metadata
import json
import runpy
import markdown
from weasyprint import HTML

ROOT = Path(__file__).resolve().parent
CSS = runpy.run_path(str(ROOT.parent/"rare_earth_yb_spectrum_2026_10_08/render_report.py"))["CSS"]
CSS = CSS.replace("谱线计算与实验对照", "矩阵自能与轨道响应")
CSS += "\ntable { break-inside: avoid; }\n"


def main():
    body = markdown.markdown((ROOT/"README.md").read_text(), extensions=["tables", "fenced_code"])
    document = f'<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8"><title>Yb⁺ 矩阵自能与轨道响应</title><style>{CSS}</style><body>{body}</body></html>'
    (ROOT/"Yb矩阵自能与轨道响应.html").write_text(document)
    HTML(string=document, base_url=str(ROOT)).write_pdf(ROOT/"Yb矩阵自能与轨道响应.pdf")
    (ROOT/"render_dependencies.json").write_text(json.dumps({x: importlib.metadata.version(x) for x in ("markdown", "weasyprint")}, indent=2)+"\n")


if __name__ == "__main__":
    main()
